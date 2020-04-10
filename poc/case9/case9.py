""" A rigid reimplementation of pypower dynamics' nine bus example. """
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from itertools import chain
import pandas as pd
from scikits.odes.dae import dae
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def rad(theta):
    """ Degrees to radians """
    return theta * np.pi/180


def cis(theta):
    """ cos(theta) + j*sin(theta)"""
    return np.cos(theta) + 1j * np.sin(theta)


def get_load_admittances(ybus_zeroed, net):
    """ Return the ybus_zeroed matrix, populated with admittances of loads """
    for r, r_res in zip(net.load.itertuples(), net.res_load.itertuples()):
        pp_bus = r.bus  # Index of bus in pp dataframes.
        bus = net._pd2ppc_lookups["bus"][pp_bus]  # Index of bus in ybus.

        res_bus = net.res_bus.loc[pp_bus]
        v_bus = res_bus['vm_pu'] * cis(rad(res_bus['va_degree']))

        # S = VI*  =>  I = (S/V)*
        # S = |V|^2/Z*  =>  Z = |V|^2/S*  =>  Y = S*/|V|^2
        s = (r_res.p_mw + 1j * r_res.q_mvar) / net.sn_mva
        y = np.conj(s)/(np.abs(v_bus)**2)

        ybus_zeroed[bus, bus] += y

    return ybus_zeroed


def get_gen_s_at_bus(net, bus):
    """ Return pu complex power output of the generator at given bus. """
    for r, r_res in chain(
            zip(net.gen.itertuples(), net.res_gen.itertuples()),
            zip(net.sgen.itertuples(), net.res_sgen.itertuples()),
            zip(net.ext_grid.itertuples(), net.res_ext_grid.itertuples()),
    ):
        if r.bus == bus:
            s = (r_res.p_mw + 1j * r_res.q_mvar) / net.sn_mva
            return s
    raise ValueError(f'No generator at bus {bus}')


def get_v_at_bus(net, bus):
    """ Return pu complex voltage at given bus. """
    vm = net.res_bus.loc[bus, 'vm_pu']
    va = net.res_bus.loc[bus, 'va_degree']
    v = vm * cis(rad(va))
    return v


def check_unsupported(net):
    """ Raise exception if network contains unsupported elements. """
    unsupported = False
    unsupported |= net.ward.shape[0] > 0
    unsupported |= net.xward.shape[0] > 0
    unsupported |= net.dcline.shape[0] > 0
    unsupported |= net.storage.shape[0] > 0
    unsupported |= net.load.const_z_percent.sum() > 0
    unsupported |= net.load.const_i_percent.sum() > 0
    gen_buses = net.gen.bus.tolist() + net.ext_grid.bus.tolist()
    unsupported |= len(gen_buses) != len(set(gen_buses))
    if unsupported:
        print('Unsupported elements exist in the network')


def get_net():
    """ Get a pp network consistent with pypower dynamic nine bus case. """
    net = nw.case9()

    # Make consistent with the case in pypower dynamics.
    net.sn_mva = 100.0
    net.gen.vm_pu = 1.025
    net.ext_grid.vm_pu = 1.04
    pp.create_continuous_bus_index(net, 1)
    pp.reindex_buses(net, {
        5: 6,
        6: 9,
        7: 8,
        8: 7,
        9: 5,
    })
    net.bus = net.bus.sort_index()
    net.load = net.load.sort_values('bus').reset_index(drop=True)
    net.line = net.line.sort_values('from_bus').reset_index(drop=True)

    return net


class Machine:

    def __init__(self, pp_bus, bus, fn, vt0, s0, xdp, h):

        self.params = {
            'pp_bus': pp_bus,  # Index into pandapower bus dataframe.
            'bus': bus,  # Index into ybus matrix and v index.
            'fn': fn,  # Synchronous frequency (Hz).
            'xdp': xdp,
            'h': h,
            'pm': None,
            'eq': None,
        }
        self.signals = {
            'vt': [],
            'delta': [],
            'omega': [],
            'p': [],
        }

        ia0 = np.conj(s0/vt0)
        theta0 = np.angle(vt0)
        eq0 = vt0 + np.complex(0, self.params['xdp']) * ia0
        delta0 = np.angle(eq0)
        omega0 = 1

        # Mechanical power.
        self.params['pm'] = (1 / self.params['xdp']) \
            * np.abs(vt0) * np.abs(eq0) * np.sin(delta0 - theta0)

        self.params['eq'] = np.abs(eq0)

        self.init_state_vector = np.array([
            omega0,  # differential
            delta0,  # differential
            vt0.real,  # algebraic
            vt0.imag,  # algebraic
        ])

    def get_i(self, t, x, xdot):
        """ x is the same x as used in the DAE residual function. """
        delta = x[1]
        i_grid = self.params['eq'] * np.exp(1j * delta) / np.complex(0, self.params['xdp'])
        return i_grid

    def residual(self, t, x, xdot, vt_calc):
        omega = x[0]
        omegadot = xdot[0]
        delta = x[1]
        deltadot = xdot[1]
        vt = x[2] + 1j*x[3]

        p = np.abs(vt) * self.params['eq'] * np.sin(delta - np.angle(vt)) / self.params['xdp']

        omegadot_calc = 1/(2*self.params['h']) * (self.params['pm'] - p)

        deltadot_calc = 2 * np.pi * self.params['fn'] * (omega - 1)

        resid = np.abs(np.array([
            omegadot_calc - omegadot,  # omega
            deltadot_calc - deltadot,  # delta
            vt_calc.real - vt.real,  # v real
            vt_calc.imag - vt.imag,  # v imag
        ]))

        return resid


def main():

    net = get_net()
    print(net)
    check_unsupported(net)

    pp.runpp(net)
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    ybus += get_load_admittances(np.zeros_like(ybus), net)

    opt = {'t_sim': 2.0, 'fn': 60}
    # Map from pp_bus to machine.
    all_mach_params = {
        1: {'xdp': 0.0608, 'h': 23.64},
        2: {'xdp': 0.1198, 'h': 6.01},
        3: {'xdp': 0.1813, 'h': 3.01},
    }

    machs = [
        Machine(
            pp_bus=pp_bus,
            bus=net._pd2ppc_lookups["bus"][pp_bus],
            fn=opt['fn'],
            vt0=get_v_at_bus(net, pp_bus),
            s0=get_gen_s_at_bus(net, pp_bus),
            xdp=mach_params['xdp'],
            h=mach_params['h'],
        )
        for pp_bus, mach_params in all_mach_params.items()
    ]

    if [mach.params['bus'] for mach in machs] != [0, 1, 2]:
        raise ValueError

    # Need to properly understand current injection equations.
    for mach in machs:
        bus = mach.params['bus']
        ybus[bus, bus] += 1/(1j * mach.params['xdp'])
    ybus_inv = np.linalg.inv(ybus)

    resid_t = []
    resid_vals = []

    # Define function here so it has access to outer scope variables.
    def residual(t, x, xdot, result):
        """ Aggregate machine residual functions. """

        # Calculate bus voltages.
        currents = np.zeros(ybus.shape[0], dtype=ybus.dtype)
        for i, mach in enumerate(machs):  # Assume ordered dict.
            x_sub = x[4*i:4*i+4]
            xdot_sub = xdot[4*i:4*i+4]
            mach_i = mach.get_i(t, x_sub, xdot_sub)
            bus = mach.params['bus']
            currents[bus] += mach_i
        v_calc = np.squeeze(ybus_inv @ currents)
        _ = net  # Grab reference to net from outer scope to assist debugging.
        # print(abs(np.squeeze(np.array(net._ppc["internal"]["V"])) - v_calc).max())
        # print()

        # Now, get residuals
        for i, mach in enumerate(machs):  # Assume ordered dict.
            bus = mach.params['bus']
            vt_calc = v_calc[bus]
            x_sub = x[4*i:4*i+4]
            xdot_sub = xdot[4*i:4*i+4]
            resid = mach.residual(t, x_sub, xdot_sub, vt_calc)

            result[4*i:4*i+4] = resid[:]

        abs_resid = np.sum(np.abs(result))
        if t > 1e-3:
            resid_t.append(t)
            resid_vals.append(abs_resid)

        if abs(t - 0.32728) < 1e-3:
            print(result)
            print(abs_resid)

    init_x = np.abs(np.concatenate([mach.init_state_vector for mach in machs]))
    init_xdot = np.zeros_like(init_x)

    a = np.zeros_like(init_x)
    residual(0, init_x, init_xdot, a)
    print(a, '\n', np.sum(np.abs(a)))  # should be about zero.

    solver = dae(
        'ida',
        residual,
        # compute_initcond='yp0',
        first_step_size=1e-18,
        atol=1e-4,
        rtol=1e-4,
        algebraic_vars_idx=[2, 3, 6, 7, 10, 11],
        old_api=False,
        max_steps=5000,
        max_step_size=0.1,
    )

    solution = solver.solve(
        np.linspace(0, 2, 1000),
        init_x,
        init_xdot
    )

    # gen1_vt = abs(solution.values.y[:, 2] + 1j * solution.values.y[:, 3])
    # gen2_vt = abs(solution.values.y[:, 6] + 1j * solution.values.y[:, 7])
    gen3_vt = abs(solution.values.y[:, 10] + 1j * solution.values.y[:, 11])
    # plt.plot(solution.values.t[-30:], gen1_vt[-30:])
    # plt.plot(solution.values.t, gen1_vt)
    # plt.plot(solution.values.t, gen2_vt)
    plt.plot(solution.values.t, gen3_vt)

    plt.figure()
    plt.plot(solution.values.t, solution.values.y[:, 0])

    plt.figure()
    plt.plot(solution.values.t, solution.values.y[:, 1])

    plt.figure()
    plt.scatter(resid_t, resid_vals)

    plt.show()


if __name__ == '__main__':
    main()
