""" A rigid reimplementation of pypower dynamics' nine bus fault with 6th order machines. """
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from itertools import chain
import pandas as pd
from scikits.odes.dae import dae
import matplotlib.pyplot as plt
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
import time
import munch


residual_counter = 0
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

    def __init__(self, vt0, s0, p):

        p = munch.munchify(p)
        self.params = p

        # internal variables
        p.gamma_d1 = (p.xdpp - p.xa) / (p.xdp - p.xa)
        p.gamma_d2 = (1 - p.gamma_d1) / (p.xdp - p.xa)
        p.gamma_q1 = (p.xqpp - p.xa) / (p.xqp - p.xa)
        p.gamma_q2 = (1 - p.gamma_q1) / (p.xqp - p.xa)

        ia0 = np.conj(s0/vt0)

        eq0 = vt0 + np.complex(p.ra, p.xq) * ia0
        delta0 = np.angle(eq0)
        psi0 = np.angle(ia0)   # inconsistent with wiki.openelectrical

        # convert currents to rotor reference frame
        id0 = np.abs(ia0) * np.sin(delta0 - psi0)
        iq0 = np.abs(ia0) * np.cos(delta0 - psi0)

        vd0 = np.abs(vt0) * np.sin(delta0 - np.angle(vt0))
        vq0 = np.abs(vt0) * np.cos(delta0 - np.angle(vt0))

        edp0 = vd0 - p.xqpp * iq0 + p.ra * id0 - (1 - p.gamma_q1) * (p.xqp - p.xa) * iq0
        eqp0 = vq0 + p.xdpp * id0 + p.ra * iq0 + (1 - p.gamma_d1) * (p.xdp - p.xa) * id0
        psid_pp0 = eqp0 - (p.xdp - p.xa) * id0
        psiq_pp0 = -edp0 - (p.xqp - p.xa) * iq0
        vfd0 = (
            eqp0 + (p.xd - p.xdp)
            * (id0 - p.gamma_d2 * psid_pp0
               - (1 - p.gamma_d1) * id0 + p.gamma_d2 * eqp0)
        )

        # calculate active and reactive power
        # somewhat inconsistent with openelectrical.
        p0 = vd0 * id0 + vq0 * iq0
        q0 = vq0 * id0 - vd0 * iq0

        omega0 = 1

        # Mechanical power.
        self.params['pm'] = p0
        self.params['vfd'] = vfd0
        self.params['id'] = id0
        self.params['iq'] = iq0

        self.yg = (p.ra - 1j * 0.5 * (p.xdpp + p.xqpp)) / (p.ra ** 2 + (p.xdpp * p.xqpp))

        self.init_state_vector = np.array([
            omega0,
            delta0,
            eqp0.real,
            eqp0.imag,
            psiq_pp0.real,
            psiq_pp0.imag,
            edp0.real,
            edp0.imag,
            psid_pp0.real,
            psid_pp0.imag,
        ])

    def get_i(self, t, x, vt):
        """ x is the same x as used in the DAE residual function. """
        omega = x[0]
        delta = x[1]
        eqp = np.complex(x[2], x[3])
        psiq_pp = np.complex(x[4], x[5])
        edp = np.complex(x[6], x[7])
        psid_pp = np.complex(x[8], x[9])

        p = self.params

        vd = np.abs(vt) * np.sin(delta - np.angle(vt))
        vq = np.abs(vt) * np.cos(delta - np.angle(vt))

        # ``id`` is a built-in function.
        id_ = (  # sorry
            (
                -vq / omega + p.gamma_d1 * eqp
                + (1 - p.gamma_d1) * psid_pp
                - p.ra / (omega * p.xqpp) * (
                    vd - p.gamma_q1 * edp
                    + (1 - p.gamma_q1) * psiq_pp
                )
             )
            / (p.xdpp + p.ra ** 2 / (omega * omega * p.xqpp))
        )
        iq = (
                (vd / omega + (p.ra * id_ / omega)
                 - p.gamma_q1 * edp + (1 - p.gamma_q1) * psiq_pp) / p.xqpp
        )

        # calculate machine current injection (norton equivalent current injection in network frame)
        in_ = (iq - 1j * id_) * np.exp(1j * delta)
        im = in_ + self.yg * vt

        return im

    def calc_diff(self, t, x, vt):
        omega = x[0]
        delta = x[1]
        eqp = np.complex(x[2], x[3])
        psiq_pp = np.complex(x[4], x[5])
        edp = np.complex(x[6], x[7])
        psid_pp = np.complex(x[8], x[9])

        p = self.params

        # Calculate terminal voltage in dq reference frame
        vd = np.abs(vt) * np.sin(delta - np.angle(vt))
        vq = np.abs(vt) * np.cos(delta - np.angle(vt))
        id_ = (  # sorry
                (
                        -vq / omega + p.gamma_d1 * eqp
                        + (1 - p.gamma_d1) * psid_pp
                        - p.ra / (omega * p.xqpp) * (
                                vd - p.gamma_q1 * edp
                                + (1 - p.gamma_q1) * psiq_pp
                        )
                )
                / (p.xdpp + p.ra ** 2 / (omega * omega * p.xqpp))
        )
        iq = (
                (vd / omega + (p.ra * id_ / omega)
                 - p.gamma_q1 * edp + (1 - p.gamma_q1) * psiq_pp) / p.xqpp
        )

        deqp = (
            (p.vfd
             - (p.xd - p.xdp) * (
                    id_ - p.gamma_d2 * psid_pp
                    - (1 - p.gamma_d1) * id_ + p.gamma_d2 * eqp
             ) - eqp
             ) / p.td0p
        )

        dedp = (
            (p.xq - p.xqp) * (
                 iq - p.gamma_q2 * psiq_pp
                 - (1 - p.gamma_q1) * iq - p.gamma_q2 * edp
            )
            - edp
        ) / p.tq0p

        dpsid_pp = (eqp - (p.xdp - p.xa) * id_ - psid_pp) / p.td0pp

        dpsiq_pp = (-edp - (p.xqp - p.xa) * iq - psiq_pp) / p.tq0pp

        pe = (vd + p.ra * id_) * id_ + (vq + p.ra * iq) * iq

        domega = 1/(2*p.h) * (p.pm/omega - pe)

        ddelta = 2 * np.pi * p.fn * (omega - 1)

        diff = np.array([
            domega,
            ddelta,
            deqp.real,
            deqp.imag,
            dpsiq_pp.real,
            dpsiq_pp.imag,
            dedp.real,
            dedp.imag,
            dpsid_pp.real,
            dpsid_pp.imag,
        ], dtype=np.float64)

        return diff


# https://stackoverflow.com/a/32655449/8899565
@cached(cache=LRUCache(maxsize=128), key=lambda t, *args, **kwargs: hashkey(t))
def get_ybus_inv(t, ybus_og, ybus_states, d=1e-6):
    ybus = ybus_og
    for event_t, event_ybus in ybus_states:
        factor = 1 / (1 + np.exp(np.clip(-(t - event_t) / d, -50, 50)))
        ybus = ybus + factor * event_ybus  # don't use in-place += as it mutates.

    ybus_inv = np.linalg.inv(ybus)
    return ybus_inv


# @profile
def residual(t, x, xdot, result, machs, ybus_og, ybus_states):
    """ Aggregate machine residual functions. """
    t1 = time.perf_counter()

    # Number of elements in each state vector.
    k = machs[0].init_state_vector.shape[0]

    global residual_counter
    residual_counter += 1
    # print(residual_counter)
    print(f't={t}')

    # Calculate bus voltages.
    currents = np.zeros(ybus_og.shape[0], dtype=ybus_og.dtype)
    for i, mach in enumerate(machs):  # Assume ordered dict.
        bus = mach.params['bus']
        vt_given = np.complex(x[-18 + bus], x[-18 + 9 + bus])  # Given by solver
        x_sub = x[k*i:k*i+k]
        mach_i = mach.get_i(t, x_sub, vt_given)
        currents[bus] += mach_i

    t_inv = time.perf_counter()
    ybus_inv = get_ybus_inv(t, ybus_og, ybus_states)
    # print(f'Inverse in {(time.perf_counter() - t_inv) * 1e6:.2f} us')
    v_calc = np.squeeze(ybus_inv @ currents)
    result[-18:-18+9] = v_calc.real - x[-18:-18+9]
    result[-18+9:] = v_calc.imag - x[-18+9:]

    # Now, get residuals
    for i, mach in enumerate(machs):  # Assume ordered dict.
        bus = mach.params['bus']
        vt_given = np.complex(x[-18+bus], x[-18+9+bus])  # Given by solver
        x_sub = x[k*i:k*i+k]
        xdot_sub = xdot[k*i:k*i+k]
        diff_values = mach.calc_diff(t, x_sub, vt_given)

        result[k*i:k*i+k] = (diff_values - xdot_sub).copy()[:]
    # print(f'Residual in {(time.perf_counter() - t1)*1e6:.2f} us')


def main():

    net = get_net()
    print(net)
    check_unsupported(net)

    pp.runpp(net)
    ybus_og = np.array(net._ppc["internal"]["Ybus"].todense())
    ybus_og += get_load_admittances(np.zeros_like(ybus_og), net)

    opt = {'t_sim': 5.0, 'fn': 60}
    # Map from pp_bus to machine.
    all_mach_params = {
        1: {
            'ra': 0.01, 'xa': 0.0, 'xd': 0.36, 'xq': 0.23, 'xdp': 0.15,
            'xqp': 0.15, 'xdpp': 0.1, 'xqpp': 0.1, 'td0p': 8.952,
            'tq0p': 5.76, 'td0pp': 0.075, 'tq0pp': 0.075, 'h': 8
        },
        2: {
            'ra': 0.0, 'xa': 0.0, 'xd': 1.72, 'xq': 1.66, 'xdp': 0.378,
            'xqp': 0.378, 'xdpp': 0.2, 'xqpp': 0.2, 'td0p': 5.982609,
            'tq0p': 4.5269841, 'td0pp': 0.0575, 'tq0pp': 0.0575, 'h': 4
        },
        3: {
            'ra': 0.0, 'xa': 0.0, 'xd': 1.68, 'xq': 1.61, 'xdp': 0.32,
            'xqp': 0.32, 'xdpp': 0.2, 'xqpp': 0.2, 'td0p': 5.5,
            'tq0p': 4.60375, 'td0pp': 0.0575, 'tq0pp': 0.0575, 'h': 2
        },
    }

    machs = []
    for pp_bus, mach_params in all_mach_params.items():

        mach_params = {
            **mach_params,
            'pp_bus': pp_bus,
            'bus': net._pd2ppc_lookups["bus"][pp_bus],
            'fn': opt['fn'],
        }

        vt0 = get_v_at_bus(net, pp_bus)
        s0 = get_gen_s_at_bus(net, pp_bus)
        mach = Machine(vt0, s0, mach_params)
        machs.append(mach)

    # Need to properly understand current injection equations.
    for mach in machs:
        bus = mach.params['bus']
        ybus_og[bus, bus] += mach.yg

    ybus_2 = np.zeros_like(ybus_og)
    ybus_2[6, 6] += 1e4 - 1j * 1e4
    ybus_3 = ybus_2 * -1
    ybus_states = [(1.0, ybus_2), (1.1, ybus_3)]

    # Define function here so it has access to outer scope variables.
    def residual_wrapper(t, x, xdot, result):
        return residual(t, x, xdot, result, machs, ybus_og, ybus_states)

    init_x = np.concatenate([mach.init_state_vector for mach in machs])
    init_x = np.concatenate([init_x,
                             np.squeeze(np.array(net._ppc["internal"]["V"])).real,
                             np.squeeze(np.array(net._ppc["internal"]["V"])).imag])
    init_xdot = np.zeros_like(init_x)

    a = np.zeros_like(init_x)
    residual_wrapper(0, init_x, init_xdot, a)
    print(a, '\n', np.sum(np.abs(a)))  # should be about zero.

    solver = dae(
        'ida',
        residual_wrapper,
        # compute_initcond='yp0',
        first_step_size=1e-18,
        atol=1e-6,
        rtol=1e-6,
        algebraic_vars_idx=list(range(6, 6+9*2)),
        old_api=False,
        max_steps=500000,
        max_step_size=1e-3,
    )

    solution = solver.solve(
        np.linspace(0, 5, 5000),
        init_x,
        init_xdot
    )

    gen1_vt = solution.values.y[:, -18+0] + 1j * solution.values.y[:, -18+9+0]
    gen1_delta = solution.values.y[:, 1]

    gen1_vd = np.abs(gen1_vt) * np.sin(gen1_delta - np.angle(gen1_vt))
    gen1_vq = np.abs(gen1_vt) * np.cos(gen1_delta - np.angle(gen1_vt))

    # Data saved from pypower dynamics.
    # Data saved from pypower dynamics.
    df = pd.read_csv('./pypower_dynamic_output.csv')

    print()

    plt.figure()
    plt.plot(df['time'], df['GEN1:Vd'], '-', label='GEN1:Vd pydyn')
    plt.plot(solution.values.t, gen1_vd, '-.', label='Gen1 Vd')

    plt.plot(df['time'], df['GEN1:Vq'], '-', label='GEN1:Vq pydyn')
    plt.plot(solution.values.t, gen1_vq, '-.', label='Gen1 Vq')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
