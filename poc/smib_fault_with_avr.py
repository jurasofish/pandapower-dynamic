""" A rigid reimplementation of pypower dynamics' two bus fault with 6th order machine.
The AVR has been disabled in pypower dynamics for this example.
"""
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
    """ Get a pp network consistent with pypower dynamic two bus case. """
    net = pp.create_empty_network(sn_mva=100)
    pp.create_bus(net, vn_kv=345, index=1)
    pp.create_bus(net, vn_kv=345, index=2)
    pp.create_sgen(net, 2, 12, 5)  # TODO: pull request to pp to make default q a float.
    pp.create_ext_grid(net, 1)
    zn = 345**2/net.sn_mva
    pp.create_line_from_parameters(net, 1, 2, 1, 0.01*zn, 0.0576*zn, 0, 1e6)
    pp.create_line_from_parameters(net, 1, 2, 1, 0.01*zn, 0.085*zn, 0, 1e6)

    return net


class ExtGrid:

    # def __init__(self, pp_bus, bus, fn, vt0, s0, xdp, h):
    def __init__(self, vt0, s0, p):

        p = munch.munchify(p)
        self.params = p

        ia0 = np.conj(s0/vt0)
        theta0 = np.angle(vt0)
        eq0 = vt0 + np.complex(0, p.xdp) * ia0
        delta0 = np.angle(eq0)
        omega0 = 1

        # Mechanical power.
        p.pm = (1 / p.xdp) * np.abs(vt0) * np.abs(eq0) * np.sin(delta0 - theta0)

        p.eq = np.abs(eq0)

        self.yg = 1 / (1j * p.xdp)

        self.init_state_vector = np.array([
            omega0,  # differential
            delta0,  # differential
        ])

    def get_i(self, t, x, vt):
        """ x is the same x as used in the DAE residual function. """
        delta = x[1]
        i_grid = self.params['eq'] * np.exp(1j * delta) / np.complex(0, self.params['xdp'])
        return i_grid

    def calc_diff(self, t, x, vt):
        omega = x[0]
        delta = x[1]

        p = np.abs(vt) * self.params['eq'] * np.sin(delta - np.angle(vt)) / self.params['xdp']

        omegadot_calc = 1/(2*self.params['h']) * (self.params['pm']/omega - p)

        deltadot_calc = 2 * np.pi * self.params['fn'] * (omega - 1)

        diff = np.array([
            omegadot_calc,
            deltadot_calc,
        ])

        return diff


class SauerPaiOrderSix:

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


class AndersonFouadOrderSix:

    def __init__(self, vt0, s0, p):

        p = munch.munchify(p)
        self.params = p

        ia0 = np.conj(s0/vt0)

        eq0 = vt0 + np.complex(p.ra, p.xq) * ia0
        delta0 = np.angle(eq0)
        psi0 = np.angle(ia0)   # inconsistent with wiki.openelectrical

        # convert currents to rotor reference frame
        id0 = np.abs(ia0) * np.sin(delta0 - psi0)
        iq0 = np.abs(ia0) * np.cos(delta0 - psi0)

        vfd0 = np.abs(eq0) + (p.xd - p.xq) * id0

        eqp0 = vfd0 - (p.xd - p.xdp) * id0
        eqpp0 = eqp0 - (p.xdp - p.xdpp) * id0

        edp0 = (p.xq - p.xqp) * iq0
        edpp0 = edp0 + (p.xqp - p.xqpp) * iq0

        vd0 = edpp0 + p.xqpp * iq0 - p.ra * id0
        vq0 = eqpp0 - p.xdpp * id0 - p.ra * iq0

        # Calculate active and reactive power
        p0 = vd0 * id0 + vq0 * iq0

        omega0 = 1

        # Mechanical power.
        self.params['pm'] = p0
        self.params['vfd'] = vfd0

        self.yg = (p.ra - 1j * 0.5 * (p.xdpp + p.xqpp)) / (p.ra ** 2 + (p.xdpp * p.xqpp))

        self.init_state_vector = np.array([
            omega0,
            delta0,
            eqp0,
            eqpp0,
            edp0,
            edpp0,
        ])

    def get_i(self, t, x, vt):
        """ x is the same x as used in the DAE residual function. """
        omega = x[0]
        delta = x[1]
        eqp = x[2]
        eqpp = x[3]
        edp = x[4]
        edpp = x[5]

        p = self.params

        vd = np.abs(vt) * np.sin(delta - np.angle(vt))
        vq = np.abs(vt) * np.cos(delta - np.angle(vt))

        id_ = (eqpp - p.ra / (p.xqpp * omega) * (vd - edpp) - vq / omega) / (p.xdpp + p.ra ** 2 / (omega * omega * p.xqpp))
        iq = (vd / omega + p.ra * id_ / omega - edpp) / p.xqpp

        # calculate machine current injection (norton equivalent current injection in network frame)
        in_ = (iq - 1j * id_) * np.exp(1j * delta)
        im = in_ + self.yg * vt

        return im

    def calc_diff(self, t, x, vt):
        omega = x[0]
        delta = x[1]
        eqp = x[2]
        eqpp = x[3]
        edp = x[4]
        edpp = x[5]

        p = self.params

        vd = np.abs(vt) * np.sin(delta - np.angle(vt))
        vq = np.abs(vt) * np.cos(delta - np.angle(vt))

        id_ = (eqpp - p.ra / (p.xqpp * omega) * (vd - edpp) - vq / omega) / (p.xdpp + p.ra ** 2 / (omega * omega * p.xqpp))
        iq = (vd / omega + p.ra * id_ / omega - edpp) / p.xqpp

        deqp = (p.vfd - (p.xd - p.xdp) * id_ - eqp) / p.td0p
        dedp = ((p.xq - p.xqp) * iq - edp) / p.tq0p
        deqpp = (eqp - (p.xdp - p.xdpp) * id_ - eqpp) / p.td0pp
        dedpp = (edp + (p.xqp - p.xqpp) * iq - edpp) / p.tq0pp

        pe = (vd + p.ra * id_) * id_ + (vq + p.ra * iq) * iq

        domega = 1/(2*p.h) * (p.pm/omega - pe)

        ddelta = 2 * np.pi * p.fn * (omega - 1)

        diff = np.array([
            domega,
            ddelta,
            deqp,
            deqpp,
            dedp,
            dedpp,
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

    global residual_counter
    residual_counter += 1
    # print(residual_counter)
    print(f't={t}')

    # Calculate bus voltages.
    currents = np.zeros(ybus_og.shape[0], dtype=ybus_og.dtype)
    base = 0  # Starting point for a machine's state vector within ultimate vector.
    for i, mach in enumerate(machs):  # Assume ordered dict.
        k = mach.init_state_vector.shape[0]
        bus = mach.params['bus']
        vt_given = np.complex(x[-4 + bus], x[-4+2 + bus])  # Given by solver
        x_sub = x[base:base+k]
        mach_i = mach.get_i(t, x_sub, vt_given)
        currents[bus] += mach_i
        base += k

    t_inv = time.perf_counter()
    ybus_inv = get_ybus_inv(t, ybus_og, ybus_states)
    # print(f'Inverse in {(time.perf_counter() - t_inv) * 1e6:.2f} us')
    v_calc = np.squeeze(ybus_inv @ currents)
    result[-4:-4+2] = v_calc.real - x[-4:-4+2]
    result[-4+2:] = v_calc.imag - x[-4+2:]

    # Now, get residuals
    base = 0  # Starting point for a machine's state vector within ultimate vector.
    for i, mach in enumerate(machs):  # Assume ordered dict.
        k = mach.init_state_vector.shape[0]
        bus = mach.params['bus']
        vt_given = np.complex(x[-4 + bus], x[-4+2 + bus])  # Given by solver
        x_sub = x[base:base+k]
        xdot_sub = xdot[base:base+k]
        diff_values = mach.calc_diff(t, x_sub, vt_given)

        result[base:base+k] = (diff_values - xdot_sub).copy()[:]

        base += k
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
            'mach_type': 'ext_grid',
            'ra': 0, 'xdp': 0.1, 'h': 99999
        },
        2: {
            'mach_type': 'anderson_fouad_six',
            'ra': 0.0, 'xd': 2.29, 'xq': 2.18, 'xdp':0.25,
            'xqp': 0.25, 'xdpp':0.18, 'xqpp': 0.18, 'td0p': 13.1979,
            'tq0p': 3.2423, 'td0pp': 0.0394, 'tq0pp': 0.1157, 'h': 5.8
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
        if mach_params['mach_type'] == 'ext_grid':
            mach = ExtGrid(vt0, s0, mach_params)
        elif mach_params['mach_type'] == 'sauer_pai_six':
            mach = SauerPaiOrderSix(vt0, s0, mach_params)
        elif mach_params['mach_type'] == 'anderson_fouad_six':
            mach = AndersonFouadOrderSix(vt0, s0, mach_params)
        else:
            raise ValueError(f'Unknown machine type: {mach_params["mach_type"]}')
        machs.append(mach)

    # Need to properly understand current injection equations.
    for mach in machs:
        bus = mach.params['bus']
        ybus_og[bus, bus] += mach.yg

    ybus_2 = np.zeros_like(ybus_og)
    ybus_2[1, 1] += 1e4 - 1j * 1e4
    ybus_3 = ybus_2 * -1
    ybus_states = [(1.0, ybus_2), (1.2, ybus_3)]

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
        old_api=False,
        max_steps=500000,
        max_step_size=1e-3,
    )

    solution = solver.solve(
        np.linspace(0, 8, 5000),
        init_x,
        init_xdot
    )

    # The "Grid"
    grid_vt = solution.values.y[:, -4+0] + 1j * solution.values.y[:, -4+2+0]
    grid_delta = solution.values.y[:, 1]

    # The machine we're interested in.
    gen1_vt = solution.values.y[:, -4+1] + 1j * solution.values.y[:, -4+2+1]

    # Data saved from pypower dynamics.
    df = pd.read_csv('./smib_fault_no_avr_pydn.csv')

    plt.figure()
    plt.plot(df['time'], df['GEN1:Vt'], '-', label='GEN1:Vt pydyn')
    plt.plot(solution.values.t, abs(gen1_vt), '-.', label='Gen Bus 2 Vt')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

"""
import pandapower as pp
#create empty net
net = pp.create_empty_network()

#create buses
b1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

#create bus elements
pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
pp.create_load(net, bus=b3, p_mw=0.1, q_mvar=0.05, name="Load")

#create branch elements
tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV", name="Trafo")
pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.1, name="Line",std_type="NAYY 4x50 SE")

pp.runpp(net, numba=False)
"""
