""" A rigid reimplementation of pypower dynamics' nine bus example. """
import numpy as np
import pandapower as pp
import pandapower.networks as nw
from itertools import chain
import pandas as pd


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


def main():

    net = get_net()
    print(net)
    check_unsupported(net)


    pp.runpp(net)
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    ybus += get_load_admittances(np.zeros_like(ybus), net)

    opt = {'t_sim': 2.0, 'fn': 60}
    # Map from bus to machine parameters.
    machs = {
        1: {'xdp': 0.0608, 'h': 23.64, 'ra': 0},
        2: {'xdp': 0.1198, 'h': 6.01, 'ra': 0},
        3: {'xdp': 0.1813, 'h': 3.01, 'ra': 0},
    }

    for bus, params in machs.items():
        vt0 = get_v_at_bus(net, bus)
        s0 = get_gen_s_at_bus(net, bus)
        ia0 = np.conj(s0/vt0)
        theta0 = np.angle(vt0)
        eq0 = vt0 + np.complex(0, params['xdp']) * ia0
        delta0 = np.angle(eq0)

        p0 = (1 / (params['xdp'] + 1j * params['ra'])) \
             * np.abs(vt0) * np.abs(eq0) * np.sin(delta0 - theta0)

        print()


if __name__ == '__main__':
    main()
