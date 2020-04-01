""" Demonstrate calculating bus voltages using ybus inverse and current injections. """

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from itertools import chain


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


def get_i(zeroed_i, net):
    """ Return current vector of bus current injections. """
    for r, r_res in chain(
            zip(net.gen.itertuples(), net.res_gen.itertuples()),
            zip(net.sgen.itertuples(), net.res_sgen.itertuples()),
            zip(net.ext_grid.itertuples(), net.res_ext_grid.itertuples()),
    ):
        pp_bus = r.bus  # Index of bus in pp dataframes.
        bus = net._pd2ppc_lookups["bus"][pp_bus]  # Index of bus in ybus.

        res_bus = net.res_bus.loc[pp_bus]
        v_bus = res_bus['vm_pu'] * cis(rad(res_bus['va_degree']))

        # S = VI*  =>  I = (S/V)*
        s = (r_res.p_mw + 1j * r_res.q_mvar) / net.sn_mva
        i = np.conj(s / v_bus)
        zeroed_i[bus] += i
    return zeroed_i


def main():

    # net = nw.case14()
    # net = nw.case1354pegase()
    net = nw.case3120sp()
    print(net)

    unsupported = False
    unsupported |= net.ward.shape[0] > 0
    unsupported |= net.xward.shape[0] > 0
    unsupported |= net.dcline.shape[0] > 0
    unsupported |= net.storage.shape[0] > 0
    unsupported |= net.load.const_z_percent.sum() > 0
    unsupported |= net.load.const_i_percent.sum() > 0
    if unsupported:
        print('Unsupported elements exist in the network')

    pp.runpp(net)
    ybus = np.array(net._ppc["internal"]["Ybus"].todense())
    v = np.squeeze(np.array(net._ppc["internal"]["V"]))  # solved bus voltages

    # Convert loads to admittances. More stable matrix inversion, and no
    # current injection required.
    ybus += get_load_admittances(np.zeros_like(ybus), net)

    # Check that we can calculate an I vector and use it to recalculate V.
    # This is really a check of numerical stability.
    i_ = np.squeeze(ybus @ v)
    v_ = np.squeeze(np.linalg.inv(ybus) @ i_)
    print('Reconstructed voltage is the same?', np.allclose(v, v_))

    i = np.zeros(ybus.shape[0], dtype=ybus.dtype)
    i += get_i(np.zeros_like(i), net)

    calculated_v = np.squeeze(np.linalg.inv(ybus) @ i)

    print("Is calculated V same as load flow solved V?",
          np.allclose(v, calculated_v))


if __name__ == '__main__':
    main()
