# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.stats import chi2

from pandapower.estimation.util import set_bb_switch_impedance, reset_bb_switch_impedance
from pandapower.estimation.results import eppci2pp
from pandapower.estimation.algorithm.base import (WLSAlgorithm,
                                                  WLSZeroInjectionConstraintsAlgorithm,
                                                  IRWLSAlgorithm)
from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.lp import LPAlgorithm
from Matrix_base import BaseAlgebra
from ppc_conversion import pp2eppci, _initialize_voltage, PQ_indices

ALGORITHM_MAPPING = {'wls': WLSAlgorithm,
                     'wls_with_zero_constraint': WLSZeroInjectionConstraintsAlgorithm,
                     'opt': OptAlgorithm,
                     'irwls': IRWLSAlgorithm,
                     'lp': LPAlgorithm}

# added by Nitty
def buildAdmittanceMat(net, init='flat', calculate_voltage_angles=False, zero_injection='aux_bus'):
    v_start, delta_start = _initialize_voltage(net, init, calculate_voltage_angles)
    se = StateEstimation(net)

    #for getting Ybus
    Ybus, Yf, Yt, ppc, eppci = se.buildAdmittanceMat(v_start, delta_start, calculate_voltage_angles, zero_injection)

    return [Ybus, Yf, Yt, ppc, eppci]

# added by Nitty
def create_hx(x, net, eppci):
    eppci.E = x.ravel()
    ekf_se = StateEstimation(net)
    h = ekf_se.h_at(eppci)
    return h

# added by Nitty
def create_jac_hx(x, net, eppci):
    eppci.E = x.ravel()
    ekf_se = StateEstimation(net)
    Jmeas = ekf_se.jac_hx_at(eppci)
    return Jmeas

# added by Nitty
def create_jac_inp(x, net, eppci):
    eppci.E = x.ravel()
    ekf_se = StateEstimation(net)
    Jinp = ekf_se.jac_inp_at(eppci)
    return Jinp

# added by Nitty
def eppci_conv(net, init='results', calculate_voltage_angles = False, zero_injection='aux_bus'):
    v_start, delta_start = _initialize_voltage(net, init, calculate_voltage_angles)
    net, ppc, eppci = pp2eppci(net, v_start=v_start, delta_start=delta_start,
                                    calculate_voltage_angles=calculate_voltage_angles,
                                    zero_injection=zero_injection)
    return [net, ppc, eppci]

class StateEstimation:
    """
    Any user of the estimation module only needs to use the class state_estimation. It contains all
    relevant functions to control and operator the module. Two functions are used to configure the
    system according to the users needs while one function is used for the actual estimation
    process.
    """
    def __init__(self, net):
        self.net = net

    def buildAdmittanceMat(self, v_start, delta_start, calculate_voltage_angles, zero_injection):
        """
        function to get eppci for calculating Ybus
        """
        self.net, ppc, eppci = pp2eppci(self.net, v_start=v_start, delta_start=delta_start,
                                        calculate_voltage_angles=calculate_voltage_angles,
                                        zero_injection=zero_injection)
        sem = BaseAlgebra(eppci)
        return [sem.Ybus, sem.Yf, sem.Yt, ppc, eppci]

    def h_at(self, eppci):
        """
        function to get measurements from given states
        """
        E = eppci.E
        sem = BaseAlgebra(eppci)
        h = sem.create_hx(E)
        h = h.reshape(-1,1)
        return h

    def jac_hx_at(self, eppci):
        """
        function to get measurements from given states
        """
        E = eppci.E
        sem = BaseAlgebra(eppci)
        Jmeas = sem.create_hx_jacobian(E)
        return Jmeas

    def jac_inp_at(self, eppci):
        """
        function to get measurements from given states
        """
        E = eppci.E
        sem = BaseAlgebra(eppci)
        Jinput = sem.create_inp_jacobian(E)
        return Jinput
