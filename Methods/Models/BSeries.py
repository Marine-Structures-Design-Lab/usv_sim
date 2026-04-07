'''
Wageningen B-Series Propeller Regressions for Open-Water Curves


This code implements the regression

Authors: Matt Collette

(c) 2024 Regents of the University of Michigan
'''

import numpy as np
import math
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def BSeries(PD, AE, AO, z, JStep=0.02):
    '''
    Start with base regression
    
    '''
    # Constants
    Kt_Coeff = np.array([0.00880496,
                         -0.20455400,
                         0.166351,
                         0.158114,
                         -0.14758100,
                         -0.48149700,
                         0.415437,
                         0.0144043,
                         -0.05300540,
                         0.0143481,
                         0.0606826,
                         -0.01258940,
                         0.0109689,
                         -0.13369800,
                         0.00638407,
                         -0.00132718,
                         0.168496,
                         -0.05072140,
                         0.0854559,
                         -0.05044750,
                         0.010465,
                         -0.00648272,
                         -0.00841728,
                         0.0168424,
                         -0.00102296,
                         -0.03177910,
                         0.018604,
                         -0.00410798,
                         -0.000606848,
                         -0.004981900,
                         0.0025983,
                         -0.000560528,
                         -0.001636520,
                         -0.000328787,
                         0.000116502,
                         0.000690904,
                         0.00421749,
                         5.65229E-05,
                         -0.001465640])
    Kt_s_exp = np.array([0,
                         1,
                         0,
                         0,
                         2,
                         1,
                         0,
                         0,
                         2,
                         0,
                         1,
                         0,
                         1,
                         0,
                         0,
                         2,
                         3,
                         0,
                         2,
                         3,
                         1,
                         2,
                         0,
                         1,
                         3,
                         0,
                         1,
                         0,
                         0,
                         1,
                         2,
                         3,
                         1,
                         1,
                         2,
                         0,
                         0,
                         3,
                         0])
    Kt_t_exp = np.array([0,
                         0,
                         1,
                         2,
                         0,
                         1,
                         2,
                         0,
                         0,
                         1,
                         1,
                         0,
                         0,
                         3,
                         6,
                         6,
                         0,
                         0,
                         0,
                         0,
                         6,
                         6,
                         3,
                         3,
                         3,
                         3,
                         0,
                         2,
                         0,
                         0,
                         0,
                         0,
                         2,
                         6,
                         6,
                         0,
                         3,
                         6,
                         3])
    Kt_u_exp = np.array([0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         0,
                         0,
                         0,
                         1,
                         2,
                         2,
                         2,
                         2,
                         2,
                         0,
                         0,
                         0,
                         1,
                         2,
                         2,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         2])
    Kt_v_exp = np.array([0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2])
    
    Kq_coeff = np.array([0.00379368,
                         0.00886523,
                         -0.032241,
                         0.00344778,
                         -0.0408811,
                         -0.108009,
                         -0.0885381,
                         0.188561,
                         -0.00370871,
                         0.00513696,
                         0.0209449,
                         0.00474319,
                         -0.00723408,
                         0.00438388,
                         -0.0269403,
                         0.0558082,
                         0.0161886,
                         0.00318086,
                         0.015896,
                         0.0471729,
                         0.0196283,
                         -0.0502782,
                         -0.030055,
                         0.0417122,
                         -0.0397722,
                         -0.00350024,
                         -0.0106854,
                         0.00110903,
                         -0.000313912,
                         0.0035985,
                         -0.00142121,
                         -0.00383637,
                         0.0126803,
                         -0.00318278,
                         0.00334268,
                         -0.00183491,
                         0.000112451,
                         -0.0000297228,
                         0.000269551,
                         0.00083265,
                         0.00155334,
                         0.000302683,
                         -0.0001843,
                         -0.000425399,
                         8.69243E-05,
                         -0.0004659,
                         5.54194E-05])
    Kq_s_exp = np.array([0,
                         2,
                         1,
                         0,
                         0,
                         1,
                         2,
                         0,
                         1,
                         0,
                         1,
                         2,
                         2,
                         1,
                         0,
                         3,
                         0,
                         1,
                         0,
                         1,
                         3,
                         0,
                         3,
                         2,
                         0,
                         0,
                         3,
                         3,
                         0,
                         3,
                         0,
                         1,
                         0,
                         2,
                         0,
                         1,
                         3,
                         3,
                         1,
                         2,
                         0,
                         0,
                         0,
                         0,
                         3,
                         0,
                         1])
    Kq_t_exp = np.array([0,
                         0,
                         1,
                         2,
                         1,
                         1,
                         1,
                         2,
                         0,
                         1,
                         1,
                         1,
                         0,
                         1,
                         2,
                         0,
                         3,
                         3,
                         0,
                         0,
                         0,
                         1,
                         1,
                         2,
                         3,
                         6,
                         0,
                         3,
                         6,
                         0,
                         6,
                         0,
                         2,
                         3,
                         6,
                         1,
                         2,
                         6,
                         0,
                         0,
                         2,
                         6,
                         0,
                         3,
                         3,
                         6,
                         6])
    Kq_u_exp = np.array([0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         0,
                         0,
                         0,
                         1,
                         1,
                         2,
                         2,
                         2,
                         2,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2,
                         2,
                         2])
    Kq_v_exp = np.array([0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         0,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         1,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2,
                         2])
    
    #Make the constant part of the Kt equation - these terms do not change with J
    Kt_leading_coeff = np.empty_like(Kt_Coeff)
    AeAo = AE/AO
    for i in range(len(Kt_Coeff)):
        Kt_leading_coeff[i] = Kt_Coeff[i] *  PD**Kt_t_exp[i] * AeAo**Kt_u_exp[i] * z**Kt_v_exp[i] 

    #Make the constant part of the Kq equation - these terms do not change with J
    Kq_leading_coeff = np.empty_like(Kq_coeff)
    for i in range(len(Kq_coeff)):
        Kq_leading_coeff[i] = Kq_coeff[i] *  PD**Kq_t_exp[i] * AeAo**Kq_u_exp[i] * z**Kq_v_exp[i]

    #Make the J-dependent part of the Kt equation - here we need to search for where Kt goes below zero again
    Kt_vals = []
    Kq_vals = []
    Effy_vals = []
    J_vals = []

    #We are going search up in J, should never really be more than 3
    num_steps_safety = 3./JStep
    count = 0

    while True:
        Kt = 0
        Kq = 0
        eta = 0
        J =  count * JStep
        for i in range(len(Kt_Coeff)):
            Kt += Kt_leading_coeff[i] * J**Kt_s_exp[i]
        
        for i in range(len(Kq_coeff)):
            Kq += Kq_leading_coeff[i] * J**Kq_s_exp[i]
        
        eta = J*Kt/(2*math.pi*Kq)


        if Kt < 0:
            delta = Kt_vals[count-1]/(Kt_vals[count-1] - Kt)
            J = J_vals[count-1] + delta * JStep
            Kt = 0
            eta = 0
            #Some trouble with interpolation when Kt is negative for Kq so grab last two good points for slope
            Kq_diff = Kq_vals[count-2] - Kq_vals[count-1]
            logger.debug('Kq Diff ' + str(Kq_diff))
            Kq = Kq_vals[count-1] - delta * Kq_diff
            Kt_vals.append(Kt)
            J_vals.append(J)
            Kq_vals.append(Kq)
            Effy_vals.append(eta)
            break

        Kt_vals.append(Kt)
        J_vals.append(J)
        Kq_vals.append(Kq)
        Effy_vals.append(eta)
        
        if count > num_steps_safety:
            logger.error('B-series propeller regression failed to find Kt < 0')
            logger.warn('List is incomplete, returning what we have')
            break

        count += 1
    
    #Make the lists into numpy arrays for future use
    Kt_return = np.array(Kt_vals)
    J_return = np.array(J_vals)
    Kq_return = np.array(Kq_vals)
    eta_return = np.array(Effy_vals)

    return (J_return, Kt_return, Kq_return, eta_return)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)    
    logger.debug('Demo for BSeries.py')
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    J, Kt, Kq, eta = BSeries(1, 0.8, 1.0, 4)
    plt.plot(J, Kt, label='Kt')
    plt.plot(J, 10*Kq, linestyle='dotted', label='10Kq')
    plt.plot(J, eta, linestyle='dashed', label=r'$\eta$')
    plt.xlabel ('Advance Ratio, J')
    plt.ylabel ('Kt, 10Kq, Open-Water Efficiency ' + r'$\eta$')
    plt.title ('B-Series Propeller Regression test - P/D = 1, BAR = 0.8, Blades = 4')
    plt.grid()

    plt.legend()
    plt.show()



    



