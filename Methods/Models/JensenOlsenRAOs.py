# -*- coding: utf-8 -*-
"""
This method implements the simple RAOs 


(c) 2023 Regents of the University of Michigan

"""
import numpy as np
import matplotlib.pyplot as plt


def JensenHeavePitchRAO(L, B0, Cb, T, V, Beta, Omega):
    '''

    Parameters
    ----------
    :param L : Float
        Ovreall Length, meters.
    :param B0 : Float
        Breadth at waterline, meters.
    :param Cb : Float
        Block coefficient.
    :param T : Float
        Ovreall draft, meters.
    :param V : Float
        Speed, knots.
    :param Beta : Float
        Heading angle, 180 degrees is head seas.
    :param Omega : Numpy Array
        Ground-based frequencies to calculate RAO at.

    Returns
    -------
    Tuple of four numpy arrays, first is encounter frequencies, second is 
    the ground-based frequency, and third is the heave RAO values, fourth
    is the pitch RAO values 

    '''
    #Constants
    g = 9.81

    #Effective b
    B = B0 * Cb

    #Convert speed to m/s
    Vms = V * 0.514444

    #Convert heading to radians
    BetaRad = np.radians(Beta)

    #Calculate Wave Number
    K = Omega**2/g

    #Alpha is related to the speed/heading combo
    FroudeNumber = Vms / np.sqrt(g * L)
    Alpha = 1.0 - FroudeNumber*(K * L)**(0.5)*np.cos(BetaRad)

    #Calculate omega bar:
    OmegaBar = Omega * Alpha

    A = 2.0 * np.sin(np.square(OmegaBar) * B/(2.0 * g))*np.exp(-np.square    (OmegaBar) * T/g)

    #Forcing functions
    k_eff = np.abs(K * np.cos(BetaRad))
    Kappa = np.exp(-k_eff * T)

    script_f = np.power(np.square(1. - K*T) + np.square(np.square(A)/(K*B*np.power(Alpha,3.0))), 0.5)

    F = Kappa * script_f * (2.0/(k_eff *L))*np.sin(k_eff * L/2.0)

    G = Kappa * script_f * 24.0/(np.square(k_eff * L) * L) * (np.sin(k_eff * L/2.0) - k_eff * L/2.0 * np.cos(k_eff * L/2.0))

    print(A)
    print(Alpha)

    eta = 1./(np.power(np.square(1.0 - 2.0*K*T*np.square(Alpha)) +
                       np.square(np.square(A)/(K*B*np.square(Alpha)))
                       , 0.5))

    HeaveRAO = np.abs(eta * F)
    PitchRAO = np.abs(eta * G)

    return(HeaveRAO, PitchRAO)

Omega = np.linspace(0.1, 2, 100)
beta_vals = [180, 150, 120, 90]
for beta in beta_vals:
    Heave, Pitch = JensenHeavePitchRAO(88, 12.83, 0.419, 2.627, 0.70*(9.81*88)**(0.5),beta, Omega)
    plt.plot(Omega, Heave, label='Heave')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('RAO, Heave, m/m')
    plt.xlim(0, 2.0)
    plt.ylim(0, 1.5)
    plt.grid()
    plt.title('Heave RAO at a Heading of '+str(beta)+' degrees')
    plt.show()
    plt.clf()
    plt.plot(Omega, Pitch, label='Pitch')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('RAO, Pitch, rad/m?')
    plt.xlim(0, 2.0)
    plt.ylim(0, 1.5)
    plt.grid()
    plt.title('Pitch RAO at a Heading of '+str(beta)+' degrees')
    plt.show()
