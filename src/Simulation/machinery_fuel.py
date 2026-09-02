"""
Machinery and fuel consumption models for autonomous vessels

Authors: Matt Collette, Rachel Mecca

(c) 2024 Regents of the University of Michigan
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
from scipy.interpolate import Rbf
from scipy.interpolate import CubicSpline
import math as math
import numpy as np
import logging
from src.Methods.Models.BSeries import BSeries
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AbstractMachineryModel:
    """
    Virtual base class for machinery models and fuel consumption calculations
    """

    def __init__(self):
        pass

    def runMachinery(self, speed, vessel, time):
        """
        Increments the machinery model by a time step and returns the fuel consumed
        internally, it may also update the machinery component state and health

        Parameters
        -----------
        :param speed:      speed of the vessel in knots
        :param weight:     weight of the vessel in kilonewtons (kN)
        :param time:       time in hours

        returns:
        --------
        fuel_consumed:  fuel consumed in kN
        """
        pass


class MachineryFuel(AbstractMachineryModel):
    """
    Simple constant-burn fuel machinery model for testing purposes
    Always returns 5 kN of fuel burned/hour
    """

    def __init__(self):
        super().__init__()

    def runMachinery(self, speed, vessel, time):
        fuel_weight = 5.0 * time
        return fuel_weight, None, None, None


class propulsionSimulationBase(AbstractMachineryModel):
    """
    A partially complete base class for propulsion simulation. Takes in different classes for
    resistance estimation, propulsion estimation, and engine fuel maps.  Provides base caclulations
    if methods aren't provided
    """

    def __init__(self, resistance_model=None, propulsion_model=None, fuel_map=None):
        """
        Constructor for the propulsion simulation base class

        """
        self.resistance_model = resistance_model
        self.propulsion_model = propulsion_model
        self.fuel_map = fuel_map

    def runMachinery(self, speed, vessel, time):
        """
        Run the machinery model for a given speed, weight, and time

        Parameters
        -----------------
        :param speed: float - speed of the vessel in knots
        :param weight: float - weight of the vessel in kN
        :param time: float - time to run the machinery in hours

        Returns:
        float - the amount of fuel burned in kN
        """

        resistance_in_kn, wake_fraction, thrust_deduction = (
            self.resistance_model.estimateResistance(speed, vessel)
        )
        eta_O, P_deliv, RPM = self.propulsion_model.estimatePropulsion(
            speed,
            resistance_in_kn,
            wake_fraction,
            thrust_deduction,
            vessel,
            self.resistance_model.rho,
        )

        sfc = self.fuel_map.getSFC(RPM, P_deliv, vessel)

        # Units are g/(kW*hr)*kW*hr = g
        fuel_mass = sfc * P_deliv * time

        # Need to move to kN as the model works in weight force - factor of 1000 for g to kg
        # second factor of 1000 is to get to kN from N
        fuel_weight = 9.81 * fuel_mass * 1.0e-6

        return (
            fuel_weight,
            sfc,
            (eta_O, P_deliv, RPM),
            (resistance_in_kn, wake_fraction, thrust_deduction),
        )


class resistanceModelBase:
    """
    Base class for resistance models.  Provides a simple interface for the propulsion simulation
    to call.  Should be subclassed for specific resistance models.
    """

    def __init__(self):
        pass

    def estimateResistance(self, speed, vessel):
        """
        Estimate the resistance for a given speed and weight
        ---------------------------------------------------

        Parameters
        -------------------------------------------
        :param speed: float - speed of the vessel in knots
        :param weight: float - weight of the vessel in kN

        Returns
        -------------------------------------------
        A tuple consisting of:
        float - the resistance in kN
        float - the wake fraction
        float - the thrust deduction
        """
        pass


class propulsionModelBase:
    """
    Base class for propulsion models, implements a simple fixed efficiency routine that can be
    overloaded by more complex classes
    """

    def __init__(self, eta_0):
        """
        Constructor for the propulsion model
        ------------------------------------------

        Parameters:
        ----------
        :param eta_0: float - constant open water efficiency of the propeller
        """
        self.eta_0 = eta_0

    def estimatePropulsion(
        self, speed, resistance_in_kn, wake_fraction, thrust_deduction, vessel, rho
    ):
        """
        Returns fixed propulsion efficiency and delivered power.  This is a simple model that can be overloaded.  RPM is returned as None.

        Parameters:
        -----------
        :param speed: float - speed in knots
        :param resistance_in_kn: float - bare hull resistance in kN
        :param wake_fraction: float - wake fraction
        :param thrust_deduction: f
        :param vessel: vessel class instance - not used here but could be used in subclasses
        :param rho: density of water(kg/m^3)

        Returns:
        -----------
        Tuple of floats in order:
        eta_O: float - open water efficiency of the propeller
        P_deliv: float - delivered power in kW
        RPM: float - propeller RPM
        """
        VMS = speed * 0.514444  # Convert to m/s
        thrust_needed = resistance_in_kn / (1.0 - thrust_deduction)
        P_deliv = thrust_needed * VMS / self.eta_0

        return (self.eta_0, P_deliv, None)


class engineModelBase:
    """
    Base class for engine models, implements a simple fixed SFC routine that can be
    overloaded by more complex classes
    """

    def __init__(self, sfc):
        """
        Constructor for the engine model

        Parameters:
        ----------
        :param sfc: float - constant specific fuel consumption, in g/(kW*hr)
        """
        self.sfc = sfc

    def getSFC(self, RPM, P_deliv, vessel):
        """
        Returns fixed specific fuel consumption.  This is a simple model that can be overloaded.

        Parameters:
        -----------
        :param RPM: float - propeller RPM
        :param P_deliv: float - delivered power to propeller.  Engine class should include any shaft/gear effy.
        :param vessel: vessel class instance - not used here but could be used in subclasses

        Returns:
        --------
        sfc: Specific fuel consumption in g/(kW*hr)
        """

        return self.sfc


class NPL_ResistanceModel(resistanceModelBase):
    """
    Implementation of the NPL resistance model as presented in Molland et al. Ship Resistance and Propulsion.

    ITTC 1978 frictional resistance, and air drag are also included.

    Bailey's NPL extension is used for thrust deduction and wake fraction.

    """

    def __init__(
        self, rho, rho_air, nu_seawater, C_A, res_file="Molland_NPL_Data_NumpyRead.csv"
    ):
        """
        Constructor for the NPL resistance model

        Parameters
        --------------------------------
        :param rho: float - density of seawater in kg/m^3
        :param rho_air: float - density of air in kg/m^3
        :param nu_seawater: float - kinematic viscosity of seawater in m^2/s
        :param C_A: float - correlation allowance for the resistance model
        :param res_file: string - file containing the resistance data

        Constructor does a lot of setup, include data loading to make resistance calls faster
        """

        self.rho = rho  # kg/m^3
        self.g = 9.81  # m/s^2
        self.rho_air = rho_air  # kg/m^3 Air density
        self.nu = nu_seawater  # Kinematic viscosity of SW, 15C
        # Pre-load the resistance data for the interpolation
        self.res_data = np.genfromtxt(res_file, delimiter=",")
        self.C_A = C_A  # Correlation allowance

        # Process the resistance data from the file
        # The first column is the Lvol
        self.Lvol = self.res_data[0, 1:]
        # The second columns are the b/t
        self.BT = self.res_data[1, 1:]
        # Then we need the number of speeds remembering two header rows
        self.numspeed_tested = self.res_data.shape[0] - 2

        # We need to build the interpolator for the resistance data once.
        # There is one model for each speed tested, which is stored in a list
        self.res_speeds = []  # Speed, in m/s for the interploting model
        self.res_models = []  # RBF models for each speed tested
        # Skipping header rows.
        for k in range(2, self.numspeed_tested + 2):
            i = k - 2

            # Do speed
            Vk = self.res_data[k, 0]

            # Do an RBF
            crvals = self.res_data[k, 1:]
            # We have to work around the missing data - lowest L/Vol^(1/3) not tested at higher speeds
            if self.res_data[k, 0] < 3.3:
                LvolUse = self.Lvol
                BTUse = self.BT
                crvalsUse = crvals
            else:
                LvolUse = self.Lvol[3:]
                BTUse = self.BT[3:]
                crvalsUse = crvals[3:]
            self.res_models.append(Rbf(LvolUse, BTUse, crvalsUse))
            self.res_speeds.append(Vk)

        ###This is data for wake fraction and thrust deduction
        self.wt_block_below_45 = CubicSpline([0.6, 1.4, 2.6], [0.0, -0.04, 0.0])
        self.wt_block_above_45 = CubicSpline([0.6, 1.4, 2.2], [0.08, -0.02, 0.04])
        self.t_block_below_45 = CubicSpline([0.6, 1.4, 2.6], [0.12, 0.07, 0.08])
        self.t_block_above_45 = CubicSpline([0.6, 1.4, 2.2], [0.15, 0.07, 0.06])

    def _checkBounds(self, BT, L_vol, VkLf):
        """
        Checks if vessel is in the bounds of the NPL dataset

        Parameters:
        -----------
        :param BT:     Beam to draft ratio
        :param L_Vol:  Slenderness ratio
        :param VkLf:   Velocity in knots/sqrt length in feet

        Returns:
        ---------
        Tuple in same order (Status, BT, L_Vol, VkLf) Status is true if all in bounds, false if not, with variables moved within bounds if necessary
        """
        status = True

        # Check that we are in the bounds of the regression
        if VkLf < 0.8:
            VkLf = 0.8
            status = False
            # logger.warning("Speed is below the regression range for resistance estimation - set to lower bound of 0.8")
        if VkLf > 4.1:
            VkLf = 4.1
            status = False
            logger.warning(
                "Speed is above the regression range for resistance estimation - set to upper bound of 4.1"
            )
        # B/T and slenderness ratio changed with speed this needs to be done by interpolation
        # At slower speeds, more heavily laden hulls were tested
        if VkLf < 3.2:
            if L_vol < 4.47:
                status = False
                L_vol = 4.47
                logger.warning(
                    "L/Vol is below the regression range for resistance estimation - set to lower bound of 4.47"
                )
            if L_vol > 8.3:
                status = False
                L_vol = 8.3
                logger.warning(
                    "L/Vol is above the regression range for resistance estimation - set to upper bound of 8.3"
                )
            BT_lower_lim = np.interp(
                L_vol,
                [4.47, 4.86, 5.23, 5.76, 6.59, 7.1, 8.3],
                [1.72, 2.19, 1.95, 1.93, 2.01, 2.51, 4.02],
            )
            BT_upper_lim = np.interp(
                L_vol,
                [4.47, 4.86, 5.23, 5.76, 6.59, 7.1, 8.3],
                [3.19, 4.08, 5.10, 6.8, 5.49, 6.87, 5.80],
            )
            if BT < BT_lower_lim:
                status = False
                BT = BT_lower_lim
                logger.warning(
                    "B/T is below the regression range for resistance estimation - set to lower bound "
                    + str(BT_lower_lim)
                )
            if BT > BT_upper_lim:
                status = False
                BT = BT_upper_lim
                logger.warning(
                    "B/T is above the regression range for resistance estimation - set to upper bound"
                    + str(BT_upper_lim)
                )
        else:  # At higher speeds, only the lighter hulls were tested
            if L_vol < 4.86:
                status = False
                L_vol = 4.86
                logger.warning(
                    "L/Vol is below the regression range for resistance estimation - set to lower bound of 4.86"
                )
            if L_vol > 8.3:
                status = False
                L_vol = 8.3
                logger.warning(
                    "L/Vol is above the regression range for resistance estimation - set to upper bound of 8.3"
                )
            BT_lower_lim = np.interp(
                L_vol,
                [4.86, 5.23, 5.76, 6.59, 7.1, 8.3],
                [2.19, 1.95, 1.93, 2.01, 2.51, 4.02],
            )
            BT_upper_lim = np.interp(
                L_vol,
                [4.86, 5.23, 5.76, 6.59, 7.1, 8.3],
                [4.08, 5.10, 6.8, 5.49, 6.87, 5.80],
            )
            if BT < BT_lower_lim:
                BT = BT_lower_lim
                status = False
                logger.warning(
                    "B/T is below the regression range for resistance estimation - set to lower bound "
                    + str(BT_lower_lim)
                )
            if BT > BT_upper_lim:
                BT = BT_upper_lim
                status = False
                logger.warning(
                    "B/T is above the regression range for resistance estimation - set to upper bound"
                    + str(BT_upper_lim)
                )

        return (status, BT, L_vol, VkLf)

    def estimateResistance(self, speed, vessel):

        # This class needs some vessel-specific characteristics to estimate resistance
        # Not all vessel classes have these
        L = 0
        B = 0
        WettedSurface = 0
        T = 0
        Vol = 0
        Windage_Area = 0
        C_Air = 0.0
        VMS = speed * 0.514444  # Convert to m/s
        try:
            hydrostatic_prop = vessel.getHydrostaticProperties()
            L = hydrostatic_prop["Length"]
            B = hydrostatic_prop["Beam"]
            WettedSurface = hydrostatic_prop["WettedSurface"]
            T = hydrostatic_prop["Draft"]
            Vol = hydrostatic_prop["Volume"]
            Windage_Area = hydrostatic_prop["WindageArea"]
            C_Air = hydrostatic_prop["AirDragCoefficient"]
        except:
            logger.error(
                "Vessel does not have hydrostatic properties needed for resistance estimation"
            )
            raise  # Push higher, as this is fatal to calculating resistance

        # We need a couple of different speeds factors here
        Vol_Froude_Number = VMS / (self.g * Vol ** (1.0 / 3.0)) ** 0.5
        # Old school velocity in knots over length in feet
        VkLf = speed / (L * 3.28084)

        # Resistance regression relies on slenderness ratio and B/t ratios calculate those
        L_vol = L / (Vol ** (1.0 / 3.0))
        BT = B / T

        status, BT, L_vol, VkLf = self._checkBounds(BT, L_vol, VkLf)

        # Build a speed vs. wave resistance spline from model test data
        wave_res_spline = []
        for i in range(len(self.res_speeds)):
            wave_res_spline.append(self.res_models[i](L_vol, BT))

        # Make the spline
        spline = CubicSpline(self.res_speeds, wave_res_spline)

        # interpolate the wave resistance at this speed
        Cr = spline(VkLf) / 1000.0  # Raw data was x1000 in textbook

        # Calculate the frictional resistance
        # Calculate ITTC 1957 friction line
        Rn = VMS * L / self.nu
        Cf = 0.075 / ((math.log10(Rn) - 2.0) ** (2.0))

        # Total hydro drag is residual(wave), friction, and correlation allowance
        Ct = Cr + Cf + self.C_A
        Rt_hydro = Ct * 0.5 * self.rho * WettedSurface * VMS**2.0 / 1000.0  # in Kn

        # Do air drag
        R_air = C_Air * 0.5 * Windage_Area * self.rho_air * VMS**2.0 / 1000.0  # in Kn

        # Total drag is sum of hydro and air
        Rt = Rt_hydro + R_air

        # We now need thrust deduction and wake fraction estimates
        # This is a from Table 8.4 in Molland et al. Ship Resistance and Propulsion
        # Very simple table given by Bailey from NPL data but values are very low anyhow.
        block = Vol / (L * B * T)

        if block < 0.45:
            wt = self.wt_block_below_45(block).item()
            t = self.t_block_below_45(block).item()
        else:
            wt = self.wt_block_above_45(block).item()
            t = self.t_block_above_45(block).item()

        return Rt, wt, t


class PropellerPropulsionModel(propulsionModelBase):

    def __init__(self, pitch, PD_ratio, AeAo_ratio, blades):
        """

        ----------------------------------
        Parameters
        ----------------------------------
        :param pitch:propeller pitch
        :param PD_ratio: pitch/propeller diameter
        :param AeAo ratio:Ae/Ao
        :param blades: number of blades
        :param diameter: propeller diameter(meters)
        :param J:Propeller advance coefficient
        :param Kt:Propeller thrust coefficient
        :param Kq: Propeller torque coefficent

        """
        self.pitch = pitch
        self.PD_ratio = PD_ratio
        self.AeAo_ratio = AeAo_ratio
        self.blades = blades
        self.diameter = pitch / PD_ratio
        self.J, self.Kt, self.Kq, self.eta = BSeries(self.PD_ratio, AeAo_ratio, 1.0, 4)

    def estimatePropulsion(
        self, speed, resistance_in_kn, wake_fraction, thrust_deduction, vessel, rho
    ):
        """
        Returns fixed propulsion efficiency, delivered power, and RPM
        by solving for the point at which the propeller thrust matches the provided
        resistance and thrust deduction.
        -----------------------------------------------------------------------------
         Parameters
         ------------------------------------------------------------------------------
         :param speed: float - speed in knots
         :param resistance_in_kn: float - bare hull resistance in kN
         :param wake_fraction: float - wake fraction
         :param thrust_deduction: f
         :param vessel: vessel class instance - not used here but could be used in subclasses
         :param rho: density of water(kg/m^3)
         -----------------------------------------------------------------------------
         Returns
         -----------------------------------------------------------------------------
         Tuple of floats in order:
         eta_O: float - open water efficiency of the propeller
         P_deliv: float - delivered power in kW
         RPM: float - propeller RPM
        """
        final_J = 0
        eta_O = 0
        goal_treq = resistance_in_kn / (1 - thrust_deduction)
        # convert from knots to meters/second
        speed_m_per_s = speed * 0.5144444
        Va = speed_m_per_s * (1 - wake_fraction)
        for i in range(1, len(self.J)):
            if self.J[i] != 0 and self.J[i - 1] != 0:
                n1 = Va / (self.diameter * self.J[i - 1])
                T1 = self.Kt[i - 1] * rho * (n1**2) * (self.diameter**4) / 1000
                n2 = Va / (self.diameter * self.J[i])
                T2 = self.Kt[i] * rho * (n2**2) * (self.diameter**4) / 1000

                if T1 < T2:
                    if T2 >= goal_treq and T1 <= goal_treq:
                        x = (i - 1) + (goal_treq - T1) / (T2 - T1)
                        final_J = self.J[i - 1] + (
                            (x - (i - 1)) * (self.J[i] - self.J[i - 1])
                        )
                        eta_O = self.eta[i - 1] + (
                            (x - (i - 1)) * (self.eta[i] - self.eta[i - 1])
                        )
                        break
                else:
                    if T2 <= goal_treq and T1 >= goal_treq:
                        x = (i - 1) + (goal_treq - T2) / (T1 - T2)
                        final_J = self.J[i - 1] + (
                            (x - (i - 1)) * (self.J[i] - self.J[i - 1])
                        )
                        eta_O = self.eta[i - 1] + (
                            (x - (i - 1)) * (self.eta[i] - self.eta[i - 1])
                        )
                        break

        RPM = (Va / (self.diameter * final_J)) * 60
        P_deliv = (goal_treq * speed_m_per_s) / eta_O
        return eta_O, P_deliv, RPM


class RPM_EngineModel(engineModelBase):
    def __init__(self, RPM, power, fuel, gear_ratio, shaft_eff=0.97):
        """
        RPM Engine Model Constructor
        --------------------------------------------

        Paramters
        ----------------------------------------------
        :param RPM: list of floats, list of propeller RPM values
        :param power: list of floats, list of propeller power values (bkW)
        :param fuel: list of floats, list of fuel consumed by the vessel (g/bKw*hr)
        :param gear_ratio: float
        :param shaft_eff: float, shafting efficency; Includes losses in shaft bearings (~1%) and gearbox (~2%).
        """
        self.RPM = RPM
        self.power = power
        self.fuel = fuel
        self.gear_ratio = gear_ratio
        self.shaft_eff = shaft_eff
        for i in range(0, len(self.RPM)):
            self.RPM[i] /= self.gear_ratio

    def getSFC(self, RPM, P_deliv, vessel):
        """
        Returns fixed specific fuel consumption.
        -------------------------------------------

        Parameters
        -----------------------------------------
         :param RPM: float, propeller RPM value
         :param P_deliv: float, power delivered
         :param Vessel: Vessel Object

         Returns
         ------------------------------------------
         float, cs fuel
        """
        brake_power = P_deliv / self.shaft_eff
        cs_power = CubicSpline(self.RPM, self.power)
        cs_fuel = CubicSpline(self.RPM, self.fuel)
        if RPM < np.min(self.RPM):
            # logger.error("RPM out of bounds (below min). Returning SFC at min RPM.")
            RPM = np.min(self.RPM)
        if RPM > np.max(self.RPM):
            logger.error("RPM out of bounds (above max). Returning SFC at max RPM.")
            RPM = np.max(self.RPM)

        if brake_power > cs_power(RPM):
            logger.error("The requested power is beyond what the engine can produce")
        result = cs_fuel(RPM)
        return float(result[0])


class RPM_Power_EngineModel:
    def __init__(self, RPM, power, fuel, gear_ratio, shaft_eff=0.97):
        """
        RPM Power Engine Model Constructor
        ------------------------------------------------------------------------------------------

        Parameters
        ------------------------------------------------------------------------------------------
        :param RPM: list of floats, list of propeller RPM values
        :param power: list of floats, list of propeller power values (bkW)
        :param fuel: list of floats, list of fuel consumed by the vessel (g/bKw*hr)
        :param gear_ratio: float
        :param shaft_eff: shafting efficiency; Includes losses in shaft bearings (~1%) and gearbox (~2%).
        """
        self.RPM = RPM
        self.power = power
        self.fuel = fuel
        self.gear_ratio = gear_ratio
        self.shaft_eff = shaft_eff
        for i in range(0, len(self.RPM)):
            self.RPM[i] /= self.gear_ratio

        # Flatten the RPM, power, and fuel arrays for Rbf
        self.RPM_flat, self.power_flat, self.fuel_flat = self.flatten_data(
            RPM, power, fuel
        )

    def flatten_data(self, RPM, power, fuel):
        """
        Flattens the RPM, power, and fuel arrays
        ------------------------------------------------

        Parameters
        ------------------------------------------------
        :param RPM: float, propeller RPM value
        :param power: float, propeller power value (bkW)
        :param fuel: float, fuel consumed by the vessel (g/bKw*hr)
        """
        RPM_flat = []
        power_flat = []
        fuel_flat = []
        for i, rpm in enumerate(RPM):
            for j, pwr in enumerate(power[i]):
                RPM_flat.append(rpm)
                power_flat.append(pwr)
                fuel_flat.append(fuel[i][j])
        return np.array(RPM_flat), np.array(power_flat), np.array(fuel_flat)

    def getSFC(self, RPM, P_deliv, vessel=None):
        """
        Returns fixed specific fuel consumption.
        -------------------------------------------

        Parameters
        -----------------------------------------
        :param RPM: float, propeller RPM value
        :param P_deliv: float, power delivered
        :param Vessel: Vessel Object

        Returns
        ------------------------------------------
        float, rbf fuel
        """
        brake_power = P_deliv / self.shaft_eff
        cs_power = CubicSpline(self.RPM, self.power)
        rbf_fuel = Rbf(
            self.RPM_flat, self.power_flat, self.fuel_flat, function="linear"
        )
        if RPM < np.min(self.RPM_flat):
            logger.error("RPM out of bounds (below min). Returning SFC at min RPM.")
            RPM = np.min(self.RPM)
        if RPM > np.max(self.RPM_flat):
            logger.error("RPM out of bounds (above max). Returning SFC at max RPM.")
            RPM = np.max(self.RPM)
        if brake_power > np.max(cs_power(RPM)):
            logger.error(
                "The requested power is beyond what the engine can produce. Returning SFC at max power"
            )
            brake_power = np.max(cs_power(RPM))

        return float(
            rbf_fuel(RPM, brake_power)
        )  # Convert the result to a single scalar value


def Engine_model_test(vessel):
    RPM1 = [800, 1000, 1300, 1600, 1800, 2000, 2100, 2300]
    RPM2 = [800, 1000, 1200, 1400, 1600, 1800]
    power1 = [[46], [89], [195], [364], [518], [711], [823], [1081]]
    power2 = [
        [43, 49, 56, 66],
        [84, 96, 109, 128],
        [146, 166, 188, 221],
        [231, 263, 298, 351],
        [346, 393, 445, 524],
        [492, 559, 634, 746],
    ]
    fuel1 = [[256.8], [230.8], [217.3], [212.1], [208.9], [208.3], [212.2], [222.4]]
    fuel2 = [
        [249.5, 241.4, 245.3, 222.4],
        [226.2, 224.0, 226.5, 211.5],
        [222.1, 214.2, 216.2, 208.5],
        [212.0, 207.5, 209.0, 208.9],
        [207.6, 205.8, 206.9, 203.9],
        [204.4, 201.8, 201.5, 201.0],
    ]

    engine1 = RPM_EngineModel(RPM1, power1, fuel1, 10)
    print(f"RPM Engine Model, RPM=100: {engine1.getSFC(100, 90, vessel)}")
    print(f"RPM Engine Model, RPM=170: {engine1.getSFC(170, 90, vessel)}")
    print(f"RPM Engine Model RPM=300(too large) {engine1.getSFC(300, 90, vessel)} ")
    print(
        f"RPM Engine Model, RPM=100(power too large): {engine1.getSFC(100, 92, vessel)}"
    )
    engine2 = RPM_Power_EngineModel(RPM2, power2, fuel2, 10)
    print(
        f"RPM Power Engine Model, RPM=100,Power 96 kW: {engine2.getSFC(100, 96, vessel)}"
    )
    print(
        f"RPM Power Engine Model, RPM=170, Power 500 kW: {engine2.getSFC(170, 500, vessel)}"
    )
    print(
        f"RPM Power Engine Model, RPM=170, Power 700(too big)  kW: {engine2.getSFC(170, 700, vessel)}"
    )
    print(
        f"RPM Power Engine Model, RPM=7(too small), Power 50 kW: {engine2.getSFC(7, 50, vessel)}"
    )
    print(
        f"RPM Power Engine Model, RPM=7(too small), Power 500kW(too big) kW: {engine2.getSFC(7, 5000, vessel)}"
    )


if __name__ == "__main__":
    # Create a resistance model using the NPL Series
    res_model = NPL_ResistanceModel(
        1025.0,
        1.225,
        1.1395e-6,
        0.0,
        "/home/remecca/usv_sim/src/Simulation/Molland_NPL_Data_NumpyRead.csv",
    )

    # Create a simple vessel only for testing - can't navigate/not a real vessel
    class TestVessel:
        def __init__(self):
            pass

        def getHydrostaticProperties(self):
            return {
                "Length": 45.0,
                "Beam": 7.0,
                "WettedSurface": 100.0,
                "Draft": 3.0,
                "Volume": 472.5,
                "WindageArea": 0.0,
                "AirDragCoefficient": 0.0,
            }

    vessel = TestVessel()

    # Get three drag values

    drag_25kts = res_model.estimateResistance(25.0, vessel)
    drag_20kts = res_model.estimateResistance(20.0, vessel)
    drag_10kts = res_model.estimateResistance(10.0, vessel)

    print("Drag at 25 kts: " + str(drag_25kts))
    print("Drag at 20 kts: " + str(drag_20kts))
    print("Drag at 10 kts: " + str(drag_10kts))

    # A propeller that should work D = 2.43m, Pitch = 3.40 Ae/Ao = 0.67
    # This was sized with a code that also corrects for Rn, while our currrnt model does not

    J, Kt, Kq, eta = BSeries(1.39, 0.67, 1, 4)

    plt.plot(J, Kt, label="Kt")
    plt.plot(J, 10 * Kq, linestyle="dotted", label="10Kq")
    plt.plot(J, eta, linestyle="dashed", label=r"$\eta$")
    plt.xlabel("Advance Ratio, J")
    plt.ylabel("Kt, 10Kq, Open-Water Efficiency " + r"$\eta$")
    plt.title("B-Series Propeller")
    plt.grid()

    plt.legend()
    plt.show()
    Rt, wt, t = res_model.estimateResistance(25.0, vessel)
    propeller = PropellerPropulsionModel(3.40, 1.39, 0.67, 4)
    eta, delivered_power, RPM = propeller.estimatePropulsion(
        25.0, Rt, wt, t, vessel, res_model.rho
    )
    print(f"open water efficiency at 25 kts: {eta}")
    print(f"delivered power at 25 kts: {delivered_power}")
    print(f"RPM at 25 kts: {RPM}")
    eta, delivered_power, RPM = propeller.estimatePropulsion(
        15.0, Rt, wt, t, vessel, res_model.rho
    )
    print(f"open water efficiency at 15 kts: {eta}")
    print(f"delivered power at 15 kts: {delivered_power}")
    print(f"RPM at 15 kts: {RPM}")
    eta, delivered_power, RPM = propeller.estimatePropulsion(
        10.0, Rt, wt, t, vessel, res_model.rho
    )
    print(f"open water efficiency at 10 kts: {eta}")
    print(f"delivered power at 10 kts: {delivered_power}")
    print(f"RPM at 10 kts: {RPM}")

    # Now we need a propuslion model sub-class - it should take in Pitch, P/D ratio, Ae/Ao, and number of blades.  It should generated the J, Kt. Kq, and eta curves above in its constructor.  It should then overload the estimatePropulsion method to return the correct values by solving for the point at which the propeller thrust matches the provided resistance and thrust deduction.

    Engine_model_test(vessel)
