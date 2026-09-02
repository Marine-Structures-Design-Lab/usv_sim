"""
Vessel simulation engine for fleets of autonomous vessels

Authors: Sanjana Jain, Rachel Mecca, Matt Collette

(c) 2024 Regents of the University of Michigan
"""

import logging
import time
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path
from matplotlib.transforms import Affine2D
from geographiclib.geodesic import Geodesic
from Simulation.machinery_fuel import (
    propulsionSimulationBase,
    PropellerPropulsionModel,
    RPM_EngineModel,
    RPM_Power_EngineModel,
    NPL_ResistanceModel,
)
from Ocean.basespectra import Bretschneider
from Ocean.RAO_library import RAO_Library
from Simulation.ga_mission_planner import VesselProblem, Optimizer
from Wave_Predictor.Wave_Data_Calc import Wave_Data_Calc
from Simulation.db_manager import DatabaseManager

db_manager = DatabaseManager()
# set up geographiclib
geod = Geodesic.WGS84
# setup logger
logger = logging.getLogger(__name__)

logging.basicConfig(filename="simulation.log", filemode="w", level=logging.INFO)
# file paths
script_dir = Path(__file__).resolve().parent
weath_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "Ocean/weather_2_nc/1201_1231_lat30_lon30_fc0_ts6"
)
fc_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "Ocean/weather_2_nc/1215_1225_lat10_lon10_fc384_ts9"
)
weath_input_dir_lat35_lon155 = (
    script_dir.parent / "Ocean/weather_2_nc/1210_1231_lat30_40_lon150_160_fc0_ts6"
)
fc_input_dir_lat35_lon155 = (
    script_dir.parent / "Ocean/weather_2_nc/1218_1228_lat30_40_lon150_160_fc350_ts9"
)
model_path = "Wave_Predictor/torch_data/model.pth"
npl_path = script_dir.parent / "Simulation" / "Molland_NPL_Data_NumpyRead.csv"


class Waypoint:
    """
    A point in space and time that a vessel must travel through.
    """

    def __init__(self, lat, long):
        """
        Waypoint Constructor
        ---------------------

        Parameters
        -------------------
        :param lat: latitude of waypoint
        :param long: longitude of waypoint
        """
        self.lat = lat
        self.long = long


class Destination:
    """
    A waypoint with a goal arrival time and priority score
    """

    def __init__(self, waypoint, arrival_time, priority):
        """
        Destination Constructor
        ---------------------

        Parameters
        -------------------
        :param waypoint: waypoint object
        :param arrival_time: datetime object, goal time for the vessel to arrive at the waypoint
        :param priority: int, priority score for this waypoint
        """
        self.waypoint = waypoint
        self.arrival_time = arrival_time
        self.estimated_arrival_time = (
            arrival_time  # set based on GA mission planning prediction
        )
        self.priority = priority
        self.visited = False
        self.id = 0  # db id
        self.time_blacklist = (
            []
        )  # list of datetime objects, holds times when the weather is dangerous around this waypoint


class Mission:
    """
    A mission is a list of destinations that a vessel travels through
    """

    def __init__(self, destinations, speeds=[]):
        """
        MissionConstructor
        ---------------------

        Parameters
        -------------------
        :param destinations: a list of destination objects
        :param speeds: a list a floats, vessel speeds through each waypoint

        """
        self.destinations = destinations
        self.speeds = speeds
        if self.speeds == []:
            for i in range(len(destinations)):
                self.speeds.append(10)
        self.current_waypoint_index = (
            1  # the current destination the vessel is traveling to
        )
        self.total_priority = 0
        self.update_total_priority()  # sum of all priority values in a mission(used for optimal mission planning)

    def current_waypoint(self):
        """
        returns the waypoint the vessel is currently heading towards
        """
        if self.current_waypoint_index < len(self.destinations):
            return self.destinations[self.current_waypoint_index].waypoint
        return None

    def current_goal_time(self):
        """
        returns the goal time of the waypoint the vessel is currently heading towards
        """
        if self.current_waypoint_index < len(self.destinations):
            return self.destinations[self.current_waypoint_index].arrival_time
        return None

    def current_predicted_arrival_time(self):
        """
        returns the goal time of the waypoint the vessel is currently heading towards
        """
        if self.current_waypoint_index < len(self.destinations):
            return self.destinations[self.current_waypoint_index].estimated_arrival_time
        return None

    def current_priority(self):
        """
        returns the priority score of the waypoint the vessel is currently heading towards
        """
        if self.current_waypoint_index < len(self.destinations):
            return self.destinations[self.current_waypoint_index].priority
        return None

    def current_speed(self):
        """
        returns the planned speed towards the next waypoint
        """
        if self.current_waypoint_index < len(self.speeds):
            return self.speeds[self.current_waypoint_index]
        return None

    def advance_waypoint(self):
        """
        Summary
        -----------------------------------------------------------------
        Changes the vessel's current waypoint to the next one in the list.

        Returns
        -------------------------------------------------------------------
        Boolean: false if the current waypoint is the last one in the list, else true
        """
        self.destinations[self.current_waypoint_index].visited = True
        if self.current_waypoint_index < len(self.destinations) - 1:
            self.current_waypoint_index += 1
            return True

        return False

    def update_total_priority(self):
        """
        Updates total_priority variable to sum of priority scores of all destinations in destination list
        """
        self.total_priority = 0
        for dest in self.destinations:
            self.total_priority += dest.priority


class baseCalc:
    """
    Base class for calculation methods.
    It defines methods that should be overridden by subclasses
    """

    def tableOutputs(self):
        """
        Base-class for table output
        Should be overridden by subclasses to provide a table name and dictionry of output columns.
        """

        pass


class vesselBase:
    """
    Simple base class for vessel objects
    """

    def __init__(
        self,
        name,
        sim_name,
        start,
        end,
        mission,
        weather_input_dir,
        fc_input_dir=None,
        model_path=None,
        speed=20,
        heading=0,
        fuel_weight=100,
        speed_options=[],
    ):
        """
        Summary
        -----------------------
        vesselBase constructor

        Parameters
        ------------------------
        :param name: string, vessel name
        :param sim_name: string simulation name
        :param start: Destination Object, vessel starting point
        :param end: Destination Object, vessel ending point
        :param mission: mission object
        :param weather_input_dir: string, directory to nowcast wave data files
        :param forecast_input_dir: string, directory to forecast wave data files
        :param model_path: string. path to ml model for wave predictions
        :param speed: float, vessel's starting speed in knots
        :param heading: float, vessel's starting heading in degrees (0 = north, 90 = east)
        :param fuel_weight: float, vessel's starting fuel weight in kN
        :param speed_options: list of floats, all different speed settings for the vessel
        """
        self.name = name

        self.start = start
        self.start.visited = True
        self.end = end
        self.full_mission = mission
        self.speed = speed
        self.heading = heading
        self.fuel_weight = fuel_weight
        self.speed_options = speed_options
        self.rao_lib = RAO_Library()  # Rao Library object for RAO calculations

        self.model_path = model_path
        self.wave_calc = Wave_Data_Calc(
            weather_input_dir, fc_input_dir, model_path
        )  # Wave_Data_Calc object for calculating wave height, heading, and period
        self.opt_mission = None
        self.position = Waypoint(
            start.waypoint.lat, start.waypoint.long
        )  # vessel's current position
        self.methods = []
        self.active = True
        self.replan_count = 0
        self.vessel_id = db_manager.insert_vessel(
            self.name, sim_name, "cargo", 100, "good", 100
        )
        self.mission_id = db_manager.insert_mission(
            self.vessel_id,
            self.full_mission.total_priority,
            f"{self.name}'s mission",
            start.arrival_time,
        )
        self.ml_preds = (
            []
        )  # list of all (height, heading, period) tuples predicted by the simulation)
        self.actual_values = []  # list of all (height, heading, period)
        # self.count=0
        for dest in self.full_mission.destinations:
            dest.id = db_manager.insert_waypoint(
                dest.waypoint.lat, dest.waypoint.long, dest.priority
            )

    def addMethod(self, method):
        """
         Summary
         -----------------------------------------
         Adds base_calc sub classes to methods array

         Parameters
         --------------------------------------------
        :param method: base calc sub class
        """
        self.methods.append(method)

    def writeTimestep(self):
        """
        Summary
        ------------------------------------------------------------
        Gets each methods output, and appends the vessel ID to each dictionary
        so the calculation can be identified in post-processing.

        Returns
        -------------------------------------------------------------
        list of dictionaries
        """
        timestep_values = []
        for method in self.methods:
            name, val = method.provideValues()
            val["vessel_id"] = self.vessel_id
            timestep_values.append((name, val))

        return timestep_values

    def makeTables(self):
        """
        Summary
        ---------------------------------------------------------------------
        Creates a list of tables created from methods

        Returns
        -----------------------------------------------------------------------
        list of tuples of table names and dictionaries of column names
        """
        table_list = []
        for method in self.methods:
            method_name, output_dict = method.tableOutputs()
            db_manager.drop_table(method_name)
            output_dict["timestep"] = "text"
            db_manager.create_table(method_name, output_dict)
            print(method_name, output_dict)
            table_list.append((method_name, output_dict))
        return table_list

    def navigate(self, current_time, time_interval):
        """
        Summary
        ----------------------------------------------------------------------------------
        Moves vessel towards towards the next waypoint.
        Distance is based on time_interval.
        Increments waypoint in mission if vessel reaches waypoint during the time interval

        Parameters
        ----------------------------------------------------------------------------------
        :param current_time: datetime object, time the navigation starts at.

        :param time_interval: float, amount of time that will pass in this navigation

        Returns
        --------------------------------------------------------------------------------
        boolean: False if the vessel is at the last waypoint, else True
        """
        destination = Waypoint(
            self.opt_mission.current_waypoint().lat,
            self.opt_mission.current_waypoint().long,
        )
        goal_time = self.opt_mission.current_goal_time()
        estimated_time = self.opt_mission.current_predicted_arrival_time()
        g = geod.Inverse(
            self.position.lat, self.position.long, destination.lat, destination.long
        )
        self.heading = g["azi1"]
        distance = g["s12"] / 1852
        wave_height, wave_heading, wave_period = self.wave_calc.calc_WaveData(
            self.position, current_time, current_time
        )
        # position or destination??
        self.speed = self.opt_mission.current_speed()
        speed = self.adjust_speed(self.position, distance, current_time, current_time)
        # if can't get to next waypoint because of ocean conditions, replan entire mission
        if speed == -1:
            self.opt_mission.destinations[
                self.opt_mission.current_waypoint_index
            ].time_blacklist.append(current_time)
            self.opt_mission.destinations[
                self.opt_mission.current_waypoint_index
            ].time_blacklist.append(estimated_time)
            problem_dest = self.opt_mission.destinations[
                self.opt_mission.current_waypoint_index
            ]
            for dest in self.opt_mission.destinations:
                if dest != problem_dest:
                    g = geod.Inverse(
                        problem_dest.waypoint.lat,
                        problem_dest.waypoint.long,
                        dest.waypoint.lat,
                        dest.waypoint.long,
                    )
                    distance = g["s12"] / 1852
                    if distance < 10:
                        dest.time_blacklist.append(estimated_time)
                        dest.time_blacklist.append(current_time)
            check_time = current_time
            self.plan_Optimal_Mission(check_time)
            self.add_opt_mission_to_db(
                f" Detected danger at ({self.position.lat:.2f}, {self.position.long:.2f}) on {check_time}\n",
                check_time,
            )

            # self.add_opt_mission_to_db("Danger Detected",check_time)
            log_entry = (
                "\n===========================\n"
                f"[{current_time}] Replanning Triggered for {self.name}\n"
                "===========================\n\n"
                "Reason:\n"
                f"  Detected danger at ({self.position.lat:.2f}, {self.position.long:.2f}) on {check_time}\n\n"
                "Updated Mission\n"
                f"{self.mission_status_string()}\n"
            )
            logger.info(log_entry)

            return True
        else:
            self.speed = speed

        # Time elapsed in the current interval
        remaining_time_interval = time_interval

        while remaining_time_interval > 0:

            if self.is_at_waypoint(destination, remaining_time_interval):
                distance_to_waypoint = (
                    g["s12"] / 1852
                )  # Convert meters to nautical miles
                time_to_waypoint = distance_to_waypoint / self.speed  # Time in hours

                remaining_time_interval -= time_to_waypoint

                current_time += timedelta(hours=time_to_waypoint)  # convert to time

                logger.info(
                    f"{self.name} reached waypoint at ({destination.lat}, {destination.long}) at {current_time} with a fuel weight of {self.fuel_weight} kN\n"
                )
                self.calc_time_diff(current_time, goal_time, True)
                time_diff = self.calc_time_diff(current_time, estimated_time)
                if time_diff < -10:
                    logger.info(
                        f"{self.name}'s mission is {abs(time_diff)} hours behind predicted schedule, planning new mission"
                    )
                    self.plan_Optimal_Mission(current_time)

                    self.add_opt_mission_to_db(
                        f"{abs(time_diff)} hours behind predicted schedule",
                        current_time,
                    )
                    log_entry = (
                        "\n===========================\n"
                        f"[{current_time}] Replanning Triggered for {self.name}\n"
                        "===========================\n\n"
                        "Reason:\n"
                        f" mission is {abs(time_diff)} hours behind predicted schedule\n\n"
                        "Updated Mission\n"
                        f"{self.mission_status_string()}"
                    )
                    logger.info(log_entry)

                self.position = Waypoint(
                    destination.lat, destination.long
                )  # set current position to waypoint
                self.opt_mission.destinations[
                    self.opt_mission.current_waypoint_index
                ].estimated_arrival_time = current_time
                self.timestep(remaining_time_interval, current_time)

                if not self.opt_mission.advance_waypoint():
                    print(f"{self.name}  completed the mission")
                    print(f"{self.fuel_weight} kN of fuel remaining")
                    mission_status = self.mission_status_string()
                    log_entry = f"\nFinal Mission For {self.name}\n" f"{mission_status}"
                    logger.info(log_entry)

                    return False

                # calculate heading towards next waypoint
                destination = Waypoint(
                    self.opt_mission.current_waypoint().lat,
                    self.opt_mission.current_waypoint().long,
                )
                g = geod.Inverse(
                    self.position.lat,
                    self.position.long,
                    destination.lat,
                    destination.long,
                )
                self.heading = g["azi1"]
                self.speed = self.opt_mission.current_speed()

            else:
                pos = self.calculate_new_position(
                    self.position, self.speed, remaining_time_interval
                )
                self.position.lat = pos[0]
                self.position.long = pos[1]
                self.timestep(remaining_time_interval, current_time)
                remaining_time_interval = 0
        # update log datatable
        db_manager.insert_log(
            self.vessel_id,
            self.mission_id,
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            self.position.lat,
            self.position.long,
            self.speed,
            "Good",
            self.fuel_weight,
            float(wave_height),
            float(wave_period),
        )
        logger.info(
            f"{current_time+timedelta(hours=time_interval)}: {self.name} at ({self.position.lat},{self.position.long}) with speed {self.speed} knots \n"
        )  # and fuel weight {self.fuel_weight}")
        return True

    def adjust_speed(self, startpoint, distance, start_time, current_time):
        """
        Summary
        -----------------------------------------------------------
        Uses ocean data, waypoint, and current time to adjust speed

        Parameters
        -----------------------------------------------------------
        :param startpoint: waypint object, staring postion of path
        :param distance: float, distance in nautical miles vessel will travel
        :param start_time: datetime object, start time of the simulation
        :param current_time: datetime object, current time of the simulation

        Returns
        -----------------------------------------------------------
        float: the new speed
        """
        speed = self.speed
        num_checks = 4
        increment = distance / num_checks
        position = startpoint

        for i in range(1, num_checks + 1):
            # Compute next position
            g = geod.Direct(position.lat, position.long, self.heading, increment * 1852)
            position = Waypoint(g["lat2"], g["lon2"])

            # Get predicted wave data

            wave_height, wave_heading, wave_period = self.wave_calc.calc_WaveData(
                position, start_time, current_time
            )
            # self.count+=1
            """
         if (start_time!= current_time) and self.count%500==0:
          wave_height_act, wave_heading_act, wave_period_act = self.wave_calc.calc_WaveData(position, current_time, current_time)
          self.actual_values.append((wave_height_act, wave_heading_act, wave_period_act))
          self.ml_preds.append((wave_height, wave_heading, wave_period))
          """

            rel_heading = self.calc_relative_heading(wave_heading, self.heading)

            # Cache wave model if not already in library
            wave_model_key = (round(wave_height, 2), round(wave_period, 2))
            if wave_model_key not in self.rao_lib.model_dict:
                model = Bretschneider(
                    "wave spectrum", wave_model_key[0], wave_model_key[1], 0.01, 2, 500
                )
                self.rao_lib.add_wave_model(wave_model_key[0], wave_model_key[1], model)
            else:
                model = self.rao_lib.model_dict[wave_model_key]

            # Find safe speed for this segment
            while speed >= 1:
                rao_key = (
                    round(wave_height, 2),
                    round(wave_period, 2),
                    round(speed, 2),
                    round(rel_heading, 1),
                )

                if rao_key not in self.rao_lib.RAO_dict:
                    self.rao_lib.add_rao(
                        rao_key[0], rao_key[1], rao_key[2], rao_key[3], model
                    )

                rao = self.rao_lib.RAO_dict[rao_key]
                heave, pitch = rao.JensenHeavePitchRAO(45.0, 8.910, 0.61, 3.0)
                response_spectra = rao.generate_response_spectra(heave)

                integration = np.sqrt(np.trapezoid(response_spectra, x=rao.freq))
                if speed != 0:
                    current_time += timedelta(hours=increment / speed)

                if integration <= 1.5 and wave_height > 8:
                    print(f"error")
                    """
               rows = [
                ['Heave'] + heave.tolist(),
                ['Frequency']+rao.freq.tolist(),
                ['Amplitude'] +rao.amplitude.tolist(),
                ['Wave Spectrum'] + rao.spectrum.tolist(),
                ['Response Spectrum'] + response_spectra.tolist()
                
               ]

               # Write to CSV
               with open('rao.csv', 'w', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerows(rows)
            """
                if integration > 1.5:
                    speed /= 2

                else:
                    break
        # print(f"Forecast: wave_height={wave_height}, integration={integration}, speed={speed}")
        if speed < 1:

            return -1

        return speed

    def calculate_new_position(self, current_pos, speed, time_elapsed):
        """
        Summary
        ------------------------------------------------------------------
        Calcuates new vessel position based on current heading, speed, and elapsed time.

        Parameters
        -------------------------------------------------------------------------------
        :param current_pos: waypoint object, starting position
        :param speed: float, speed in knots
        :param time_elapsed: float, number of hours the vessel will travel

        Returns
        ------------------------------------
        tuple of floats: (latitude, longitude)
        """
        dist_traveled = speed * time_elapsed
        dist_traveled *= 1852  # converts from nautical miles to meters
        position = geod.Direct(
            current_pos.lat, current_pos.long, self.heading, dist_traveled
        )
        return position["lat2"], position["lon2"]

    def is_at_waypoint(self, waypoint, time_interval):
        """
        Summary
        ----------------------------------------------------------------------------
        Checks if the vessel will arrive at the target waypoint in the time interval

        Parameters
        ----------------------------------------------------------------------------
        :param waypoint: destination object, goal waypoint
        :param time_interval: float, amount of time the vessel has to travel

        Returns
        ---------------------------------------------------------------------------
        boolean: true if the vessel's distance from the waypoint is less than
        the distance the vessel will travel during the time interval and false if not
        """
        g = geod.Inverse(
            self.position.lat, self.position.long, waypoint.lat, waypoint.long
        )

        return g["s12"] / 1852 < self.speed * time_interval

    def calc_time_diff(self, current_time, goal_time, verbose=False):
        """
        Summary
        ----------------------------------------------------------------
        Calculates the tiem difference between current time and goal time
        prints out a message depending on the calculation.

        Parameters
        ---------------------------------------------------------------
        :param current_time: datetime object, current time the simulation is at
        :param goal_time: datetime object, goal time to arrive at a destination
        :param verbose: bool, if true will print out messages

        Returns
        ----------------------------------------------------------------
        Float: time difference in hours
        """
        time_difference = abs((current_time - goal_time).total_seconds())
        hrs_difference = int(time_difference // 3600)
        min_difference = int((time_difference % 3600) // 60)
        sec_difference = int(time_difference % 60)
        if verbose:
            if current_time > goal_time:
                print(
                    f"{self.name} arrived {hrs_difference} hours {min_difference} minutes and {sec_difference} seconds late"
                )
            elif current_time < goal_time:
                print(
                    f"{self.name} arrived {hrs_difference} hours {min_difference} minutes and {sec_difference} seconds early"
                )
            else:
                print(f"{self.name} arrived on time")
        return time_difference / 3600

    def calc_relative_heading(self, wave_heading, vessel_heading):
        """
        Summary
        ------------------------------------------------------------
        calculates relative heading (0-360) between a given wave and vessel heading

        Parameters
        --------------------------------------------------------------------------
        :param wave_heading: float, heading angle of wave
        :param vessel_heading: float,heading angle of vessel

        Returns
        --------------------------------------------------------------------------
        float(0-360), relative heading between wave and vessel
        """
        heading = 180 + vessel_heading - wave_heading
        if heading < 0:
            heading += 360
        return heading

    def plan_Optimal_Mission(self, current_time):
        """
        Summary
        ------------------------------------------------------------------
        Calls ga to plan an optimal mission.

        Parameters
        --------------------------------------------------------------------
        :param current_time: datetime object, current time of the simulation


        """
        self.opt_mission = self.full_mission

    def add_opt_mission_to_db(self, reason, current_time):
        """
        Summary
        ----------------------------------------------------
        Adds mission repaln to database

        Parameters
        ----------------------------------------------------
        :param reason: string, reason for mission replan
        :param current_time: datetime object, time of mission replan
        """
        for order, dest in enumerate(self.opt_mission.destinations):
            db_manager.insert_mission_waypoint(
                self.mission_id,
                dest.id,
                self.replan_count,
                order,
                dest.arrival_time,
                dest.estimated_arrival_time,
            )
        db_manager.insert_replan(
            self.replan_count,
            self.vessel_id,
            self.mission_id,
            current_time,
            self.position.lat,
            self.position.long,
            self.opt_mission.total_priority,
            len(self.opt_mission.destinations),
            reason,
        )

    def mission_status_string(self):
        """
        Summary
        -------------------------------------------------------
        Creates a string listing all the destinations in the vessel's mission in order

        Returns
        ---------------------------------------------------------
        string, lists all the destinations in the vessel's mission in order
        """
        if self.opt_mission == None:
            return
        counter = 0
        log_entry = "\n"
        for dest in self.opt_mission.destinations:
            wp = f"WP {counter}"
            if counter == 0:
                wp = "start"
            if counter == (len(self.opt_mission.destinations) - 1):
                wp = "end"
            if dest.visited:
                log_entry += f"{wp}:({dest.waypoint.lat},{dest.waypoint.long}), Arrival Time: {dest.estimated_arrival_time}\n"
            else:
                log_entry += f"{wp}:({dest.waypoint.lat},{dest.waypoint.long}), ETA: {dest.estimated_arrival_time}\n"

            counter += 1
        return log_entry

    def timestep(self, time_interval, current_time):
        # This should be overridden by subclasses to do whatever calculation is necessary
        pass


class fuelCalc(baseCalc):
    """
    baseCalc sub class for fuel data table
    """

    def __init__(self, vessel):
        """
        fuelCalc Constructor
        ---------------------
        Parameters
        --------------------
        :param vessel: vessel object
        """
        self.vessel = vessel

    def tableOutputs(self):
        return ("fuel", {"vessel_id": "integer", "name": "text", "fuel_weight": "real"})

    def provideValues(self):
        return (
            "fuel",
            {
                "vessel_id": self.vessel.vessel_id,
                "name": self.vessel.name,
                "fuel_weight": self.vessel.fuel_weight,
            },
        )


class resistanceCalc(baseCalc):
    """
    baseCalc sub class for resistance data table
    """

    def __init__(self, vessel):
        """
        resistanceCalc Constructor
        ---------------------
        Parameters
        --------------------
        :param vessel: vessel object
        """
        self.vessel = vessel

    def tableOutputs(self):
        return (
            "resistance_model",
            {
                "vessel_id": "integer",
                "name": "text",
                "speed": "real",
                "resistance_in_kn": "real",
                "wake_fraction": "real",
                "thrust_deduction": "real",
            },
        )

    def provideValues(self):
        return (
            "resistance_model",
            {
                "vessel_id": self.vessel.vessel_id,
                "name": self.vessel.name,
                "speed": self.vessel.speed,
                "resistance_in_kn": self.vessel.resistance_in_kn,
                "wake_fraction": self.vessel.wake_fraction,
                "thrust_deduction": self.vessel.thrust_deduction,
            },
        )


class propulsionCalc(baseCalc):
    """
    baseCalc sub class for propulsion model data table
    """

    def __init__(self, vessel):
        """
        propulsionCalc Constructor
        ---------------------
        Parameters
        --------------------
        :param vessel: vessel object
        """
        self.vessel = vessel

    def tableOutputs(self):
        return (
            "propulsion_model",
            {
                "vessel_id": "integer",
                "name": "text",
                "speed": "real",
                "resistance_in_kn": "real",
                "wake_fraction": "real",
                "thrust_deduction": "real",
                "eta_O": "real",
                "rpm": "real",
                "p_deliv": "real",
            },
        )

    def provideValues(self):
        return (
            "propulsion_model",
            {
                "vessel_id": self.vessel.vessel_id,
                "name": self.vessel.name,
                "speed": self.vessel.speed,
                "resistance_in_kn": self.vessel.resistance_in_kn,
                "wake_fraction": self.vessel.wake_fraction,
                "thrust_deduction": self.vessel.thrust_deduction,
                "eta_O": self.vessel.eta_O,
                "rpm": self.vessel.RPM,
                "p_deliv": self.vessel.P_deliv,
            },
        )


class engineCalc(baseCalc):
    """
    baseCalc sub class for engine model data table
    """

    def __init__(self, vessel):
        """
        engineCalc Constructor
        ---------------------
        Parameters
        --------------------
        :param vessel: vessel object
        """
        self.vessel = vessel

    def tableOutputs(self):
        return (
            "engine_model",
            {
                "vessel_id": "integer",
                "name": "text",
                "rpm": "real",
                "p_deliv": "real",
                "fuel_burned": "real",
            },
        )

    def provideValues(self):
        return (
            "engine_model",
            {
                "vessel_id": self.vessel.vessel_id,
                "name": self.vessel.name,
                "rpm": self.vessel.RPM,
                "p_deliv": self.vessel.P_deliv,
                "fuel_burned": self.vessel.fuel_burned,
            },
        )


class vesselFuel(vesselBase):
    """
    sub class of vesselBase adds fuel calculations
    """

    def __init__(
        self,
        name,
        sim_name,
        start,
        end,
        mission,
        fuel_weight,
        weather_input_dir,
        fc_input_dir=None,
        model_path=None,
        speed=0,
        heading=0,
        speed_options=[],
    ):
        """
        vesselFuel Constructor
         -------------------------
        Parameters:
        ------------------------
        :param name: string, vessel name
        :param sim_name: string simulation name
        :param start: tuple of (lat, long), vessel starting point
        :param end: tuple of (lat, long), vessel ending point
        :param mission: mission object
        :param fuel_weight: float, vessel's starting fuel weight in kN
        :param weather_input_dir: string, directory to nowcast wave data files
        :param forecast_input_dir: string, directory to forecast wave data files
        :param model_path: string. path to ml model for wave predictions
        :param speed: float, vessel's starting speed in knots
        :param heading: float, vessel's starting heading in degrees (0 = north, 90 = east)
        :param speed_options: list of floats, all different speed settings for the vessel
        """
        super().__init__(
            name,
            sim_name,
            start,
            end,
            mission,
            weather_input_dir,
            fc_input_dir,
            model_path,
            speed,
            heading,
            fuel_weight,
            speed_options,
        )
        self.fuel_burned = 0  # float-fuel burned in the current time interval(kN)
        self.eta_O = 0  # float-propeller fixed propulsion efficiency
        self.RPM = 0  # float-propeller revolutions/minute
        self.P_deliv = 0  # float-propeller delivered power(kW)
        self.resistance_in_kn = 0  # float - bare hull resistance in kN
        self.wake_fraction = 0  # float - wake fraction
        self.thrust_deduction = 0  # float - thrust deduction factor
        self.propeller = PropellerPropulsionModel(3.40, 1.39, 0.67, 4)
        self.res_model = NPL_ResistanceModel(1025.0, 1.225, 1.1395e-6, 0.0, npl_path)
        self.RPM_vals = [800, 1000, 1300, 1600, 1800, 2000, 2100, 2300]
        self.power_vals = [[46], [89], [195], [364], [518], [711], [823], [1081]]
        self.fuel_vals = [
            [256.8],
            [230.8],
            [217.3],
            [212.1],
            [208.9],
            [208.3],
            [212.2],
            [222.4],
        ]
        engine1 = RPM_EngineModel(self.RPM_vals, self.power_vals, self.fuel_vals, 10)
        self.propSim1 = propulsionSimulationBase(
            self.res_model, self.propeller, engine1
        )
        # add methods for database
        self.addMethod(fuelCalc(self))
        self.addMethod(resistanceCalc(self))
        self.addMethod(propulsionCalc(self))
        self.addMethod(engineCalc(self))

    def timestep(self, time_interval, current_time=datetime(2018, 1, 1, 1, 1)):
        """
        Summary
        --------------------------------------------------
        Performs fuel calculations and updates fuel weight
        -------------------------------------------------
        Parameters
        -------------------------------------------------
        :param time_interval: float, amount of time that passed in hours
        :param current_time: datetime object, current time of the simulation
        """
        self.calc_fuel_burned(self.speed, time_interval)
        self.fuel_weight -= self.fuel_burned
        if self.fuel_weight <= 0:
            self.get_fuel_time(current_time, time_interval, self.fuel_burned)
            self.fuel_weight = 0
            self.speed = 0
            self.active = False

    def calc_fuel_burned(self, speed, time_interval):
        """
        Summary
        -----------------------------------------------------
        Updates fuel calculation variables for time_interval.

        Parameters
        ----------------------------------------------------
        :param speed: float, vessel's current speed
        :param time_interval: float, amount of time that passed in hours
        """
        (
            self.fuel_burned,
            sfc,
            (self.eta_O, self.P_deliv, self.RPM),
            (self.resistance_in_kn, self.wake_fraction, self.thrust_deduction),
        ) = self.propSim1.runMachinery(speed, self, time_interval)

    def getHydrostaticProperties(self):
        """
        returns the set hydrostatic properties
        """
        return {
            "Length": 45.0,
            "Beam": 7.0,
            "WettedSurface": 100.0,
            "Draft": 3.0,
            "Volume": 472.5,
            "WindageArea": 0.0,
            "AirDragCoefficient": 0.0,
        }

    def get_fuel_time(self, current_time, time_interval, fuel_burned):
        """
        Summary
        --------------------------------------------------------------------------
        Calculates the time the vessel ran out of fuel based on the negative fuel weight

        Parameters
        --------------------------------------------------------------------------------
        :param current_time: datetime object, current time of the simulation
        :param time_interval: float, amount of time that passed in hours
        :param fuel_burned: float, amount of fuel burned by the vessel in the time interval
        """
        time_diff = time_interval * (self.fuel_weight) / (time_interval * fuel_burned)
        current_time += timedelta(hours=time_diff)
        pos = self.calculate_new_position(self.position, self.speed, time_diff)
        self.position.lat = pos[0]
        self.position.long = pos[1]
        logger.info(
            f"{self.name} ran out of fuel at {current_time} and position ({self.position.lat}, {self.position.long})\n"
        )

    def plan_Optimal_Mission(self, current_time):
        """
        Summary
        --------------------------------------------------------------------
        calls ga to plan an optimal mission and changes the vessel's current
        mission to the optimal one.

        Parameters
        --------------------------------------------------------------------
        :param current_time: datetime object, current time of the simulation


        """
        problem = VesselProblem(self, current_time, self.speed_options)
        opt = Optimizer(problem, 94140, 100)
        opt_dests, speeds, fuel_left, fitness = opt.run(300)
        if self.opt_mission != None:
            for i in range(0, self.opt_mission.current_waypoint_index - 1):
                opt_dests.insert(i, self.opt_mission.destinations[i])
                speeds.insert(i, self.opt_mission.speeds[i])
            self.replan_count += 1
            self.opt_mission.destinations = opt_dests
            self.opt_mission.speeds = speeds
            self.opt_mission.update_total_priority()
        else:
            self.opt_mission = Mission(opt_dests, speeds)


class Simulation:
    """
    Runs a simulation consisting of multiple vessels.
    Creates an animation that shows each vessel navigating to their waypoints.
    Adds data from the simulation to databases
    """

    def __init__(self, vessels, start_time, end_time, display_map=True):
        """
        Summary
        ---------------------------
        Constructor for Simulation

        Parameters:
        ------------------------------
        :param vessels: list of vessel objects
        :param start_time: datetime object, start time of the simulation
        :param end_time: datetime object, time when the simulation ends
        :param display_map: displays animation when true
        """
        self.vessels = vessels
        for vessel in vessels:
            vessel.plan_Optimal_Mission(vessel.start.arrival_time)
            vessel.add_opt_mission_to_db("Initial Mission Plan", start_time)
            mission_status = vessel.mission_status_string()
            log_entry = f"\nIntial Mission For {vessel.name}\n" f"{mission_status}"
            logger.info(log_entry)

        self.start_time = start_time
        self.end_time = end_time
        self.display_map = display_map
        self.sc = []  # list of ship icons
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        self.colors = [
            "lightcoral",
            "orangered",
            "gold",
            "honeydew",
            "lawngreen",
            "cyan",
            "violet",
            "deeppink",
            "lightpink",
        ]
        counter = 0

        for vessel in self.vessels:
            color = self.colors[counter % len(self.colors)]
            self.sc.append(
                self.ax.scatter(
                    vessel.position.long,
                    vessel.position.lat,
                    marker="^",
                    color=color,
                    s=100,
                    transform=ccrs.PlateCarree(),
                    label=vessel.name,
                )
            )

            # Adds dots at the location of each waypoint
            count = 0
            for destination in vessel.full_mission.destinations:
                waypoint = destination.waypoint
                if count == 0:
                    self.ax.scatter(
                        waypoint.long,
                        waypoint.lat,
                        marker=".",
                        color=color,
                        s=100,
                        transform=ccrs.PlateCarree(),
                        label=f"{vessel.name}'s waypoint",
                    )
                    count = 1
                else:
                    self.ax.scatter(
                        waypoint.long,
                        waypoint.lat,
                        marker=".",
                        color=color,
                        s=100,
                        transform=ccrs.PlateCarree(),
                    )
            self.ax.scatter(
                vessel.end.waypoint.long,
                vessel.end.waypoint.lat,
                marker=".",
                color=color,
                s=100,
                transform=ccrs.PlateCarree(),
            )
            self.ax.legend()
            vessel.makeTables()
            counter += 1
        plt.ion()

    def run(self, time_step):
        """
        Summary
        ---------------------------
        Runs the simulation

         Parameters
        ------------------------------
         :param time_step: float, length of time interval for each call to navigate (hours)

        """
        current_time = self.start_time
        while current_time <= self.end_time:
            i = 0  # counter
            for vessel in self.vessels:
                if vessel.active and current_time >= vessel.start.arrival_time:
                    if not vessel.navigate(current_time, time_step):
                        # moves vessel to final waypoint
                        lastwp = vessel.opt_mission.destinations[
                            len(vessel.opt_mission.destinations) - 1
                        ].waypoint
                        self.sc[i].set_offsets((lastwp.long, lastwp.lat))
                        db_manager.insert_log(
                            vessel.vessel_id,
                            vessel.mission_id,
                            current_time.strftime("%Y-%m-%d %H:%M:%S"),
                            vessel.position.lat,
                            vessel.position.long,
                            vessel.speed,
                            "Good",
                            vessel.fuel_weight,
                            0,
                            0,
                        )
                        vessel.active = False
                    else:
                        # updates marker position/rotation to match vessel
                        if self.display_map:
                            self.sc[i].set_offsets(
                                (vessel.position.long, vessel.position.lat)
                            )
                            self.ax.set_title(current_time, fontsize=15)
                            self.update_marker_rotation(self.sc[i], vessel.heading)
                            plt.pause(0.1)
                            plt.show()
                        timestep_values = vessel.writeTimestep()
                        for calc_val in timestep_values:
                            # Add the timestep to the dictionary
                            calc_val[1]["timestep"] = current_time
                            # print(calc_val[0])
                            # print(calc_val[1])
                            db_manager.insert_data(calc_val[0], calc_val[1])
                i += 1

            current_time += timedelta(hours=time_step)
            time.sleep(0.1)
        if self.display_map:
            plt.ioff()
            plt.legend()
            plt.show()

        print("Simulation ended.")

    def makeMap(self):
        """
        Makes the map of the ocean from latitude and longitude ranges
        """
        # Initialize with extreme values
        lon_min = 180
        lon_max = -180
        lat_min = 90
        lat_max = -90

        # Find min/max coordinates from all vessel destinations
        for v in self.vessels:
            for dest in v.full_mission.destinations:
                lon = dest.waypoint.long
                lat = dest.waypoint.lat
                lon_min = min(lon_min, lon)
                lon_max = max(lon_max, lon)
                lat_min = min(lat_min, lat)
                lat_max = max(lat_max, lat)

        # Add padding of 5 degrees to each side
        padding = 2
        lon_min -= padding
        lon_max += padding
        lat_min -= padding
        lat_max += padding

        print(f"Longitude range: {lon_min} to {lon_max}")
        print(f"Latitude range: {lat_min} to {lat_max}")

        # Plotting
        self.ax.coastlines()
        self.ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        self.ax.set_facecolor("blue")
        self.ax.set_title(self.start_time, fontsize=15)

    def update_marker_rotation(self, sc, heading):
        """
        Summary
        --------------------------------------------------------------
        Adjusts the rotation of the vessel marker based on the heading

        Parameters:
        --------------------------------------------------------------
        :param sc: vessel marker
        :param heading: float, vessel's heading
        """
        if (heading >= -5 and heading <= 5) or (heading >= 175 and heading <= 185):
            transform = Affine2D().rotate_deg(heading)
        elif np.abs(heading) >= 85 and np.abs(heading) <= 95:
            transform = Affine2D().rotate_deg(heading + 180)
        elif heading > 0:
            transform = Affine2D().rotate_deg(heading - 90)
        else:
            transform = Affine2D().rotate_deg(heading + 90)

        sc.set_transform(transform)


if __name__ == "__main__":
    # Example usage

    db_manager.clear_tables()
    destinations1 = [
        Destination(Waypoint(38.5, -137), datetime(2024, 12, 20, 5, 0), 7),
        Destination(Waypoint(38.5, -138), datetime(2024, 12, 26, 6, 0), 9),
        Destination(Waypoint(37.5, -139), datetime(2024, 12, 29, 20, 0), 32),
        Destination(Waypoint(38, -138), datetime(2024, 12, 20, 15, 0), 77),
        Destination(Waypoint(38, -137.5), datetime(2024, 12, 23, 16, 0), 15),
        Destination(Waypoint(37.5, -136.5), datetime(2024, 12, 20, 5, 0), 100),
        Destination(Waypoint(38, -136), datetime(2024, 12, 21, 15, 0), 10),
        Destination(Waypoint(38.1, -139), datetime(2024, 12, 25, 10, 0), 45),
        Destination(Waypoint(40.5, -139), datetime(2024, 12, 22, 17, 0), 10),
        Destination(Waypoint(41.5, -139), datetime(2024, 12, 25, 19, 0), 47),
        Destination(Waypoint(42, -138), datetime(2024, 12, 22, 15, 0), 77),
        Destination(Waypoint(40, -137), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 9),
        Destination(Waypoint(39, -136.5), datetime(2024, 12, 22, 15, 0), 6),
    ]
    destinations2 = [
        Destination(Waypoint(40, -137), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 9),
        Destination(Waypoint(41.5, -139), datetime(2024, 12, 25, 19, 0), 47),
        Destination(Waypoint(41.5, -139.5), datetime(2024, 12, 25, 23, 0), 37),
        Destination(Waypoint(42, -138), datetime(2024, 12, 22, 15, 0), 77),
        Destination(Waypoint(40, -137.5), datetime(2024, 12, 19, 15, 0), 27),
        Destination(Waypoint(39, -136.5), datetime(2024, 12, 22, 15, 0), 6),
        Destination(Waypoint(40, -139), datetime(2024, 12, 21, 17, 0), 7),
        Destination(Waypoint(42, -137), datetime(2024, 12, 23, 5, 0), 17),
        Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 23),
        Destination(Waypoint(43, -139), datetime(2024, 12, 25, 20, 0), 57),
        Destination(Waypoint(42.5, -139.5), datetime(2024, 12, 25, 23, 0), 72),
        Destination(Waypoint(42.5, -136), datetime(2024, 12, 22, 15, 0), 27),
        Destination(Waypoint(40, -136), datetime(2024, 12, 29, 15, 0), 34),
        Destination(Waypoint(43, -136.5), datetime(2024, 12, 23, 12, 0), 54),
        Destination(Waypoint(42.5, -137), datetime(2024, 12, 21, 11, 0), 18),
    ]
    destinations3 = [
        Destination(Waypoint(35.5, 157), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(36, 158), datetime(2024, 12, 26, 2, 0), 9),
        Destination(Waypoint(36.5, 159), datetime(2024, 12, 19, 15, 0), 47),
        Destination(Waypoint(37.5, 159.5), datetime(2024, 12, 23, 15, 0), 37),
        Destination(Waypoint(35, 158), datetime(2024, 12, 19, 15, 0), 73),
        Destination(Waypoint(35, 157.5), datetime(2024, 12, 22, 15, 0), 7),
        Destination(Waypoint(37.5, 156.5), datetime(2024, 12, 21, 15, 0), 100),
        Destination(Waypoint(35, 159), datetime(2024, 12, 22, 17, 0), 17),
        Destination(Waypoint(35, 157), datetime(2024, 12, 21, 15, 0), 87),
        Destination(Waypoint(35.5, 158), datetime(2024, 12, 26, 2, 0), 19),
        Destination(Waypoint(36, 159), datetime(2024, 12, 22, 15, 0), 7),
        Destination(Waypoint(38, 159), datetime(2024, 12, 23, 12, 0), 55),
        Destination(Waypoint(38, 158), datetime(2024, 12, 20, 4, 0), 3),
        Destination(Waypoint(38, 157), datetime(2024, 12, 21, 9, 0), 22),
    ]
    destinations4 = [
        Destination(Waypoint(35.5, 152), datetime(2024, 12, 21, 5, 0), 117),
        Destination(Waypoint(36, 153), datetime(2024, 12, 26, 2, 0), 91),
        Destination(Waypoint(36.5, 151), datetime(2024, 12, 19, 15, 0), 47),
        Destination(Waypoint(37.5, 152.5), datetime(2024, 12, 23, 15, 0), 7),
        Destination(Waypoint(35, 151), datetime(2024, 12, 19, 15, 0), 73),
        Destination(Waypoint(35, 153.5), datetime(2024, 12, 22, 15, 0), 17),
        Destination(Waypoint(37.5, 154.5), datetime(2024, 12, 21, 15, 0), 10),
        Destination(Waypoint(35.5, 155), datetime(2024, 12, 22, 17, 0), 17),
        Destination(Waypoint(35.5, 152), datetime(2024, 12, 21, 20, 0), 19),
        Destination(Waypoint(36, 152), datetime(2024, 12, 26, 12, 0), 10),
        Destination(Waypoint(36, 151), datetime(2024, 12, 22, 15, 0), 28),
        Destination(Waypoint(37, 152), datetime(2024, 12, 23, 15, 0), 33),
        Destination(Waypoint(35.5, 152), datetime(2024, 12, 25, 3, 0), 16),
        Destination(Waypoint(35, 151.5), datetime(2024, 12, 23, 4, 0), 54),
        Destination(Waypoint(34, 154.5), datetime(2024, 12, 29, 16, 0), 100),
        Destination(Waypoint(36, 152), datetime(2024, 12, 22, 17, 0), 8),
    ]
    dests_small = [
        Destination(Waypoint(38.5, -137), datetime(2024, 12, 20, 5, 0), 7),
        Destination(Waypoint(38.5, -138), datetime(2024, 12, 26, 6, 0), 9),
    ]
    mission_small = Mission(dests_small)
    fc_test_mission1 = Mission(destinations1)
    fc_test_mission2 = Mission(destinations2)
    fc_test_mission3 = Mission(destinations3)
    fc_test_mission4 = Mission(destinations4)

    vessel_fc_test1 = vesselFuel(
        "vessel1",
        "sim1",
        Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0),
        fc_test_mission1,
        50,
        weath_input_dir_lat40_lon_neg_140,
        model_path=model_path,
        speed=20,
        speed_options=[15, 20, 25],
    )
    vessel_fc_test2 = vesselFuel(
        "vessel2",
        "sim1",
        Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0),
        fc_test_mission2,
        50,
        weath_input_dir_lat40_lon_neg_140,
        model_path=model_path,
        speed=20,
        speed_options=[15, 20, 25],
    )
    vessel_fc_test3 = vesselFuel(
        "vessel3",
        "sim1",
        Destination(Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0),
        fc_test_mission3,
        50,
        weath_input_dir_lat35_lon155,
        model_path=model_path,
        speed=20,
        speed_options=[15, 20, 25],
    )
    vessel_fc_test4 = vesselFuel(
        "vessel4",
        "sim1",
        Destination(Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0),
        fc_test_mission4,
        50,
        weath_input_dir_lat35_lon155,
        fc_input_dir=fc_input_dir_lat35_lon155,
        speed=20,
        speed_options=[15, 20, 25],
    )
    vessel_small = vesselFuel(
        "vessel1",
        "sim1",
        Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0),
        mission_small,
        50,
        weath_input_dir_lat40_lon_neg_140,
        model_path=model_path,
        speed=20,
        speed_options=[15, 20, 25],
    )
    simulation = Simulation(
        [vessel_fc_test1, vessel_fc_test2, vessel_fc_test3, vessel_fc_test4],
        datetime(2024, 12, 20, 0),
        datetime(2024, 12, 31, 0, 0),
    )

    simulation.makeMap()
    simulation.run(3)

    db_manager.close()
