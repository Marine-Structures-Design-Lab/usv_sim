import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from src.Simulation.vessel_sim_engine import (
    Waypoint,
    Destination,
    Mission,
    vesselFuel,
    vesselBase,
    Simulation,
)
import copy

# file paths
script_dir = Path(__file__).resolve().parent
weath_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1201_1231_lat30_lon30_fc0_ts6"
)
fc_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1215_1225_lat10_lon10_fc384_ts9"
)
weath_input_dir_lat35_lon155 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1210_1231_lat30_40_lon150_160_fc0_ts6"
)
fc_input_dir_lat35_lon155 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1218_1228_lat30_40_lon150_160_fc350_ts9"
)
model_path = "src/Wave_Predictor/torch_data/model.pth"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime

from src.Simulation.vessel_sim_engine import (
    Waypoint,
    Destination,
    Mission,
    vesselFuel,
    vesselBase,
    Simulation,
)


class MultiSimulationRunner:
    def __init__(self, simulation_tasks, time_step=3, max_workers=4):
        """
        Constructor for the multi simulation runner
        --------------------------------------------------

        Parameters
        --------------------------------------------------
        :param simulation_tasks: list of dictionaries, simualtions to run
        :param time_step: int, simulation timestep in hours
        :param max_workers: int, number of simulations running in parallel at a time
        """
        self.simulation_tasks = simulation_tasks
        self.time_step = time_step
        self.max_workers = max_workers

    @staticmethod
    def _build_vessel(task_name, v):
        """
        builds a vessel class object
        ----------------------------

        Parameters
        ----------------------------
        :param task_name: string, simulaton task name
        :param v: dictionary, one vessel
        """
        dest_copy = copy.deepcopy(v["mission_destinations"])
        mission = Mission(dest_copy)

        vessel_class = v["vessel_class"]
        if vessel_class not in (vesselFuel, vesselBase):
            raise RuntimeError(f"Invalid vessel class: {vessel_class}")

        return vessel_class(
            v["name"],
            task_name,
            v["start_obj"],
            v["end_obj"],
            mission,
            v.get("fuel", 50),
            v.get("weather_input_dir"),
            model_path=v.get("model_path"),
            fc_input_dir=v.get("fc_input_dir"),
            speed=v.get("speed", 20),
            speed_options=v.get("speed_options", [15, 20, 25]),
        )

    @staticmethod
    def run_single_sim(task, time_step):
        """
        builds a vessel class object
        ------------------------------------------------------

        Parameters
        ------------------------------------------------------
        :param  task: dictionary, simualtion dictionary
        :param time_step: int, time_step, simulation timestep in hours

        Returns
        -------------------------------------------------------
        string, simulation name
        """
        vessels = [
            MultiSimulationRunner._build_vessel(task["name"], v)
            for v in task["vessels"]
        ]

        sim = Simulation(vessels, task["start_time"], task["end_time"], False)

        sim.run(time_step=time_step)

        sim_id = task["name"]
        print(f"Finished {sim_id}")
        return sim_id

    def run_all(self):
        """
        runs all taks in simulation tasks dictionary
        """
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(
                MultiSimulationRunner.run_single_sim,
                self.simulation_tasks,
                [self.time_step] * len(self.simulation_tasks),
            )

        print(f"Finished {len(list(results))} simulations.")


if __name__ == "__main__":
    # db_manager.clear_tables()
    destinations1_ml = [
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
    destinations2_ml = [
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
    destinations3_ml = [
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
    destinations4_ml = [
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

    destinations1_fc = [
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
    destinations2_fc = [
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
    destinations3_fc = [
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
    destinations4_fc = [
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

    simulation_tasks = [
        {
            "name": "12/18",
            "start_time": datetime(2024, 12, 18, 0),
            "end_time": datetime(2024, 12, 31, 0),
            "vessels": [
                {
                    "name": "vessel1_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel1_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
            ],
        },
        {
            "name": "12/19",
            "start_time": datetime(2024, 12, 19, 0),
            "end_time": datetime(2024, 12, 31, 0),
            "vessels": [
                {
                    "name": "vessel1_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel1_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
            ],
        },
        {
            "name": "12/20",
            "start_time": datetime(2024, 12, 20, 0),
            "end_time": datetime(2024, 12, 31, 0),
            "vessels": [
                {
                    "name": "vessel1_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel1_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 20, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
            ],
        },
        {
            "name": "12/21",
            "start_time": datetime(2024, 12, 21, 0),
            "end_time": datetime(2024, 12, 31, 0),
            "vessels": [
                {
                    "name": "vessel1_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_ml,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_ml",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_ml,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "model_path": model_path,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel1_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations1_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel2_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations2_fc,
                    "weather_input_dir": weath_input_dir_lat40_lon_neg_140,
                    "fc_input_dir": fc_input_dir_lat40_lon_neg_140,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel3_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(36, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations3_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
                {
                    "name": "vessel4_fc",
                    "vessel_class": vesselFuel,
                    "start_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 21, 0, 0), 0
                    ),
                    "end_obj": Destination(
                        Waypoint(35, 155), datetime(2024, 12, 24, 7, 0), 0
                    ),
                    "mission_destinations": destinations4_fc,
                    "weather_input_dir": weath_input_dir_lat35_lon155,
                    "fc_input_dir": fc_input_dir_lat35_lon155,
                    "speed": 20,
                    "speed_options": [15, 20, 25],
                },
            ],
        },
    ]
    runner = MultiSimulationRunner(simulation_tasks, time_step=3, max_workers=4)
    runner.run_all()
