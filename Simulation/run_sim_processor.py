"""
tests the sim processor using data from running a mission in the simulation engine
"""

from datetime import datetime

from Simulation.vessel_sim_engine import (
    Waypoint,
    Mission,
    Destination,
    vesselFuel,
    Simulation,
)
import Simulation.sim_processor as sp
from pathlib import Path

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


def run_sim_engine():

    forecasting_test_destinations1 = [
        Destination(Waypoint(38.5, -137), datetime(2024, 12, 18, 5, 0), 7),
        Destination(Waypoint(38.5, -138), datetime(2024, 12, 12, 16, 0), 9),
        Destination(Waypoint(37.5, -139), datetime(2024, 12, 19, 15, 0), 77),
        Destination(Waypoint(37.5, -139.5), datetime(2024, 12, 19, 15, 0), 77),
        Destination(Waypoint(38, -138), datetime(2024, 12, 19, 15, 0), 77),
        Destination(Waypoint(38, -137.5), datetime(2024, 12, 19, 15, 0), 77),
        Destination(Waypoint(37.5, -136.5), datetime(2024, 12, 19, 15, 0), 100),
        Destination(Waypoint(40, -139), datetime(2024, 12, 19, 17, 0), 7),
    ]
    forecasting_test_destinations2 = [
        Destination(Waypoint(40, -137), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 9),
        Destination(Waypoint(41.5, -139), datetime(2024, 12, 25, 19, 0), 47),
        Destination(Waypoint(41.5, -139.5), datetime(2024, 12, 25, 23, 0), 37),
        Destination(Waypoint(42, -138), datetime(2024, 12, 22, 15, 0), 77),
        Destination(Waypoint(40, -137.5), datetime(2024, 12, 19, 15, 0), 27),
        Destination(Waypoint(39.5, -136.5), datetime(2024, 12, 22, 15, 0), 6),
        Destination(Waypoint(40, -139), datetime(2024, 12, 21, 17, 0), 7),
    ]
    forecasting_test_destinations3 = [
        Destination(Waypoint(35.5, 157), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(36, 158), datetime(2024, 12, 26, 2, 0), 9),
        Destination(Waypoint(36.5, 159), datetime(2024, 12, 19, 15, 0), 47),
        Destination(Waypoint(37.5, 159.5), datetime(2024, 12, 23, 15, 0), 37),
        Destination(Waypoint(35, 158), datetime(2024, 12, 19, 15, 0), 73),
        Destination(Waypoint(35, 157.5), datetime(2024, 12, 22, 15, 0), 7),
        Destination(Waypoint(37.5, 156.5), datetime(2024, 12, 21, 15, 0), 100),
        Destination(Waypoint(35, 159), datetime(2024, 12, 22, 17, 0), 17),
    ]
    forecasting_test_destinations4 = [
        Destination(Waypoint(35.5, 152), datetime(2024, 12, 21, 5, 0), 117),
        Destination(Waypoint(36, 153), datetime(2024, 12, 26, 2, 0), 91),
        Destination(Waypoint(36.5, 151), datetime(2024, 12, 19, 15, 0), 47),
        Destination(Waypoint(37.5, 152.5), datetime(2024, 12, 23, 15, 0), 7),
        Destination(Waypoint(35, 151), datetime(2024, 12, 19, 15, 0), 73),
        Destination(Waypoint(35, 153.5), datetime(2024, 12, 22, 15, 0), 17),
        Destination(Waypoint(37.5, 154.5), datetime(2024, 12, 21, 15, 0), 10),
        Destination(Waypoint(35, 152), datetime(2024, 12, 22, 17, 0), 17),
    ]
    fc_test_mission1 = Mission(forecasting_test_destinations1)
    fc_test_mission2 = Mission(forecasting_test_destinations2)
    fc_test_mission3 = Mission(forecasting_test_destinations3)
    fc_test_mission4 = Mission(forecasting_test_destinations4)
    vessel_fc_test1 = vesselFuel(
        "vessel1",
        "sim1",
        Destination(Waypoint(38.1, -139), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(38.1, -139), datetime(2024, 12, 24, 7, 0), 0),
        fc_test_mission1,
        50,
        weath_input_dir_lat40_lon_neg_140,
        fc_input_dir=fc_input_dir_lat40_lon_neg_140,
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
    simulation = Simulation(
        [vessel_fc_test1, vessel_fc_test2, vessel_fc_test3, vessel_fc_test4],
        datetime(2024, 12, 20, 0),
        datetime(2024, 12, 31, 0, 0),
    )

    simulation.makeMap()
    simulation.run(3)


run_sim_engine()
df = sp.simulation_compare("Simulation/sim_db.db", ["sim1"], True)
df.to_csv("simulation_results.csv", index=False)
sp.run_animation("Simulation/sim_db.db", "sim1", 0.2)
sp.make_fuel_graph("Simulation/sim_db.db", "sim1")
