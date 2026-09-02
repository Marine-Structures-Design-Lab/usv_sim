"""Test Cases for Sim Engine"""

from datetime import datetime
import pytest
from Simulation.vessel_sim_engine import (
    Waypoint,
    Mission,
    Destination,
    vesselBase,
    vesselFuel,
)
from geographiclib.geodesic import Geodesic
from Simulation.machinery_fuel import (
    propulsionSimulationBase,
    PropellerPropulsionModel,
    RPM_EngineModel,
    NPL_ResistanceModel,
)

geod = Geodesic.WGS84
from pathlib import Path

script_dir = Path(__file__).resolve().parent
npl_path = script_dir.parent / "Simulation" / "Molland_NPL_Data_NumpyRead.csv"
model_path = "Wave_Predictor/torch_data/model.pth"
ocean_path = (
    script_dir.parent / "Ocean/weather_2_nc/1205_1215_lat-10_5_lon50_60_fc350_ts9"
)


@pytest.fixture
def base_vessel():
    destinations = [
        Destination(Waypoint(3, 55), datetime(2024, 12, 8, 2, 0), 9),
        Destination(Waypoint(4, 55), datetime(2024, 12, 8, 17, 0), 7),
        Destination(Waypoint(2.5, 54.5), datetime(2024, 12, 9, 6, 0), 9),
    ]

    mission = Mission(destinations)
    vessel = vesselBase(
        "base_vessel",
        "test_sim",
        Destination(Waypoint(3, 54.5), datetime(2024, 12, 7, 20, 0), 0),
        Destination(Waypoint(3, 54.5), datetime(2024, 12, 9, 7, 0), 0),
        mission,
        ocean_path,
        model_path=model_path,
        speed=20,
    )
    return destinations, mission, vessel


@pytest.fixture
def fuel_vessel():
    destinations = [
        Destination(Waypoint(3, 55), datetime(2024, 12, 8, 2, 0), 9),
        Destination(Waypoint(4, 55), datetime(2024, 12, 8, 17, 0), 7),
        Destination(Waypoint(2.5, 54.5), datetime(2024, 12, 9, 6, 0), 9),
    ]

    mission = Mission(destinations)
    vessel = vesselFuel(
        "fuel_vessel",
        "test_sim",
        Destination(Waypoint(3, 54.5), datetime(2024, 12, 7, 20, 0), 0),
        Destination(Waypoint(3, 54.5), datetime(2024, 12, 9, 7, 0), 0),
        mission,
        100,
        ocean_path,
        model_path=model_path,
        speed=20,
    )
    vessel.opt_mission = vessel.full_mission
    vessel.opt_mission.current_waypoint_index = 0
    return destinations, mission, vessel


@pytest.fixture
def fuel_model():
    propeller = PropellerPropulsionModel(3.40, 1.39, 0.67, 4)
    res_model = NPL_ResistanceModel(1025.0, 1.225, 1.1395e-6, 0.0, npl_path)
    RPM_vals = [800, 1000, 1300, 1600, 1800, 2000, 2100, 2300]
    power_vals = [[46], [89], [195], [364], [518], [711], [823], [1081]]
    fuel_vals = [[256.8], [230.8], [217.3], [212.1], [208.9], [208.3], [212.2], [222.4]]
    engine1 = RPM_EngineModel(RPM_vals, power_vals, fuel_vals, 10)
    propSim1 = propulsionSimulationBase(res_model, propeller, engine1)
    return propSim1


def test_mission(base_vessel):
    assert base_vessel[1].current_waypoint() == base_vessel[0][1].waypoint
    assert base_vessel[1].current_goal_time() == base_vessel[0][1].arrival_time
    assert base_vessel[1].advance_waypoint() == True
    assert base_vessel[1].current_waypoint() == base_vessel[0][2].waypoint
    assert base_vessel[1].current_goal_time() == base_vessel[0][2].arrival_time
    assert base_vessel[1].advance_waypoint() == False


def test_navigate_heading_position(fuel_vessel):
    starting_position = fuel_vessel[2].start
    # heading between start and waypoint1
    heading1 = geod.Inverse(
        starting_position.waypoint.lat,
        starting_position.waypoint.long,
        fuel_vessel[0][0].waypoint.lat,
        fuel_vessel[0][0].waypoint.long,
    )["azi1"]
    # heading between waypoint1 and waypoint2
    heading2 = geod.Inverse(
        fuel_vessel[0][0].waypoint.lat,
        fuel_vessel[0][0].waypoint.long,
        fuel_vessel[0][1].waypoint.lat,
        fuel_vessel[0][1].waypoint.long,
    )["azi1"]
    # heading between waypoint2 and waypoint3
    heading3 = geod.Inverse(
        fuel_vessel[0][1].waypoint.lat,
        fuel_vessel[0][1].waypoint.long,
        fuel_vessel[0][2].waypoint.lat,
        fuel_vessel[0][2].waypoint.long,
    )["azi1"]
    fuel_vessel[2].fuel_weight = 10000
    fuel_vessel[2].navigate(datetime(2024, 12, 7, 20, 0), 1)
    distance = fuel_vessel[2].speed * 1852 * 1
    g1 = geod.Direct(
        starting_position.waypoint.lat,
        starting_position.waypoint.long,
        heading1,
        distance,
    )
    lat = g1["lat2"]
    long = g1["lon2"]

    assert fuel_vessel[2].heading == heading1
    assert fuel_vessel[2].position.lat == lat
    assert fuel_vessel[2].position.long == long

    # reaches waypoint 2 during
    fuel_vessel[2].navigate(datetime(2024, 12, 7, 21, 0), 5)
    distance += fuel_vessel[2].speed * 1852 * (5 - 2.998705465968354)
    g = geod.Direct(
        starting_position.waypoint.lat,
        starting_position.waypoint.long,
        heading1,
        distance,
    )
    lat = g["lat2"]
    long = g["lon2"]
    distance = fuel_vessel[2].speed * 1852 * (2.998705465968354)
    g = geod.Direct(lat, long, heading2, distance)
    lat = g["lat2"]
    long = g["lon2"]
    assert fuel_vessel[2].heading == pytest.approx(heading2, 0.1)
    assert fuel_vessel[2].position.lat == pytest.approx(lat, 0.001)
    assert fuel_vessel[2].position.long == pytest.approx(long, 0.001)
    # 49.39563163497132,-33.54210224044771,
    fuel_vessel[2].navigate(datetime(2024, 12, 8, 2, 0), 2)
    assert fuel_vessel[2].heading == pytest.approx(heading2, 0.1)
    distance = fuel_vessel[2].speed * 1852 * 2
    g = geod.Direct(lat, long, heading2, distance)
    lat = g["lat2"]
    long = g["lon2"]
    assert fuel_vessel[2].position.lat == pytest.approx(lat, 0.001)
    assert fuel_vessel[2].position.long == pytest.approx(long, 0.001)

    fuel_vessel[2].navigate(datetime(2024, 12, 8, 4, 0), 4)
    assert fuel_vessel[2].heading == heading3
    distance = fuel_vessel[2].speed * 1852 * (4 - 3.0279472542182)
    g = geod.Direct(lat, long, heading2, distance)
    lat = g["lat2"]
    long = g["lon2"]
    distance = fuel_vessel[2].speed * 1852 * 3.0279472542182
    g = geod.Direct(lat, long, heading3, distance)
    lat = g["lat2"]
    long = g["lon2"]
    assert fuel_vessel[2].position.lat == pytest.approx(lat, 0.001)
    assert fuel_vessel[2].position.long == pytest.approx(long, 0.001)

    fuel_vessel[2].navigate(datetime(2024, 12, 8, 8, 0), 5)
    distance = fuel_vessel[2].speed * 1852 * 5
    g = geod.Direct(lat, long, heading3, distance)
    lat = g["lat2"]
    long = g["lon2"]
    assert fuel_vessel[2].heading == pytest.approx(heading3, 0.1)
    assert fuel_vessel[2].position.lat == pytest.approx(lat, 0.001)
    assert fuel_vessel[2].position.long == pytest.approx(long, 0.001)


def test_navigate_waypoints(fuel_vessel):
    fuel_vessel[2].navigate(datetime(2024, 12, 7, 20, 0), 1)
    assert fuel_vessel[1].current_waypoint() == fuel_vessel[0][0].waypoint
    fuel_vessel[2].navigate(datetime(2024, 12, 7, 21, 0), 5)
    assert fuel_vessel[1].current_waypoint() == fuel_vessel[0][1].waypoint
    fuel_vessel[2].navigate(datetime(2024, 12, 8, 2, 0), 2)
    assert fuel_vessel[1].current_waypoint() == fuel_vessel[0][1].waypoint
    fuel_vessel[2].navigate(datetime(2024, 12, 8, 2, 0), 4)
    assert fuel_vessel[1].current_waypoint() == fuel_vessel[0][2].waypoint
    fuel_vessel[2].navigate(datetime(2024, 12, 8, 6, 0), 5)
    assert fuel_vessel[1].current_waypoint() == fuel_vessel[0][2].waypoint
    assert fuel_vessel[1].advance_waypoint() == False


def test_adjust_speed(fuel_vessel):
    fuel_vessel[2].heading = 45
    fuel_vessel[2].speed = 20
    fuel_vessel[2].speed = fuel_vessel[2].adjust_speed(
        Waypoint(-8, 53), 200, datetime(2024, 12, 6, 7, 0), datetime(2024, 12, 6, 7, 0)
    )
    assert fuel_vessel[2].speed == 20
    fuel_vessel[2].heading = 0
    fuel_vessel[2].speed = fuel_vessel[2].adjust_speed(
        Waypoint(0, 54),
        100,
        datetime(2024, 12, 12, 14, 0),
        datetime(2024, 12, 12, 14, 0),
    )
    assert fuel_vessel[2].speed == 20
    fuel_vessel[2].speed = 20
    fuel_vessel[2].heading = 250
    fuel_vessel[2].speed = fuel_vessel[2].adjust_speed(
        Waypoint(3, 58), 200, datetime(2024, 12, 7, 7, 0), datetime(2024, 12, 7, 7, 0)
    )
    assert fuel_vessel[2].speed == 20


def test_calculate_position(base_vessel):
    base_vessel[2].calculate_new_position(
        base_vessel[2].position, base_vessel[2].speed, 0
    )
    assert base_vessel[2].start.waypoint.lat == 3
    assert base_vessel[2].start.waypoint.long == 54.5
    dist_traveled = base_vessel[2].speed * 0.5 * 1852
    position = geod.Direct(
        base_vessel[2].position.lat,
        base_vessel[2].position.long,
        base_vessel[2].heading,
        dist_traveled,
    )
    assert base_vessel[2].calculate_new_position(
        base_vessel[2].position, base_vessel[2].speed, 0.5
    ) == (position["lat2"], position["lon2"])
    dist_traveled = base_vessel[2].speed * 2 * 1852
    position = geod.Direct(
        base_vessel[2].position.lat,
        base_vessel[2].position.long,
        base_vessel[2].heading,
        dist_traveled,
    )
    assert base_vessel[2].calculate_new_position(
        base_vessel[2].position, base_vessel[2].speed, 2
    ) == (position["lat2"], position["lon2"])


def test_is_at_waypoint(base_vessel):
    print(base_vessel[1].current_waypoint().lat)
    assert base_vessel[2].is_at_waypoint(base_vessel[0][0].waypoint, 1) == False

    assert base_vessel[2].is_at_waypoint(base_vessel[0][0].waypoint, 12) == True


def test_fuel_calc(fuel_vessel, fuel_model):
    time_interval = 5
    fuel_vessel[2].fuel_weight = 15
    fuel_level = fuel_vessel[2].fuel_weight
    (
        fuel_burned,
        sfc,
        (eta_O, P_deliv, RPM),
        (resistance_in_kn, wake_fraction, thrust_deduction),
    ) = fuel_model.runMachinery(fuel_vessel[2].speed, fuel_vessel[2], time_interval)
    fuel_level -= fuel_burned
    fuel_vessel[2].timestep(time_interval, datetime(2018, 6, 2, 18, 0))
    assert fuel_burned == fuel_vessel[2].fuel_burned
    assert fuel_level == fuel_vessel[2].fuel_weight
    assert eta_O == fuel_vessel[2].eta_O
    assert P_deliv == fuel_vessel[2].P_deliv
    assert RPM == fuel_vessel[2].RPM
    assert resistance_in_kn == fuel_vessel[2].resistance_in_kn
    assert wake_fraction == fuel_vessel[2].wake_fraction
    assert thrust_deduction == fuel_vessel[2].thrust_deduction
    assert fuel_vessel[2].active

    time_interval = 2
    (
        fuel_burned,
        sfc,
        (eta_O, P_deliv, RPM),
        (resistance_in_kn, wake_fraction, thrust_deduction),
    ) = fuel_model.runMachinery(fuel_vessel[2].speed, fuel_vessel[2], time_interval)
    fuel_level -= fuel_burned
    fuel_vessel[2].timestep(time_interval, datetime(2018, 6, 2, 18, 0))
    assert fuel_burned == fuel_vessel[2].fuel_burned
    assert fuel_level == fuel_vessel[2].fuel_weight
    assert eta_O == fuel_vessel[2].eta_O
    assert P_deliv == fuel_vessel[2].P_deliv
    assert RPM == fuel_vessel[2].RPM
    assert resistance_in_kn == fuel_vessel[2].resistance_in_kn
    assert wake_fraction == fuel_vessel[2].wake_fraction
    assert thrust_deduction == fuel_vessel[2].thrust_deduction
    assert fuel_vessel[2].active

    time_interval = 8
    (
        fuel_burned,
        sfc,
        (eta_O, P_deliv, RPM),
        (resistance_in_kn, wake_fraction, thrust_deduction),
    ) = fuel_model.runMachinery(fuel_vessel[2].speed, fuel_vessel[2], time_interval)
    fuel_level -= fuel_burned
    fuel_vessel[2].timestep(time_interval, datetime(2018, 6, 2, 18, 0))
    assert fuel_burned == fuel_vessel[2].fuel_burned
    assert fuel_level == fuel_vessel[2].fuel_weight
    assert eta_O == fuel_vessel[2].eta_O
    assert P_deliv == fuel_vessel[2].P_deliv
    assert RPM == fuel_vessel[2].RPM
    assert resistance_in_kn == fuel_vessel[2].resistance_in_kn
    assert wake_fraction == fuel_vessel[2].wake_fraction
    assert thrust_deduction == fuel_vessel[2].thrust_deduction
    assert fuel_vessel[2].active

    # test when vessel runs out of fuel
    time_interval = 20
    (
        fuel_burned,
        sfc,
        (eta_O, P_deliv, RPM),
        (resistance_in_kn, wake_fraction, thrust_deduction),
    ) = fuel_model.runMachinery(fuel_vessel[2].speed, fuel_vessel[2], time_interval)
    fuel_level -= fuel_burned
    fuel_vessel[2].timestep(time_interval, datetime(2018, 6, 2, 18, 0))
    assert fuel_burned == fuel_vessel[2].fuel_burned
    assert 0 == fuel_vessel[2].fuel_weight
    assert eta_O == fuel_vessel[2].eta_O
    assert P_deliv == fuel_vessel[2].P_deliv
    assert RPM == fuel_vessel[2].RPM

    assert resistance_in_kn == fuel_vessel[2].resistance_in_kn
    assert wake_fraction == fuel_vessel[2].wake_fraction
    assert thrust_deduction == fuel_vessel[2].thrust_deduction
    assert not fuel_vessel[2].active
