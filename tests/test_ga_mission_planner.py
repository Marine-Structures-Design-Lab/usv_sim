"""Test Cases for genetic algorithm mission_planner"""

from datetime import datetime
import pytest
from src.Simulation.vessel_sim_engine import Waypoint, Mission, Destination, vesselFuel
from geographiclib.geodesic import Geodesic
from src.Simulation.ga_mission_planner import VesselProblem, Optimizer

geod = Geodesic.WGS84
from pathlib import Path

script_dir = Path(__file__).resolve().parent
model_path = "src/Wave_Predictor/torch_data/model.pth"
weath_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1201_1231_lat30_lon30_fc0_ts6"
)
fc_input_dir_lat40_lon_neg_140 = (
    script_dir.parent / "src/Ocean/weather_2_nc/1215_1225_lat10_lon10_fc384_ts9"
)


@pytest.fixture
def ga():
    destinations = [
        Destination(Waypoint(40, -137), datetime(2024, 12, 21, 5, 0), 7),
        Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 9),
        Destination(Waypoint(41.5, -139), datetime(2024, 12, 25, 19, 0), 47),
        Destination(Waypoint(41.5, -139.5), datetime(2024, 12, 25, 23, 0), 37),
        Destination(Waypoint(42, -138), datetime(2024, 12, 22, 15, 0), 77),
        Destination(Waypoint(40, -137.5), datetime(2024, 12, 19, 15, 0), 27),
        Destination(Waypoint(39.5, -136.5), datetime(2024, 12, 22, 15, 0), 6),
        Destination(Waypoint(40, -139), datetime(2024, 12, 21, 17, 0), 7),
    ]
    mission = Mission(destinations)
    vessel = vesselFuel(
        "vessel2",
        "testsim2",
        Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0),
        Destination(Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0),
        mission,
        50,
        weath_input_dir_lat40_lon_neg_140,
        model_path=model_path,
        speed=20,
        speed_options=[15, 20, 25],
    )
    problem = VesselProblem(vessel, datetime(2024, 12, 20, 0, 0))
    opt = Optimizer(problem, 94140, 20)
    return opt


def test_calculate_fuel(ga):
    speeds = [20] * len(ga.prob.waypoints)
    assert ga.calculate_fuel(Mission(ga.prob.waypoints, speeds)) == pytest.approx(
        29.9333, 0.001
    )
    speeds = [5] * len(ga.prob.waypoints)
    assert ga.calculate_fuel(Mission(ga.prob.waypoints, speeds)) == pytest.approx(
        2.463, 0.001
    )
    new_destinations = ga.prob.waypoints[3:5]

    speeds = [5, 10, 5, 10]
    fuel = ga.calculate_fuel(Mission(new_destinations, speeds))
    assert fuel == pytest.approx(0.0902, 0.001)
    assert (
        ga.calculate_fuel(
            Mission(
                [Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 9)], [10]
            )
        )
        == 0
    )


def test_make_vessel_individual(ga):
    chrom = [[0, 1, 2, len(ga.prob.waypoints) - 1], [5, 10, 5, 10]]
    ind = ga.make_vessel_individual(chrom)
    mission = Mission(
        [ga.prob.start, ga.prob.waypoints[1], ga.prob.waypoints[2], ga.prob.end],
        [5, 10, 5, 10],
    )
    fuel = ga.calculate_fuel(mission, [15, 10, 15, 10])
    assert ind.total_priority == mission.total_priority / ga.prob.max_priority
    assert ind.chromosome == chrom
    assert ind.fuel_used == fuel
    v_sum = 0
    if fuel > ga.prob.vessel.fuel_weight:
        v_sum = (fuel - ga.prob.vessel.fuel_weight) / ga.prob.vessel.fuel_weight
    assert ind.violation_sum == v_sum

    chrom = [[0, 6, len(ga.prob.waypoints) - 1], [20, 10, 15, 20]]
    ind = ga.make_vessel_individual(chrom)
    mission = Mission(
        [ga.prob.start, ga.prob.waypoints[6], ga.prob.end], [20, 10, 15, 20]
    )
    fuel = ga.calculate_fuel(mission)
    assert ind.total_priority == mission.total_priority / ga.prob.max_priority
    assert ind.chromosome == chrom
    assert ind.fuel_used == fuel
    violation = 0
    if fuel > ga.prob.vessel.fuel_weight:
        violation = (fuel - ga.prob.vessel.fuel_weight) / ga.prob.vessel.fuel_weight
        assert ind.violation_sum == violation


def test_get_n_best(ga):

    ind1 = ga.make_vessel_individual([[0, 1], [15, 20]])
    ind2 = ga.make_vessel_individual([[0, 1], [20, 20]])
    ind3 = ga.make_vessel_individual([[0, 1, 2], [15, 20]])
    ind4 = ga.make_vessel_individual([[6], [15, 20]])
    ind5 = ga.make_vessel_individual([[0, 3, 8], [10, 15, 15]])
    ind6 = ga.make_vessel_individual([[0, 3, 8], [10, 15, 20]])
    ind7 = ga.make_vessel_individual([[6, 8], [15, 20]])
    population = [ind1, ind2, ind3, ind4, ind5, ind6, ind7]
    n_best = ga.get_n_best(population)
    assert n_best[0] == ind5
    assert n_best[1] == ind6
    population2 = [ind1, ind3, ind5]
    n_best = ga.get_n_best(population2)
    assert n_best[0] == ind3
    assert n_best[1] == ind5
    ind8 = ga.make_vessel_individual([[0, 6], [15, 15]])
    ind9 = ga.make_vessel_individual([[4, 6], [20, 20]])
    population3 = [ind8, ind9, ind1]
    n_best = ga.get_n_best(population3)
    assert n_best[0] == ind8
    assert n_best[1] == ind9


def test_individual_comparison(ga):
    ind1 = ga.make_vessel_individual([[0, 1, 4, 7], [20, 20, 10, 10]])
    ind2 = ga.make_vessel_individual([[0, 1, 4, 7], [20, 20, 20, 20]])
    ind3 = ga.make_vessel_individual([[0, 1, 2], [15, 20, 20]])
    ind4 = ga.make_vessel_individual([[6], [15]])
    ind5 = ga.make_vessel_individual(
        [[0, 3, 6, 5, 4, 1, 2, 7], [20, 25, 20, 10, 15, 20, 20, 20]]
    )
    ind6 = ga.make_vessel_individual(
        [[0, 3, 6, 5, 4, 1, 2, 7], [25, 25, 25, 25, 25, 25, 25, 20]]
    )
    ind7 = ga.make_vessel_individual([[6, 8], [15, 20]])
    ind8 = ga.make_vessel_individual([[0, 6], [15, 15]])
    ind9 = ga.make_vessel_individual([[4, 6], [20, 20]])
    assert ga.individual_comparison(ind1, ind2) == ind1
    assert ga.individual_comparison(ind2, ind3) == ind2
    assert ga.individual_comparison(ind5, ind3) == ind5
    assert ga.individual_comparison(ind5, ind6) == ind5
    assert ga.individual_comparison(ind5, ind4) == ind5
    assert ga.individual_comparison(ind6, ind5) == ind5
    assert ga.individual_comparison(ind7, ind1) == ind1
    assert ga.individual_comparison(ind8, ind9) == ind9


def test_crossover(ga):
    chrom1 = [[0, 1, 2, 9], [20, 15, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 9], [20, 12, 20]]
    chrom4 = [[0, 4, 3, 6, 9], [20, 10, 15, 15, 20]]
    chrom5 = [[0, 6, 7, 9], [20, 15, 10, 20]]
    chrom6 = [[0, 3, 6, 9], [20, 10, 12, 20]]
    chrom7 = [[0, 4, 6, 9], [20, 12, 15, 20]]
    cross1 = ga.crossover(chrom1, chrom2)
    cross2 = ga.crossover(chrom2, chrom5)
    cross3 = ga.crossover(chrom7, chrom4)
    cross4 = ga.crossover(chrom6, chrom3)

    assert cross1[0].chromosome[0] == [0, 1, 2, 3, 9]
    assert cross1[0].chromosome[1] == [20, 15, 20, 15, 20]
    assert cross1[1].chromosome[0] == [0, 1, 2, 9]
    assert cross1[1].chromosome[1] == [20, 15, 20, 20]

    assert cross2[0].chromosome[0] == [0, 1, 7, 9]
    assert cross2[0].chromosome[1] == [20, 15, 10, 20]
    assert cross2[1].chromosome[0] == [0, 6, 2, 3, 9]
    assert cross2[1].chromosome[1] == [20, 15, 20, 15, 20]

    assert cross3[0].chromosome[0] == [0, 4, 3, 6, 9]
    assert cross3[0].chromosome[1] == [20, 12, 15, 15, 20]
    assert cross3[1].chromosome[0] == [0, 4, 6, 9]
    assert cross3[1].chromosome[1] == [20, 10, 15, 20]

    assert cross4[0].chromosome[0] == [0, 6, 9]
    assert cross4[0].chromosome[1] == [20, 12, 20]
    assert cross4[1].chromosome[0] == [0, 3, 6, 9]
    assert cross4[1].chromosome[1] == [20, 10, 12, 20]


def test_partial_reverse(ga):
    chrom1 = [[0, 1, 6, 3, 8, 7, 9], [20, 15, 12, 10, 10, 15, 20]]
    chrom2 = [[0, 6, 3, 5, 9], [20, 12, 15, 10, 20]]
    chrom3 = [[0, 6, 9], [20, 20, 20]]
    chrom4 = [[0, 5, 4, 3, 8, 9], [20, 15, 12, 15, 10, 20]]
    reverse1 = ga.partial_reverse_waypoints(chrom1)
    reverse2 = ga.partial_reverse_waypoints(chrom2)
    reverse3 = ga.partial_reverse_waypoints(chrom3)
    reverse4 = ga.partial_reverse_waypoints(chrom4)
    assert reverse1.chromosome[0] == [0, 1, 3, 6, 8, 7, 9]
    assert reverse1.chromosome[1] == [20, 15, 10, 12, 10, 15, 20]

    assert reverse2.chromosome[0] == [0, 6, 3, 5, 9]
    assert reverse2.chromosome[1] == [20, 12, 15, 10, 20]

    assert reverse3.chromosome[0] == [0, 6, 9]
    assert reverse3.chromosome[1] == [20, 20, 20]

    assert reverse4.chromosome[0] == [0, 5, 4, 8, 3, 9]
    assert reverse4.chromosome[1] == [20, 15, 12, 10, 15, 20]


def test_drop_waypoint(ga):
    chrom1 = [[0, 1, 2, 9], [20, 15, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 9], [20, 12, 20]]
    chrom4 = [[0, 4, 3, 8, 9], [20, 10, 15, 15, 20]]
    drop1 = ga.drop_waypoint(chrom1)
    drop2 = ga.drop_waypoint(chrom2)
    drop3 = ga.drop_waypoint(chrom3)
    drop4 = ga.drop_waypoint(chrom4)
    assert drop1.chromosome[0] == [0, 2, 9]
    assert drop1.chromosome[1] == [20, 20, 20]
    assert drop2.chromosome[0] == [0, 2, 3, 9]
    assert drop2.chromosome[1] == [20, 20, 15, 20]
    assert drop3.chromosome[0] == [0, 9]
    assert drop3.chromosome[1] == [20, 20]
    assert drop4.chromosome[0] == [0, 3, 8, 9]
    assert drop4.chromosome[1] == [20, 15, 15, 20]


def test_add_waypoint(ga):
    chrom1 = [[0, 1, 2, 9], [20, 15, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 9], [20, 12, 20]]
    chrom4 = [[0, 4, 3, 8, 9], [20, 10, 15, 15, 20]]
    add1 = ga.add_waypoint(chrom1)
    add2 = ga.add_waypoint(chrom2)
    add3 = ga.add_waypoint(chrom3)
    add4 = ga.add_waypoint(chrom4)
    assert add1.chromosome[0] == [0, 1, 2, 3, 9]
    assert add1.chromosome[1] == [20, 15, 10, 20, 20]
    assert add2.chromosome[0] == [0, 1, 2, 3, 8, 9]
    assert add2.chromosome[1] == [20, 15, 20, 10, 15, 20]
    assert add3.chromosome[0] == [0, 6, 8, 9]
    assert add3.chromosome[1] == [20, 6.666666666666667, 12, 20]
    assert add4.chromosome[0] == [0, 4, 3, 8, 6, 9]
    assert add4.chromosome[1] == [20, 10, 15, 6.666666666666667, 15, 20]


def test_replace_waypoint(ga):
    chrom1 = [[0, 1, 2, 9], [20, 15, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 9], [20, 12, 20]]
    chrom4 = [[0, 4, 3, 8, 9], [20, 10, 15, 15, 20]]
    replace1 = ga.replace_waypoint(chrom1)
    replace2 = ga.replace_waypoint(chrom2)
    replace3 = ga.replace_waypoint(chrom3)
    replace4 = ga.replace_waypoint(chrom4)
    assert replace1.chromosome[0] == [0, 3, 2, 9]
    assert replace1.chromosome[1] == [20, 5, 20, 20]
    assert replace2.chromosome[0] == [0, 8, 2, 3, 9]
    assert replace2.chromosome[1] == [20, 6.666666666666667, 20, 15, 20]
    assert replace3.chromosome[0] == [0, 5, 9]
    assert replace3.chromosome[1] == [20, 10, 20]
    assert replace4.chromosome[0] == [0, 4, 3, 1, 9]
    assert replace4.chromosome[1] == [20, 10, 15, 10, 20]


def test_two_pass_tournament_selection(ga):
    chrom1 = [[0, 1, 2, 9], [20, 15, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 9], [20, 11, 20]]
    chrom4 = [[0, 4, 3, 8, 9], [20, 10, 15, 15, 20]]
    chrom5 = [[0, 4, 6, 2, 9], [20, 15, 12, 15, 20]]
    chrom6 = [[0, 6, 9], [20, 20, 20]]
    chrom7 = [[0, 4, 6, 9], [20, 20, 20, 20]]
    chrom8 = [[0, 4, 6, 2, 9], [20, 15, 15, 15, 20]]
    ind1 = ga.make_vessel_individual(chrom1)
    ind2 = ga.make_vessel_individual(chrom2)
    ind3 = ga.make_vessel_individual(chrom3)
    ind4 = ga.make_vessel_individual(chrom4)
    ind5 = ga.make_vessel_individual(chrom5)
    ind6 = ga.make_vessel_individual(chrom6)
    ind7 = ga.make_vessel_individual(chrom7)
    ind8 = ga.make_vessel_individual(chrom8)
    pop = [ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8]
    new_pop = ga.two_pass_tournament_selection(pop)
    assert new_pop[0].chromosome == chrom8
    assert new_pop[1].chromosome == chrom5
    assert new_pop[2].chromosome == chrom5
    assert new_pop[3].chromosome == chrom4
    assert new_pop[4].chromosome == chrom4
    assert new_pop[5].chromosome == chrom7
    assert new_pop[6].chromosome == chrom3
    assert new_pop[7].chromosome == chrom8
    new_pop = ga.two_pass_tournament_selection(new_pop)
    assert new_pop[0].chromosome == chrom4
    assert new_pop[1].chromosome == chrom8
    assert new_pop[2].chromosome == chrom8
    assert new_pop[3].chromosome == chrom4
    assert new_pop[4].chromosome == chrom8
    assert new_pop[5].chromosome == chrom4
    assert new_pop[6].chromosome == chrom4
    assert new_pop[7].chromosome == chrom5
    new_pop = ga.two_pass_tournament_selection(new_pop)
    assert new_pop[0].chromosome == chrom4
    assert new_pop[1].chromosome == chrom4
    assert new_pop[2].chromosome == chrom4
    assert new_pop[3].chromosome == chrom4
    assert new_pop[4].chromosome == chrom5
    assert new_pop[5].chromosome == chrom5
    assert new_pop[6].chromosome == chrom4
    assert new_pop[7].chromosome == chrom4


def test_local_search_speed(ga):
    chrom1 = [[0, 1, 2, 7, 9], [20, 15, 10, 20, 20]]
    chrom2 = [[0, 1, 2, 3, 9], [20, 15, 20, 15, 20]]
    chrom3 = [[0, 6, 2, 3, 9], [10, 11, 10, 10, 10]]
    chrom4 = [[0, 4, 3, 8, 9], [20, 10, 15, 15, 20]]
    chrom5 = [[0, 4, 6, 2, 9], [20, 15, 12, 15, 20]]
    chrom6 = [[0, 6, 9], [20, 20, 20]]
    chrom7 = [[0, 4, 6, 9], [20, 20, 20, 20]]
    chrom8 = [[0, 4, 6, 2, 9], [20, 15, 15, 15, 20]]
    ind1 = ga.make_vessel_individual(chrom1)
    ind2 = ga.make_vessel_individual(chrom2)
    ind3 = ga.make_vessel_individual(chrom3)
    ind4 = ga.make_vessel_individual(chrom4)
    ind5 = ga.make_vessel_individual(chrom5)
    ind6 = ga.make_vessel_individual(chrom6)
    ind7 = ga.make_vessel_individual(chrom7)
    ind8 = ga.make_vessel_individual(chrom8)
    new1 = ga.local_search_speed(ind1, 3)
    assert new1 > ind1
    new2 = ga.local_search_speed(ind2, 4)
    assert new2 > ind2
    new3 = ga.local_search_speed(ind3, 5)
    assert new3 > ind3
    new4 = ga.local_search_speed(ind4, 4)
    assert new4 > ind4
    new5 = ga.local_search_speed(ind5, 3)
    assert new5 > ind5
    new6 = ga.local_search_speed(ind6, 5)
    assert new6 > ind6
    new7 = ga.local_search_speed(ind7, 3)
    assert new7 > ind7
    new8 = ga.local_search_speed(ind8, 4)
    assert new8 > ind8


def test_run(ga):
    opt_dests = ga.run(20)[0]
    assert opt_dests[0].waypoint.lat == 40
    assert opt_dests[0].waypoint.long == -139
    assert opt_dests[0].arrival_time == opt_dests[0].estimated_arrival_time
    assert opt_dests[1].waypoint.lat == 40
    assert opt_dests[1].waypoint.long == -139
    assert opt_dests[0].estimated_arrival_time <= opt_dests[1].estimated_arrival_time
    assert opt_dests[2].waypoint.lat == 40
    assert opt_dests[2].waypoint.long == -137.5
    assert opt_dests[1].estimated_arrival_time <= opt_dests[2].estimated_arrival_time
    assert opt_dests[3].waypoint.lat == 42
    assert opt_dests[3].waypoint.long == -138
    assert opt_dests[2].estimated_arrival_time <= opt_dests[3].estimated_arrival_time
    assert opt_dests[4].waypoint.lat == 41.5
    assert opt_dests[4].waypoint.long == -139
    assert opt_dests[3].estimated_arrival_time <= opt_dests[4].estimated_arrival_time
    assert opt_dests[5].waypoint.lat == 39.5
    assert opt_dests[5].waypoint.long == -136.5
    assert opt_dests[4].estimated_arrival_time <= opt_dests[5].estimated_arrival_time
    assert opt_dests[6].waypoint.lat == 40.5
    assert opt_dests[6].waypoint.long == -138
    assert opt_dests[5].estimated_arrival_time <= opt_dests[6].estimated_arrival_time
    assert opt_dests[7].waypoint.lat == 41.5
    assert opt_dests[7].waypoint.long == -139.5
    assert opt_dests[6].estimated_arrival_time <= opt_dests[7].estimated_arrival_time
    assert opt_dests[8].waypoint.lat == 40
    assert opt_dests[8].waypoint.long == -139
    assert opt_dests[7].estimated_arrival_time <= opt_dests[8].estimated_arrival_time
    assert Mission(opt_dests).total_priority == 210


destinations = [
    Destination(Waypoint(40, -137), datetime(2024, 12, 21, 5, 0), 7),
    Destination(Waypoint(40.5, -138), datetime(2024, 12, 22, 16, 0), 9),
    Destination(Waypoint(41.5, -139), datetime(2024, 12, 25, 19, 0), 47),
    Destination(Waypoint(41.5, -139.5), datetime(2024, 12, 25, 23, 0), 37),
    Destination(Waypoint(42, -138), datetime(2024, 12, 22, 15, 0), 77),
    Destination(Waypoint(40, -137.5), datetime(2024, 12, 19, 15, 0), 27),
    Destination(Waypoint(39.5, -136.5), datetime(2024, 12, 22, 15, 0), 6),
    Destination(Waypoint(40, -139), datetime(2024, 12, 21, 17, 0), 7),
]
mission = Mission(destinations)
vessel = vesselFuel(
    "vessel2",
    "test_sim2",
    Destination(Waypoint(40, -139), datetime(2024, 12, 20, 0, 0), 0),
    Destination(Waypoint(40, -139), datetime(2024, 12, 24, 7, 0), 0),
    mission,
    50,
    weath_input_dir_lat40_lon_neg_140,
    model_path=model_path,
    speed=20,
    speed_options=[15, 20, 25],
)
problem = VesselProblem(vessel, datetime(2024, 12, 20, 0, 0))
opt = Optimizer(problem, 94140, 20)
test_run(opt)
