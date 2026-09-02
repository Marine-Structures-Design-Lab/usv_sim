"""
Creates visual representations of the data from tables in sim_schema.

Authors: Rachel Mecca

(c) 2024 Regents of the University of Michigan
"""

from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.transforms import Affine2D
from geographiclib.geodesic import Geodesic
import sqlite3
import time
import logging
import numpy as np
import pandas as pd

matplotlib.use("TKAgg")
geod = Geodesic.WGS84
logger = logging.getLogger(__name__)


def insert_initial_data(cursor):
    """
    Adds sample data into waypoints, vessels, missions, and mission_waypoints
    tables for testing.
    -------------------------------------------------------------------------
    Parameters
    -------------------------------------------------------------------------
    :param cursor: SQL cursor

    Returns
    -------------------------------------------------------------------------
    list of ints: vessel ids, 2D list of ints: waypoint ids for each vessel id
    """
    # Insert vessels
    vessels = [
        ("vessel1", "cargo", 35, "good", 500),
        ("vessel2", "fishing", 45, "bad", 550),
    ]
    vessel_ids = []
    for vessel in vessels:
        cursor.execute(
            """
            INSERT INTO vessels (name, sim_name, type, capacity, system_health, fuel_level)
            VALUES (?, ?, ?, ?, ?, ?);
        """,
            (vessel[0], "Test_Sim", vessel[1], vessel[2], vessel[3], vessel[4]),
        )
        vessel_id = cursor.lastrowid
        vessel_ids.append(vessel_id)

    # Insert waypoints and fetch waypoint IDs
    waypoint_id_groups = []
    destinations1 = [
        (36, -118.0, 0),  # Starting position
        (35, -119, 15),  # Intermediate waypoint1
        (37, -118, 5),  # Intermediate waypoint2
        (45, -135, 51),  # Intermediate waypoint3
        (38, -119, 3),  # Intermediate waypoint4
        (36, -117, 12),  # Intermediate waypoint5
        (34, -120, 0),  # Final destination
    ]
    destinations2 = [
        (31, -120.0, 0),  # Starting position2
        (32, -119, 15),  # Intermediate waypoint2
        (35, -119, 5),  # Intermediate waypoint1
        (37, -118, 25),  # Intermediate waypoint2
        (45, -135, 4),  # Intermediate waypoint3
        (38, -119, 20),  # Intermediate waypoint4
        (36, -117, 12),  # Intermediate waypoint5
        (33, -118, 0),  # Final destination2
    ]
    destination_groups = [destinations1, destinations2]
    opt_dests1 = [
        destinations1[0],
        destinations1[3],
        destinations1[2],
        destinations1[-1],
    ]
    opt_dests2 = [
        destinations2[0],
        destinations2[4],
        destinations2[1],
        destinations2[-1],
    ]
    replan1_1 = [
        destinations1[0],
        destinations1[3],
        destinations1[1],
        destinations1[5],
        destinations1[-1],
    ]
    replan1_2 = [
        destinations1[0],
        destinations1[3],
        destinations1[1],
        destinations1[4],
        destinations1[-1],
    ]
    plans1 = [opt_dests1, replan1_1, replan1_2]
    plans2 = [opt_dests2]
    plan_groups = [plans1, plans2]
    replan_dets1 = [
        (39, -134, datetime(2018, 1, 7, 8, 30), "danger detected"),
        (35.5, -118, datetime(2018, 1, 7, 9, 30), "danger detected"),
    ]
    replan_dets2 = []
    replan_groups = [replan_dets1, replan_dets2]
    waypoint_id_map = {}
    for group in destination_groups:
        for destination in group:
            cursor.execute(
                """
            INSERT INTO waypoints (latitude, longitude, priority_score)
            VALUES (?, ?, ?);
        """,
                (destination[0], destination[1], destination[2]),
            )
            waypoint_id_map[tuple(destination)] = cursor.lastrowid

    # Now build the optimal waypoint ID groups
    waypoint_id_groups = []
    for plan_group in plan_groups:
        waypoint_plans = []
        for plan in plan_group:
            waypoint_ids = []
            for destination in plan:
                waypoint_ids.append(waypoint_id_map[tuple(destination)])
            waypoint_plans.append(waypoint_ids)
        waypoint_id_groups.append(waypoint_plans)

    # Insert missions
    mission_ids = []
    for i in range(0, len(vessel_ids)):
        total = 0
        for dest in destination_groups[i]:
            total += dest[2]
        cursor.execute(
            """
            INSERT INTO missions (vessel_id,total_priority, mission_name, start_date, end_date)
            VALUES (?, ?, ?, ?, ?);
        """,
            (
                vessel_ids[i],
                total,
                "mission1",
                datetime(2018, 1, 1, 7),
                datetime(2018, 1, 2, 7),
            ),
        )
        mission_ids.append(cursor.lastrowid)

    # Associate waypoints with missions
    for i in range(len(plan_groups)):
        for j in range(len(plan_groups[i])):
            count = 0
            total = 0
            for dest in plan_groups[i][j]:
                total += dest[2]
            for order, destination in enumerate(plan_groups[i][j]):
                cursor.execute(
                    """
            INSERT INTO mission_waypoints (mission_id, waypoint_id, replan_id, waypoint_order, goal_time)
            VALUES (?, ?,?, ?, ?);
        """,
                    (
                        mission_ids[i],
                        waypoint_id_groups[i][j][count],
                        j,
                        order,
                        destination[1],
                    ),
                )
                count += 1
            if j != 0:
                cursor.execute(
                    """
            INSERT INTO replans (replan_id, vessel_id, mission_id, timestamp, latitude, longitude, total_priority, num_waypoints, reason)
            VALUES (?, ?,?, ?,?,?,?,?,?);
          """,
                    (
                        j,
                        vessel_ids[i],
                        mission_ids[i],
                        replan_groups[i][j - 1][2],
                        replan_groups[i][j - 1][0],
                        replan_groups[i][j - 1][1],
                        total,
                        len(plan_groups[i][j]),
                        replan_groups[i][j - 1][3],
                    ),
                )
    print("data inserted")
    return vessel_ids, waypoint_id_groups


def simulate_mission(cursor, vessel_ids, waypoint_id_groups):
    """
    Adds sample data into the log table for testing.
    -----------------------------------------------
    Parameters
    ----------------------------------------------
    :param cursor: SQL cursor
    :param vessel_ids: list of ints representing vessel table ids
    :param waypoint_id_groups: 2D list of ints, waypoint table ids for each vessel
    """
    done = False
    counter = 0
    timestamp = datetime(2018, 1, 1, 7)
    replan_counts = [0] * len(vessel_ids)
    while not done:
        for i in range(len(vessel_ids)):

            if len(waypoint_id_groups[i][replan_counts[i]]) > counter:
                # Get next replan details
                cursor.execute(
                    """
      SELECT timestamp, latitude,longitude,reason FROM replans WHERE vessel_id = ? AND replan_id= ?;
       """,
                    (vessel_ids[i], replan_counts[i] + 1),
                )
                row = cursor.fetchone()
                if row is not None:
                    replan_time, replan_lat, replan_lon, reason = row
                    if datetime.fromisoformat(replan_time) >= timestamp:
                        replan_counts[i] += 1
                        print(
                            f"vessel{vessel_ids[i]} replanned mission at ({replan_lat},{replan_lon}) due to {reason}"
                        )
                # Get the waypoint details
                cursor.execute(
                    """
      SELECT latitude, longitude FROM waypoints WHERE waypoint_id = ?;
     """,
                    (waypoint_id_groups[i][replan_counts[i]][counter],),
                )

                lat, lon = cursor.fetchone()

                # Log the new position
                cursor.execute(
                    """
            INSERT INTO logs (vessel_id, timestamp, latitude, longitude, speed, system_health, fuel_level)
            VALUES (?, ?, ?, ?, 12.0, 'Good', 100.0);
       """,
                    (vessel_ids[i], timestamp, lat, lon),
                )

                done = False

            else:
                done = True
        counter += 1
        # Simulate some processing time
        timestamp += timedelta(hours=1)
        time.sleep(1)


def getAllData(db_filepath, sim_name):
    """
    reads and returns data from schema.
    -----------------------------------
    Parameters
    -----------------------------------
    :param db_filepath: string, filepath to SQL database
    :param sim_name: string, name of simulation to get data from

    Returns
    ---------------------------------------------
    All data from vessels, waypoints, missions, mission_waypoints, replans, and logs tables
    """
    connection = sqlite3.connect(db_filepath)
    cursor = connection.cursor()

    # 1. Get vessels with the given sim_name
    cursor.execute(
        """
    SELECT MIN(vessel_id) AS vessel_id, name, sim_name
    FROM vessels
    WHERE sim_name = ?
    GROUP BY name, sim_name
     """,
        (sim_name,),
    )

    vessels_data = cursor.fetchall()
    vessel_ids = [row[0] for row in vessels_data]

    if not vessel_ids:
        # If no vessels found, return empty lists for everything
        cursor.close()
        connection.close()
        return [], [], [], [], [], []

    # 2. Get missions for these vessels
    query = (
        f'SELECT * FROM missions WHERE vessel_id IN ({",".join("?"*len(vessel_ids))})'
    )
    cursor.execute(query, vessel_ids)
    missions_data = cursor.fetchall()
    mission_ids = [row[0] for row in missions_data]  # mission_id is column 0

    # 3. Get replans for these vessels
    query = (
        f'SELECT * FROM replans WHERE vessel_id IN ({",".join("?"*len(vessel_ids))})'
    )
    cursor.execute(query, vessel_ids)
    replan_data = cursor.fetchall()

    # 4. Get logs for these vessels
    query = f'SELECT * FROM logs WHERE vessel_id IN ({",".join("?"*len(vessel_ids))})'
    cursor.execute(query, vessel_ids)
    logs_data = cursor.fetchall()

    # 5. Get mission_waypoints for selected missions
    if mission_ids:
        query = f'SELECT * FROM mission_waypoints WHERE mission_id IN ({",".join("?"*len(mission_ids))})'
        cursor.execute(query, mission_ids)
        mission_waypoints_data = cursor.fetchall()
        # Get unique waypoint_ids from mission_waypoints
        waypoint_ids = list(
            set(row[2] for row in mission_waypoints_data)
        )  # waypoint_id is column 2
    else:
        mission_waypoints_data = []
        waypoint_ids = []

    # 6. Get only waypoints that are used in the selected mission_waypoints
    if waypoint_ids:
        query = f'SELECT * FROM waypoints WHERE waypoint_id IN ({",".join("?"*len(waypoint_ids))})'
        cursor.execute(query, waypoint_ids)
        waypoints_data = cursor.fetchall()
    else:
        waypoints_data = []

    cursor.close()
    connection.close()

    return (
        vessels_data,
        waypoints_data,
        missions_data,
        mission_waypoints_data,
        replan_data,
        logs_data,
    )


def run_animation(db_filepath, sim_name, timestep):
    """
    runs animation of vessels sailing to waypoints using data in database
    ----------------------------------------------------------------------
    Parameters
    ----------------------------------------------------------------------
    :param db_filepath: string, filepath to SQL database
    :param sim_name: string, name of simulation to animate
    :param timestep: float, number of seconds between each vessel movement
    """
    # Set up database
    (
        vessels_data,
        waypoints_data,
        missions_data,
        mission_waypoints_data,
        replan_data,
        logs_data,
    ) = getAllData(db_filepath, sim_name)
    vessel_id_to_name = {v[0]: v[1] for v in vessels_data}
    colors = [
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

    sc = []  # list of ship icons
    vesselIds = []
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    counter = 0

    # Initial waypoint setup
    setup_waypoints(waypoints_data, ax)

    # Initial vessel setup
    for i in range(len(vessels_data)):
        color = colors[counter % len(colors)]
        sc.append(
            ax.scatter(
                0, 0, marker="^", color=color, s=100, transform=ccrs.PlateCarree()
            )
        )
        for row in missions_data:
            if vessels_data[i][0] == row[0]:
                mission_id = row[0]

        vesselIds.append(vessels_data[i][0])
        counter += 1

    # Run animation
    plt.ion()

    # Used to calculate headings
    previous_coordinates = [(0, 0) for _ in range(len(vesselIds))]

    # For replan data
    sorted_replans = sorted(replan_data, key=lambda x: x[3])  # Sort by timestamp
    printed_replans = set()

    for row in logs_data:
        log_time = row[3]
        name = vessel_id_to_name.get(row[1], f"Unknown({row[1]})")
        # Check for replans at this log time
        for repl in sorted_replans:
            replan_key = (repl[1], repl[0])  # (vessel_id, replan_id)
            repl_time = repl[3]

            if repl_time <= log_time and replan_key not in printed_replans:
                # Print replan info

                print(f"🔄 {name} replanned at {repl_time}")
                print(f"    Location: ({repl[4]:.2f}, {repl[5]:.2f})")
                print(f"    New Total Priority: {repl[6]}")
                if repl[7]:
                    print(f"    Reason: {repl[7]}")
                else:
                    print(f"    Reason: [No reason provided]")
                print("-" * 50)

                printed_replans.add(replan_key)

        # Update vessels
        for i in range(len(vesselIds)):
            if row[1] == vesselIds[i]:
                g = geod.Inverse(
                    previous_coordinates[i][0],
                    previous_coordinates[i][1],
                    row[4],
                    row[5],
                )
                heading = g["azi1"]
                sc[i].set_offsets((row[5], row[4]))
                update_marker_rotation(sc[i], heading)
                ax.set_title(row[3], fontsize=15)
                previous_coordinates[i] = (row[4], row[5])
                plt.pause(timestep)

        print(
            f"{datetime.fromisoformat(log_time)+timedelta(hours=timestep)}: {name} at ({row[4]},{row[5]}) with speed {row[6]} knots and fuel weight {row[8]}"
        )

    plt.ioff()
    plt.show()

    print("Simulation ended.")


def make_fuel_graph(db_filepath, sim_name, vesselNames=[]):
    """
    Makes a line graph showing the fuelweight over time
    -------------------------------------------------------
    Parameters
    -------------------------------------------------------
    :param db_filepath: string, filepath to SQL database
    :param sim_name: string, name of simulation
    :param vesselNames: list of vessels to be displyed on the graph.
                         If empty all vessels in the database are displayed.
    """
    (
        vessels_data,
        waypoints_data,
        missions_data,
        mission_waypoints_data,
        replan_data,
        logs_data,
    ) = getAllData(db_filepath, sim_name)
    vesselIds = []

    colors = [
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
    if len(vesselNames) == 0:
        for i in range(len(vessels_data)):
            vesselNames.append(vessels_data[i][1])
    for i in range(len(vesselNames)):
        found = False
        for row in vessels_data:
            if row[1] == vesselNames[i]:
                found = True
                vesselIds.append(row[0])
        if not found:
            logger.warning(f"{vesselNames[i]} not found in vessels_data")
            del vesselNames[i]
    for i in range(len(vesselNames)):
        start_time = datetime(2018, 1, 1, 0, 0)
        times = []
        fuel_weights = []
        count = 0
        for row in logs_data:
            if row[1] == vesselIds[i]:
                if count == 0:
                    start_time = datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S")
                    count = 1
                time_difference = abs(
                    (
                        datetime.strptime(row[3], "%Y-%m-%d %H:%M:%S") - start_time
                    ).total_seconds()
                )
                hrs_difference = int(time_difference // 3600)
                times.append(hrs_difference)
                fuel_weights.append(row[8])

        plt.plot(times, fuel_weights, label=vesselNames[i])

    plt.legend()
    plt.title("Vessel Fuel Weight Over Time", fontweight="bold")
    plt.xlabel("Time (hours)", fontweight="bold")
    plt.ylabel("Fuel Weight (kN)", fontweight="bold")
    plt.show()


def setup_waypoints(waypoints_data, ax):
    """
    Determines map size and places waypoint icons.
    ----------------------------------------------

    Parameters
    ----------------------------------------------
    :param waypoints_data: data from "waypoints" table in SQL database
    :param ax: pyplot Geoaxes object for map
    """
    lon_min = 180
    lon_max = -180
    lat_min = 90
    lat_max = -90
    for row in waypoints_data:
        lon = row[2]
        lat = row[1]
        lon_min = min(lon_min, lon)
        lon_max = max(lon_max, lon)
        lat_min = min(lat_min, lat)
        lat_max = max(lat_max, lat)
        ax.scatter(
            row[2],
            row[1],
            marker=".",
            color="white",
            s=100,
            transform=ccrs.PlateCarree(),
        )
    # Add padding of 5 degrees to each side
    padding = 5
    lon_min -= padding
    lon_max += padding
    lat_min -= padding
    lat_max += padding

    makeMap(lon_min, lon_max, lat_min, lat_max, ax, datetime(2018, 1, 1, 0, 0))


def makeMap(lon_min, lon_max, lat_min, lat_max, ax, start_time):
    """
    Makes and sizes the map based on given dimmensions.
    ---------------------------------------------------

    Parameters
    ---------------------------------------------------
    :param lon_min: float, minimum longitude of the map
    :param lon_max: float, maximum longitude of the map
    :param lat_min: float, minimum latitude of the map
    :param lat_max: float, maximum latitude of the map
    :param ax: pyplot Geoaxes object for map
    :param start_time: datetime object, timestamp the simulation starts at
    """
    ax.coastlines()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor("blue")
    ax.set_title(start_time, fontsize=15)


def update_marker_rotation(sc, heading):
    """
    Adjusts the rotation of the vessel marker based on the heading
    --------------------------------------------------------------

    Parameters
    ------------------------------------------------------------
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


def simulation_compare(db_filepath, sim_list, display_all_data=True):
    """
    Makes a table comparing priority score, number of replans, and waypoints hit
    for each vessel in sim_list
    -------------------------------------------------------

    Parameters
    -------------------------------------------------------
    :param db_filepath: string, filepath to SQL database
    :param sim_list: list of simulations to be used  in the comparison.
    :param dispaly_all_data: if true displays all data for each vessel, else just displays the mean values per vessel and fleet
    """
    vessels_data_list = []
    replans_data_list = []
    for sim in sim_list:
        vessels_data, _, _, _, replan_data, _ = getAllData(db_filepath, sim)
        vessels_data_list.append(vessels_data)
        replans_data_list.append(replan_data)

    # vessel_name → vessel_id mapping per sim
    vessel_mappings = []
    for vd in vessels_data_list:
        vessel_mappings.append({v[1]: v[0] for v in vd})

    def extract_metrics(vessel_map, replan_data):
        metrics = {}
        for vessel_name, vessel_id in vessel_map.items():
            total_priority = None
            waypoints_hit = None
            replan_count = 0
            added = False
            for i in range(len(replan_data) - 1, -1, -1):
                if replan_data[i][1] == vessel_id:
                    replan_count += 1
                    if not added:  # first encounter from the end
                        total_priority = replan_data[i][6]
                        waypoints_hit = replan_data[i][7]
                        added = True
            metrics[vessel_name] = {
                "Total Priority": total_priority,
                "Waypoints Hit": waypoints_hit,
                "Replans": replan_count,
            }
        return metrics

    # build metrics for each sim
    metrics_list = []
    for i in range(len(vessel_mappings)):
        metrics_list.append(extract_metrics(vessel_mappings[i], replans_data_list[i]))

    # collect all vessel names across sims
    all_names = set()
    for m in metrics_list:
        all_names |= set(m.keys())

    rows = []
    for vessel_name in sorted(all_names):
        row = {"Vessel Name": vessel_name}
        if display_all_data:
            # include all sim results
            for i, sim in enumerate(sim_list):
                metrics = metrics_list[i].get(vessel_name, {})
                row[f"Priorities ({sim})"] = metrics.get("Total Priority")
                row[f"Waypoints ({sim})"] = metrics.get("Waypoints Hit")
                row[f"Replans ({sim})"] = metrics.get("Replans")
        else:
            # compute averages directly
            priorities = []
            waypoints = []
            replans = []
            for i in range(len(sim_list)):
                metrics = metrics_list[i].get(vessel_name, {})
                if metrics.get("Total Priority") is not None:
                    priorities.append(metrics["Total Priority"])
                if metrics.get("Waypoints Hit") is not None:
                    waypoints.append(metrics["Waypoints Hit"])
                if metrics.get("Replans") is not None:
                    replans.append(metrics["Replans"])
            row["Avg Priorities"] = np.mean(priorities) if priorities else None
            row["Avg Waypoints"] = np.mean(waypoints) if waypoints else None
            row["Avg Replans"] = np.mean(replans) if replans else None

        rows.append(row)

    df = pd.DataFrame(rows)

    # per-vessel averages
    priority_cols = [c for c in df.columns if "Priorities" in c]
    waypoint_cols = [c for c in df.columns if "Waypoints" in c]
    replan_cols = [c for c in df.columns if "Replans" in c]

    df["Avg Priorities"] = (
        df[priority_cols].mean(axis=1, skipna=True) if priority_cols else None
    )
    df["Avg Waypoints"] = (
        df[waypoint_cols].mean(axis=1, skipna=True) if waypoint_cols else None
    )
    df["Avg Replans"] = (
        df[replan_cols].mean(axis=1, skipna=True) if replan_cols else None
    )

    if not display_all_data:
        # keep only vessel + averages
        df = df[["Vessel Name", "Avg Priorities", "Avg Waypoints", "Avg Replans"]]

    # fleet averages
    fleet_avg = {"Vessel Name": "Fleet Average"}
    for col in df.columns:
        if col != "Vessel Name":
            if pd.api.types.is_numeric_dtype(df[col]):
                fleet_avg[col] = df[col].mean(skipna=True)
            else:
                fleet_avg[col] = None

    df = pd.concat([df, pd.DataFrame([fleet_avg])], ignore_index=True)

    # rounding
    for col in df.columns:
        if col != "Vessel Name":
            df[col] = pd.to_numeric(df[col])
    df = df.round(2)

    print(df.to_string(index=False))
    return df


def main():
    db_path = "Simulation/sim_db.db"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    truncate_script = """
        DELETE FROM vessels;
        DELETE FROM waypoints;
        DELETE FROM missions;
        DELETE FROM mission_waypoints;
        DELETE FROM logs;
        DELETE FROM replans
        """
    cursor.executescript(truncate_script)
    connection.commit()

    # Set up initial data
    vessel_ids, waypoint_id_groups = insert_initial_data(cursor)
    connection.commit()

    # Simulate the mission
    simulate_mission(cursor, vessel_ids, waypoint_id_groups)

    connection.commit()
    run_animation(db_path, "Test_Sim", 1)
    make_fuel_graph(db_path, "Test_Sim")
    cursor.close()
    connection.close()

    print(f"Database operations completed successfully on {db_path}")


if __name__ == "__main__":
    main()
