# USV Simulation

## Overview

This project provides a simulation framework for evaluating autonomous vessel missions in dynamic ocean environments. The system models vessel behavior while executing missions and adapts mission plans in response to changing environmental conditions.

The mission planning algorithm is a genetic algorithm that searches for near-optimal missions. The planner selects waypoint order and vessel speeds while following constraints such as fuel capacity, waypoint priority, and ocean conditions.

The simulation engine continuously monitors environmental conditions during mission execution and can trigger replanning when hazardous weather is detected.

To model ocean conditions, the framework supports two weather prediction approaches:

- **Interpolated NOAA Data** – Forecasting based on interpolated NOAA weather datasets.
- **Neural Network Model** – A PyTorch-based ML model that predicts future wave conditions from current wave conditions.

These components allow evaluation of vessel mission performance under realistic and evolving ocean conditions.
## Features
- Autonomous vessel mission simulation
- Genetic algorithm mission planner
- Dynamic mission replanning under hazardous weather
- Ocean condition modeling using NOAA data interpolation
- Neural network wave prediction model 
- Parallel simulation runner for large-scale experiments
- Visualization of vessel missions

## Setting Up Your Environment

To ensure that the project dependencies don't interfere with other Python projects on your system, it's recommended to create a virtual environment. Follow these steps to set up your environment:

### 1. Create a Virtual Environment

Open your terminal and navigate to the project directory. Then, run the following command to create a virtual environment:

    python -m venv venv

This will create a new directory called `venv` in your project folder, which will contain a separate Python installation and all the necessary packages.

### 2. Activate the Virtual Environment

Before installing any packages, activate the virtual environment:

- **On Windows:**

  ```bash
  .\venv\Scripts\activate
  ```

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

You'll know the environment is active when you see `(venv)` at the beginning of your terminal prompt.

### 3. Install Dependencies

Once your virtual environment is activated, you can install all the required packages by running:

    pip install -r requirements.txt

This command will read the `requirements.txt` file and install the listed packages into your virtual environment.

In addition, the animation component of this program requires the user to install
external software designed to process audio and video. The implementation in this
repository makes use of the multimedia framework _FFmpeg_. The command to install _FFmpeg_
inside of Debian-based Linux distributions is shown below.

```
$ sudo apt install ffmpeg
```

## Troubleshooting

### Missing Library Errors

If you encounter an error indicating that a module or library is not installed, it will typically look something like this:

    ModuleNotFoundError: No module named 'example_library'

This error means that the required library `example_library` is not installed in your environment.

#### How to Fix It

To resolve this issue, you can manually install the missing library by running:

    pip install example_library

Replace `example_library` with the name of the missing module as shown in the error message. Once installed, try running your code again. If you encounter another missing module error, repeat the steps above for each missing library.

# Vessel Simulation Engine 
The simulation engine models the movement and behavior of vessels operating in ocean environments. It executes a mission planning algorithm to determine optimal routes to mission destinations and can dynamically replan while a mission is being executed.
### Running a Simulation

To run a simulation, you must:

1. Define a list of `Destination` objects 
2. Create a `Mission` object using the list of destinations.
3. Create one or more `Vessel` objects
4. Pass the vessels into a `Simulation` object
5. Run the simulation

A full runnable example is provided in: ```vessel_sim_engine.py```
From the project root directory, run:
    `python -m Simulation.vessel_sim_engine`
### Running Multiple Simulations in Parallel
To run large-scale experiments across multiple mission scenarios, a parallel simulation runner is provided in `sim_runner.py`. This script allows multiple simulations to be executed concurrently, making it easier to evaluate performance across many configurations or environmental conditions.

# Database Schema

### Overview

The database schema for the simulation is designed to manage and store information about vessels, their missions, waypoints, and related logs. This schema allows for the organization and retrieval of data related to vessel operations, mission planning, and simulation results.

### Tables

#### 1. vessels
This table stores information about each vessel in the simulation.

- `vessel_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each vessel.
- `name` (TEXT, NOT NULL): The name of the vessel.
- `sim_name` (TEXT, NOT NULL): The name of the simulation the vessel ran in.
- `type` (TEXT): The type of vessel, e.g., Cargo, Fishing, Research.
- `capacity` (INTEGER): The capacity of the vessel, e.g., the number of containers, weight in tons, etc.
- `system_health` (TEXT, NOT NULL): The current health status of the vessel's systems.
- `fuel_level` (REAL, NOT NULL): The current fuel level of the vessel.

#### 2. waypoints

This table stores the geographical waypoints used in missions.

- `waypoint_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each waypoint.
- `latitude` (REAL, NOT NULL): The latitude of the waypoint.
- `longitude` (REAL, NOT NULL): The longitude of the waypoint.
- `priority_score` (INTEGER, NOT NULL): The numerical priority score for the waypoint.

#### 3. missions

This table stores information about the missions assigned to vessels.

- `mission_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each mission.
- `vessel_id` (INTEGER, NOT NULL): The ID of the vessel assigned to this mission. This references the `vessel_id` in the vessels table.
- `total_priority` (INTEGER NOT NULL): The sum of the priority scores of all waypoints in the mission
- `mission_name` (TEXT): An optional name or description of the mission.
- `start_date` (DATETIME): The start date of the mission in ISO 8601 format.
- `end_date` (DATETIME): The end date of the mission in ISO 8601 format.

#### 4. mission_waypoints

This table defines the waypoints associated with each mission and the order in which they should be visited.

- `mission_waypoint_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each mission waypoint.
- `mission_id` (INTEGER, NOT NULL): The ID of the mission to which this waypoint belongs. This references the `mission_id` in the missions table.
- `waypoint_id` (INTEGER, NOT NULL): The ID of the waypoint to be visited. This references the `waypoint_id` in the waypoints table.
- `waypoint_order` (INTEGER, NOT NULL): The order in which the waypoint should be visited during the mission.
- `goal_time` (DATETIME): The target time to reach the waypoint, in ISO 8601 format.
- `est_time` (DATETIME): The estimate for the vessel's arrival time, updated to actual arrival time when the vessel reaches the waypoint
#### 5. logs

This table stores log entries that record various details about the vessel's status during a mission.

- `log_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each log entry.
- `vessel_id` (INTEGER, NOT NULL): The ID of the vessel associated with this log. This references the `vessel_id` in the vessels table.
- `mission_id` (INTEGER): The ID of the mission associated with this log. This references the `mission_id` in the missions table.
- `timestamp` (DATETIME, NOT NULL): The time at which the log entry was recorded, in ISO 8601 format.
- `latitude` (REAL, NOT NULL): The latitude of the vessel at the time of the log entry.
- `longitude` (REAL, NOT NULL): The longitude of the vessel at the time of the log entry.
- `speed` (REAL): The speed of the vessel at the time of the log entry, measured in knots.
- `system_health` (TEXT): The health status of the vessel's systems at the time of the log entry.
- `fuel_level` (REAL): The fuel level of the vessel at the time of the log entry.
- `wave_height` (REAL): The wave height in meters at the time of the log entry.
- `wave_period` (REAL): The wave period in seconds at the time of the log entry.

#### 6. replans 

This table stores details about any mission replans

 - `replan_id` (INTEGER, PRIMARY KEY, AUTOINCREMENT): A unique identifier for each replan.
 - `vessel_id` (INTEGER, NOT NULL): The ID of the vessel associated with this log. This references the `vessel_id` in the vessels table.
- `mission_id` (INTEGER): The ID of the mission associated with this log. This references the `mission_id` in the missions table.
- `timestamp` (DATETIME, NOT NULL): The time at which the replan occurred, in ISO 8601 format.
- `latitude` (REAL NOT NULL): Latitude vessel was at when replan happened
- `longitude` (REAL NOT NULL): Longitude vessel was at when replan happened
- `total_priority` (INTEGER NOT NULL): Total priority score of replanned mission
- `num_waypoints` (INTEGER NOT NULL): Number of waypoints in the replanned mission
- `reason` (TEXT): Reason why replan occurred









 