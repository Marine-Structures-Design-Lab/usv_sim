'''
Vessel Simulation Database Manager

Authors: Sanjana Jain, Rachel Mecca

(c) 2024 Regents of the University of Michigan

'''
import sqlite3


class DatabaseManager:
    
    def __init__(self, db_path='Simulation/sim_db.db'):
        '''
         Database Manager Constructor
         -------------------------------------
         Parameters
         ------------------------------------
         :param db_path: string, path to database
        '''
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        self._setup_database()
        #self.clear_tables()
        self.cursor.execute("PRAGMA table_info(mission_waypoints)")
        print(self.cursor.fetchall())
        
    def _setup_database(self):
        '''
         Summary
         -------------------------------------
         Executes SQL query to set up vessel simulation database
        '''
        schema_script = """
        -- Table to store static information about vessels
        CREATE TABLE IF NOT EXISTS vessels (
        vessel_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sim_name TEXT NOT NULL,
        type TEXT,  -- e.g., Cargo, Fishing, Research
        capacity INTEGER,  -- e.g., number of containers, weight in tons, etc.
        system_health TEXT NOT NULL,
        fuel_level REAL NOT NULL
       );

       -- Table to store waypoint information
      CREATE TABLE IF NOT EXISTS waypoints (
      waypoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
      latitude REAL NOT NULL,
      longitude REAL NOT NULL,
      priority_score INTEGER NOT NULL --
    
      );

     -- Table to link missions to vessels
     CREATE TABLE IF NOT EXISTS missions (
     mission_id INTEGER PRIMARY KEY AUTOINCREMENT,
     vessel_id INTEGER NOT NULL,
     total_priority INTEGER NOT NULL,
     mission_name TEXT,  -- Optional: Name or description of the mission
     start_date DATETIME,  -- Optional: ISO 8601 format for mission start
     end_date DATETIME,  -- Optional: ISO 8601 format for mission end
     
     FOREIGN KEY (vessel_id) REFERENCES vessels(vessel_id)
    
     );

    -- Table to store mission waypoints and their order and goal times
    CREATE TABLE IF NOT EXISTS mission_waypoints (
    mission_waypoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mission_id INTEGER NOT NULL,
    waypoint_id INTEGER NOT NULL,
    replan_id INTEGER NOT NULL,
    waypoint_order INTEGER NOT NULL,
    goal_time DATETIME,  -- ISO 8601 format for the target time to reach the waypoint
    est_time DATETIME,  -- ISO 8601 format for the estimated/actual time to reach the waypoint
    FOREIGN KEY (mission_id) REFERENCES missions(mission_id),
    FOREIGN KEY (waypoint_id) REFERENCES waypoints(waypoint_id),
    UNIQUE (mission_id, replan_id, waypoint_order)
    );

    -- Table to store logs of vessel movements and statuses
    CREATE TABLE IF NOT EXISTS logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vessel_id INTEGER NOT NULL,
    mission_id INTEGER,
    timestamp DATETIME NOT NULL,  -- ISO 8601 format for log entry time
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    speed REAL,  -- Speed in knots
    system_health TEXT,
    fuel_level REAL,
    wave_height REAL,  -- Wave height in meters
    wave_period REAL,  -- Wave period in seconds
    FOREIGN KEY (vessel_id) REFERENCES vessels(vessel_id),
    FOREIGN KEY (mission_id) REFERENCES missions(mission_id)
    );
    -- Table to store logs of vessel Replans
    CREATE TABLE IF NOT EXISTS replans (
    replan_id INTEGER,
    vessel_id INTEGER NOT NULL,
    mission_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,  -- ISO 8601 format for replan_time
    latitude REAL NOT NULL, --location vessel was at when replan happened
    longitude REAL NOT NULL, --location vessel was at when replan happened
    total_priority INTEGER NOT NULL, --total priority score of replanned mission
    num_waypoints INTEGER NOT NULL, --number of waypoints in the planned mission
    reason TEXT, --Reason why replan occured
    FOREIGN KEY (vessel_id) REFERENCES vessels(vessel_id),
    FOREIGN KEY (mission_id) REFERENCES missions(mission_id),
    PRIMARY KEY (mission_id, replan_id)
);
        """
        self.cursor.executescript(schema_script)
        self.connection.commit()

    def clear_tables(self):
        '''
        Clears all tables in database.
        '''
        truncate_script = """
        DELETE FROM vessels;
        DELETE FROM waypoints;
        DELETE FROM missions;
        DELETE FROM mission_waypoints;
        DELETE FROM logs;
        DELETE FROM replans
        """
        self.cursor.executescript(truncate_script)
        self.connection.commit()
        print("CLEAR TABLES")

    def insert_vessel(self, name,sim_name, vessel_type, capacity, system_health, fuel_level):
        '''
        Summary
        ------------------------------------------------------------
        Inserts a vessel into the `vessels` table.
        
        Parameters
        ------------------------------------------------------------
        :param name: string, name of vessel
        :param sim_name: string, name of simulation
        :param vessel_type: string, e.g., Cargo, Fishing, Research
        :param capacity: integer, number of containers, weight in tons, etc.
        :param system_health: string
        :param fuel_level: float, fuel weight in kilonewtons
        
        Returns
        --------------------------------------------------
        int, last row id
        '''

        try:
            self.cursor.execute("""
            INSERT INTO vessels (name,sim_name, type, capacity, system_health, fuel_level)
            VALUES (?, ?, ?,?, ?, ?);
        """, (name, sim_name,vessel_type, capacity, system_health, fuel_level))
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

        
        return self.cursor.lastrowid

    def insert_mission(self, vessel_id,  total_priority ,mission_name=None,start_date=None, end_date=None):
        '''
        Summary
        ------------------------------------------------------------
        Inserts a mission into the `missions` table.
       
        Parameters
        ------------------------------------------------------------
        :param vessel_id: int, vessel id in vessels table
        :param total_priority: int, total priority score of mission
        :param mission_name: string, name of the mission
        :param start_date: datetime object, start time of the mission
        :param end_date: datetime object, goal end time of the mission
        
        Returns
        --------------------------------------------------
        int, last row id
        '''
        self.cursor.execute("""
            INSERT INTO missions (vessel_id, total_priority, mission_name,  start_date, end_date)
            VALUES (?, ?, ?,?, ?);
        """, (vessel_id, total_priority, mission_name, start_date, end_date))
        self.connection.commit()
        return self.cursor.lastrowid

    def insert_waypoint(self, latitude, longitude, priority_score):
        '''
        Summary
        ----------------------------------------------------------
        Inserts a waypoint into the `waypoints` table.
        
        Parameters
        ------------------------------------------------------------
        :param latitude: float, waypoint latitude
        :param longitude: float, waypoint longitude
        :param priority_score: int, waypoint priority score
        
        Returns
        --------------------------------------------------
        int, last row id
        '''
        self.cursor.execute("""
            INSERT INTO waypoints (latitude, longitude, priority_score)
            VALUES (?, ?, ?);
        """, (latitude, longitude, priority_score))
        self.connection.commit()
        return self.cursor.lastrowid

    def insert_mission_waypoint(self, mission_id, waypoint_id, replan_id, waypoint_order, goal_time, est_time):
        '''
        Summary
        -----------------------------------------------------------
        Inserts an entry into the `mission_waypoints` table.
        
        Parameters
        ------------------------------------------------------------
        :param mission_id: int, mission id in missions table
        :param waypoint_id: int, waypoint id in waypoints table
        :param replan_id: int, replan id in replan table
        :param waypoint_order: int, placement of waypoint in mission
        :param goal_time: datetime object, goal time to arrive at waypoint
        :param est_time: datetime object, estimated/actual time to arrive at waypoint
      
        Returns
        --------------------------------------------------
        int, last row id
        '''
        self.cursor.execute("""
            INSERT INTO mission_waypoints (mission_id, waypoint_id, replan_id, waypoint_order, goal_time, est_time)
            VALUES (?, ?, ?, ?, ?,?);
        """, (mission_id, waypoint_id, replan_id, waypoint_order, goal_time, est_time))
        self.connection.commit()

    def insert_log(self, vessel_id, mission_id, timestamp, latitude, longitude, speed, system_health, fuel_level, wave_height, wave_period):
        '''
        Summary
        ------------------------------------------------------------
        Inserts an entry into the `logs` table.
        
        Parameters
        ------------------------------------------------------------
        :param vessel_id: int, vessel id in vessels table
        :param mission_id: int, mission id in missions table
        :param timestamp: datetime object, timestamp of the log
        :param latitude: float, current latitude of vessel
        :param longitude: float, current longitude of vessel
        :param speed: float, current vessel speed
        :param system_health: string
        :param fuel_level: float, fuel weight in kilnewtons
        :param wave_height: float, current wave height
        :param wave_period: float, current wave period
        
        Returns
        --------------------------------------------------
        int, last row id
        '''
        self.cursor.execute("""
            INSERT INTO logs (vessel_id, mission_id, timestamp, latitude, longitude, speed, system_health, fuel_level, wave_height, wave_period)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (vessel_id, mission_id, timestamp, latitude, longitude, speed, system_health, fuel_level, wave_height, wave_period))
        self.connection.commit()
    
    def insert_replan(self,replan_id, vessel_id, mission_id, timestamp, latitude, longitude, total_prioriy, num_waypoints, reason):
        '''
        Summary
        ------------------------------------------------------------
        Inserts a mission replan into the 'replans' table.
       
        Parameters
        ------------------------------------------------------------
        :param replan_id: int, replan number (0 for inital mission plan, 1 for first replan, etc.)
        :param vessel_id: int, vessel id in vessels table
        :param mission_id: int, id of mission being replanned
        :param timestamp: datetime object, time of mission replan
        :param latitude: float, current vessel latitude
        :param longitude: float, current vessel longitude
        :param total_priority: int, total priority of mission replan
        :param num_waypoints: int, number of waypoints in mission replan
        :param reason: string, reason for mission replan
        
        Returns
        --------------------------------------------------
        int, last row id
        '''
        self.cursor.execute("""
            INSERT INTO replans (replan_id,vessel_id, mission_id, timestamp, latitude, longitude, total_priority, num_waypoints, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (replan_id,vessel_id, mission_id, timestamp, latitude, longitude, total_prioriy, num_waypoints, reason))
        self.connection.commit()
    
    def create_table(self, table_name, columns):
        '''
        Summary
        ----------------------------------------------------------------------
        Creates a new data table.
        
        Parameters
        -----------------------------------------------------------------------
        :param table_name: string, name of data table
        :param columns: dictionary with column names as keys and data types as values
        '''
        columns_with_types = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_with_types})"
        self.cursor.execute(create_table_sql)
        self.connection.commit()

    def insert_data(self, table_name, data):
        '''
        Summary
        ---------------------------------------------------------------------
        Inserts data into a table.
        
        Parameters
        --------------------------------------------------------------------
        :param table_name: string,name of data table
        :param data: dictionary with column names as keys and data values as values
        '''
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data.values()])
        values = list(data.values())
        insert_data_sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        self.cursor.execute(insert_data_sql, values)
        self.connection.commit()
    
    def drop_table(self,table_name):
        '''
        Summary
        ----------------------------------
        Removes a table from the database.
    
        Parameters
        ----------------------------------
        :param: table_name: string, name of table
        '''
        drop_table_sql = f"DROP TABLE IF EXISTS {table_name}"
        self.cursor.execute(drop_table_sql)
        self.connection.commit()

    def close(self):
        '''
        Summary
        ----------------------------------
        Closes the cursor and connection.
        '''
        self.cursor.close()
        self.connection.close()
  