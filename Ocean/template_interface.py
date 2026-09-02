#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 1 13:38:56 2023

@author: tjflores
"""
from netCDF4 import Dataset, num2date
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import matplotlib.pyplot as plt
from bisect import bisect
import numpy as np
import calendar


class OceanMod:

    def __init__(self):
        """
        Creates OceanMod object and fills metadata dictionaries with 'None'.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        self.params = {}
        self.units = {}
        self.interpolators = {}
        self.data = {}
        self.file = None

    def load_data(self, filename):
        """
        Load any HDF5 file of ocean data from NOAA/EU systems.

        Parameters
        ----------
        filename : *.nc file
            HDF5 file in netCDF4 format containing model backend

        Returns
        -------
        None.

        """
        fin = Dataset(filename)
        fin.set_auto_mask(False)
        vars = fin.variables.keys()

        # load time, latitude, and longitude dimensions
        self.time = num2date(
            fin.variables["time"][:],
            fin.variables["time"].units,
            only_use_cftime_datetimes=False,
        )
        self.latitude = fin.variables["lat"][:]
        self.longitude = fin.variables["lon"][:]

        # fill interpolator variable with n interpolation objects
        def to_float(d):
            return calendar.timegm(d.timetuple())

        time_dim = list(map(to_float, self.time))
        lat_dim = self.latitude
        lon_dim = self.longitude
        interpolation_vars = [
            v for v in vars if v != "lat" and v != "lon" and v != "time"
        ]
        for v in interpolation_vars:
            self.data[v] = fin.variables[v][:]
            self.interpolators[v] = RegularGridInterpolator(
                (time_dim, lat_dim, lon_dim), fin.variables[v][:]
            )

        # update param dictionary
        self.params["time"] = str(self.time.min()), str(self.time.max())
        self.params["lat"] = self.latitude.min(), self.latitude.max()
        self.params["lon"] = self.longitude.min(), self.longitude.max()

        # update units dictionary
        for u in vars:
            self.units[u] = fin.variables[u].units

        # Update file
        self.file = filename

        # close file
        fin.close()

    def get_params(self):
        """
        Returns a dictionary object specifying the domain of the model

        Parameters
        ----------
        None.

        Returns
        -------
        params : Python dictionary
            maps model parameters to their valid range (inclusive) as a tuple

        """
        return self.params

    def get_units(self):
        """
        Returns a dictionary object specifying model variables' units of
        measurement

        Parameters
        ----------
        None.

        Returns
        -------
        units : Python dictionary
            maps model variables to their units of measurement

        """
        return self.units

    def weather_mission(self, dates, coordinates):
        """
        Given parallel lists of dates and latitude/longitude coordinates,
        return a list of n-element tuples representing interpolated values
        for variables in the model and a singular n-element tuple representing
        the names of the variables the interpolated data refers to

        Parameters
        ----------
        dates : list of Python native datetime objects
            one dimensional list of datetime objects to interpolate values for
        coordinates : list of tuples of floats
            one dimensional list of geographical coordinates to interpolate
            values for. Each element is a tuple taking the form (lat, lon)

        Returns
        -------
        data : list of tuples of floats
            one dimensional list of interpolated values for each variable
            of interest. Each element is an n-element tuple of floats
            representing the interpolated values for each variable
        var_names : tuple of strings
            n-element tuple of the names of the variables that map to the
            values in data. The variable names appear in the same order
            as their respective values do in each element of data

        """
        dates = list(map(lambda d: calendar.timegm(d.timetuple()), dates))
        if len(dates) != len(coordinates):
            raise ValueError('"dates" and "coordinates" are not equal length')
        if self.file is None:
            raise AttributeError("data has not been loaded into the model yet")
        input = []
        for time, (lat, lon) in zip(dates, coordinates):
            input.append([time, lat, lon])
        try:
            var_names = sorted(list(self.interpolators.keys()))
            temp = []
            for v in var_names:
                temp.append(self.interpolators[v](input))
        except ValueError as e:
            raise ValueError(
                "one or more input values is outside" "of the model domain"
            ) from e
        return list(zip(*temp)), var_names

    def plot_grid(self, timestamp, var_type):
        """
        Given an instance in time and a requested variable type, generate
        plot(s) of the grid and return .png file(s) to the current working
        directory. Plot types include contour plot and vector field

        Parameters
        ----------
        timestamp : Python native datetime object
            Timestamp that the returned plot represents. If the requested
            timestamp is not a discrete unit on the grid, the values are
            linearly interpolated
        var_type : string
            Type of variable in the model that should be plotted. Accepted
            values are "wind" (generates a vector field), "wave" (generates
            an overlaid significant wave height contour and wave direction
            vector field and a separate wave direction contour), and
            "pressure" (generates a contour plot)

        Returns
        -------
        1 (2 for var_type = "wave") .png file to the current working directory
        """
        if var_type not in ["wind", "wave", "pressure"]:
            raise ValueError(
                '"{}" is not a valid value for var_type. Supported values are '
                '"wind", "wave", and "pressure"'.format(var_type)
            )
        if (
            var_type == "pressure"
            and "surface_pressure" not in self.interpolators.keys()
        ):
            raise ValueError(
                '"pressure" is not one of the variables '
                "contained in this instance of the model"
            )
        start = datetime.fromisoformat(self.params["time"][0])
        end = datetime.fromisoformat(self.params["time"][1])
        if timestamp < start or timestamp > end:
            raise ValueError(
                "{} is outside of the model's time domain".format(timestamp)
            )
        fig = plt.figure()
        plt.xlabel("Longitude (degrees east)")
        plt.ylabel("Latitude (degrees north)")
        X, Y = np.meshgrid(self.longitude, self.latitude)
        left = bisect(self.time, timestamp) - 1
        latstep = int(len(self.latitude) / 10)
        lonstep = int(len(self.longitude) / 10)
        if var_type == "wind":
            if timestamp in self.time:
                ws = self.data["wind_speed"][:][left, :, :]
                wd = self.data["wind_direction"][:][left, :, :]
            else:
                right = left + 1
                ws_left = self.data["wind_speed"][:][left, :, :]
                ws_right = self.data["wind_speed"][:][right, :, :]
                wd_left = self.data["wind_direction"][:][left, :, :]
                wd_right = self.data["wind_direction"][:][right, :, :]
                delta = self.time[right] - self.time[left]
                ws = ws_left * ((self.time[right] - timestamp) / delta) + ws_right * (
                    (timestamp - self.time[left]) / delta
                )
                wd = wd_left * ((self.time[right] - timestamp) / delta) + wd_right * (
                    (timestamp - self.time[left]) / delta
                )
            wd = np.radians((((450 - wd) % 360) + 180) % 360)
            U = ws * np.cos(wd)
            V = ws * np.sin(wd)
            Q = plt.quiver(
                X[::latstep, ::lonstep],
                Y[::latstep, ::lonstep],
                U[::latstep, ::lonstep],
                V[::latstep, ::lonstep],
                color="g",
                pivot="middle",
            )
            qk = plt.quiverkey(
                Q,
                0.95,
                0.95,
                10,
                r"$10 \frac{m}{s}$",
                labelpos="S",
                coordinates="figure",
            )
            plt.title("Wind Speed and Direction at Time {}".format(timestamp))
            plt.grid()
            plt.savefig("wind_{}.png".format(timestamp.isoformat()))
        elif var_type == "wave":
            if timestamp in self.time:
                H = self.data["sig_wave_height"][:][left, :, :]
                wave_dir = self.data["wave_direction"][:][left, :, :]
                T = self.data["wave_period"][:][left, :, :]
            else:
                right = left + 1
                Z_left = self.data["sig_wave_height"][:][left, :, :]
                Z_right = self.data["sig_wave_height"][:][right, :, :]
                delta = self.time[right] - self.time[left]
                H = Z_left * ((self.time[right] - timestamp) / delta) + Z_right * (
                    (timestamp - self.time[left]) / delta
                )
                wave_dir_left = self.data["wave_direction"][:][left, :, :]
                wave_dir_right = self.data["wave_direction"][:][right, :, :]
                wave_dir = wave_dir_left * (
                    (self.time[right] - timestamp) / delta
                ) + wave_dir_right * ((timestamp - self.time[left]) / delta)
                T_left = self.data["wave_period"][:][left, :, :]
                T_right = self.data["wave_period"][:][right, :, :]
                T = T_left * ((self.time[right] - timestamp) / delta) + T_right * (
                    (timestamp - self.time[left]) / delta
                )
            ax = plt.contourf(X, Y, H, zorder=0)
            cbar = plt.colorbar(ax)
            cbar.set_label(self.units["sig_wave_height"])
            wave_dir = np.radians((((450 - wave_dir) % 360) + 180) % 360)
            U = np.cos(wave_dir)
            V = np.sin(wave_dir)
            plt.quiver(
                X[::latstep, ::lonstep],
                Y[::latstep, ::lonstep],
                U[::latstep, ::lonstep],
                V[::latstep, ::lonstep],
                pivot="middle",
                zorder=2,
            )
            plt.title(
                "Significant Wave Height and Wave Direction at \nTime {}".format(
                    timestamp
                )
            )
            plt.grid(zorder=1)
            plt.savefig("wave_height_direction_{}.png".format(timestamp.isoformat()))
            fig.clear()
            ax = plt.contourf(X, Y, T)
            cbar = plt.colorbar(ax)
            cbar.set_label(self.units["wave_period"])
            plt.xlabel("Longitude (degrees east)")
            plt.ylabel("Latitude (degrees north)")
            plt.title("Wave Period at Time {}".format(timestamp))
            plt.grid()
            plt.savefig("wave_period_{}.png".format(timestamp.isoformat()))
        else:
            if timestamp in self.time:
                P = self.data["surface_pressure"][:][left, :, :]
            else:
                right = left + 1
                P_left = self.data["surface_pressure"][:][left, :, :]
                P_right = self.data["surface_pressure"][:][right, :, :]
                delta = self.time[right] - self.time[left]
                P = P_left * ((self.time[right] - timestamp) / delta) + P_right * (
                    (timestamp - self.time[left]) / delta
                )
            ax = plt.contourf(X, Y, P)
            cbar = plt.colorbar(ax)
            cbar.set_label(self.units["surface_pressure"])
            plt.grid()
            plt.title("Atmospheric Pressure at Time {}".format(timestamp))
            plt.savefig("pressure_{}.png".format(timestamp.isoformat()))
        fig.clear()


# to do:
# https://www.youtube.com/watch?v=L4F3J6KvTm4
# plot ocean surface for a single t with triangles
# need to figure out how to incorporate wave number in eq.
#   and x value
