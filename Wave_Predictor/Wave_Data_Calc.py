from datetime import datetime, timedelta
import numpy as np
import torch
import os

from .WaveHeightPredictor import WaveHeightPredictor
from Ocean.weather_data_interpolation_API import (
    planning_interpolation,
    load_netcdf_folder,
)


class Wave_Data_Calc:
    """
    Finds Wave Height, Heading, and Period using Machine Learning or Forecasting
    """

    def __init__(self, weather_input_dir, fc_input_dir=None, model_path=None):
        """
        Wave_Data_Calc Constructor
        --------------------------------------------------------------------
        Parameters
        --------------------------------------------------------------------
        weather_input_dir: string,folder that contains nowcasts
        fc_input_dir: string,folder that contains forecasts
        model_path: string, machine learning model path
        """
        self.model = WaveHeightPredictor()
        self.weather_input_dir = weather_input_dir
        self.fc_input_dir = fc_input_dir

        self.use_model = model_path != None
        self.use_fc = fc_input_dir != None
        if self.use_model:
            self.model.load_state_dict(torch.load(model_path))
            self.stats = torch.load("Wave_Predictor/torch_data/normalization.pt")
            self.train_mean = self.stats["train_mean"]
            self.train_std = self.stats["train_std"]
        self.model_cache = {}
        self.forecast_cache = {}
        self.model.eval()
        # Load the NetCDF files into memory

        self.weather_data, issue_dts, fcst_dts = load_netcdf_folder(
            self.weather_input_dir
        )

        if self.use_fc:
            self.forecast_data, issue_dts, fcst_dts = load_netcdf_folder(
                self.fc_input_dir
            )

    def calc_WaveData(self, waypoint, current_time, future_time):
        """
        Finds Wave Height, Heading, and Period using Nowcasts, Machine Learning, or Forecasting
        ---------------------------------------------------------------------------------------

        Parameters
        ---------------------------------------------------------------------------------------
        :param waypoint: Waypoint Object, waypoint to get wave data at
        :param current_time: datetime object, current time of simulation
        :param future_time: datetime object, time to get wave data at

        Returns
        ---------------------------------------------------------------------------------------
        tuple: (wave_height, wave_heading, wave_period)
        """
        result = ()
        time_amount = (future_time - current_time).total_seconds() / 3600
        if self.use_fc and time_amount >= 0:
            key = (
                round(waypoint.lat, 1),
                round(waypoint.long, 1),
                self.round_datetime_to_nearest_hour(current_time),
                self.round_datetime_to_nearest_hour(future_time),
            )
            if key in self.forecast_cache:
                result = self.forecast_cache[key]
                return result
            result = self.get_forecast_data(
                waypoint.lat, waypoint.long, current_time, time_amount
            )
            self.forecast_cache[key] = result

        elif self.use_model and time_amount >= 0:

            key = (
                round(waypoint.lat, 1),
                round(waypoint.long, 1),
                self.round_datetime_to_nearest_hour(current_time),
                self.round_datetime_to_nearest_hour(future_time),
            )
            if key in self.model_cache:
                result = self.model_cache[key]
                return result
            model_data_input = self.extract_model_features(
                waypoint.lat, waypoint.long, current_time, time_amount
            )
            result = self.predict_with_model(model_data_input)
            self.model_cache[key] = result

        else:
            results = planning_interpolation(
                ds=self.weather_data,
                current_dt=future_time,
                fcst_dt=future_time,
                target_lat=waypoint.lat,
                target_lon=waypoint.long,
            )
            result = (
                results.get("swh")[0],
                results.get("dirpw")[0],
                results.get("perpw")[0],
            )

        return result

    def extract_model_features(self, lat, lon, current_time, time_amount):
        """
        extracts  machine learning model features
        ---------------------------------------------------------------------------------------

        Parameters
        ---------------------------------------------------------------------------------------
        :param lat: float, latitude to predict wave data at
        :param lon: float, longitude to predict wave data at
        :param current_time: datetime object, current time of simulation
        :param time amount: float, forecast horizon from current time

        Returns
        ---------------------------------------------------------------------------------------
        tensor: ml model input
        """

        def interp(dt):
            """
            gets ml model input for one date
            ---------------------------------

            Parameters
            ---------------------------------
            :param dt: datetime object

            Returns
            ---------------------------------
            dictionary: wave height, wave period, wave direction, wind direction, wind speed
            """
            results = planning_interpolation(
                ds=self.weather_data,
                current_dt=dt,
                fcst_dt=dt,
                target_lat=lat,
                target_lon=lon,
            )
            return {
                "swh": results.get("swh")[0],
                "perpw": results.get("perpw")[0],
                "dirpw": results.get("dirpw")[0],
                "ws": results.get("ws")[0],
                "wdir": results.get("wdir")[0],
            }

        # Query interpolated forecast data at t, t-6h, t-12h, t-18h
        t0 = current_time
        t6 = current_time - timedelta(hours=6)
        t12 = current_time - timedelta(hours=12)
        t18 = current_time - timedelta(hours=18)

        d0 = interp(t0)
        d6 = interp(t6)
        d12 = interp(t12)
        d18 = interp(t18)

        model_data_input = torch.tensor(
            np.stack(
                [
                    d0["swh"],
                    d18["swh"],
                    d12["swh"],
                    d6["swh"],
                    d0["perpw"],
                    d18["perpw"],
                    d12["perpw"],
                    d6["perpw"],
                    d0["dirpw"],
                    d18["dirpw"],
                    d12["dirpw"],
                    d6["dirpw"],
                    d0["wdir"],
                    d18["wdir"],
                    d12["wdir"],
                    d6["wdir"],
                    d0["ws"],
                    d18["ws"],
                    d12["ws"],
                    d6["ws"],
                    time_amount,
                ]
            ),
            dtype=torch.float32,
        )

        return model_data_input

    def predict_with_model(self, features):
        """
        Predicts Wave Height, Heading, and Period using Machine Learning Model
        ---------------------------------------------------------------------------------------
        Parameters
        ---------------------------------------------------------------------------------------
        :param features: tensor, model features

        Returns
        ---------------------------------------------------------------------------------------
        tuple: (wave_height, wave_heading, wave_period)
        """
        features = (features - self.train_mean) / self.train_std
        with torch.no_grad():
            output = self.model(features.unsqueeze(0)).view(-1)

        return (
            output[0].item(),
            ((torch.rad2deg(torch.atan2(output[1], output[2])) + 360) % 360).item(),
            output[3].item(),
        )

    def get_forecast_data(self, lat, lon, current_time, time_amount):
        """
        Predicts Wave Height, Heading, and Period using forecast data
        ---------------------------------------------------------------------------------------
        Parameters
        ---------------------------------------------------------------------------------------
        :param lat: float, latitude to predict wave data at
        :param lon: float, longitude to predict wave data at
        :param current_time: datetime object, current time of simulation
        :param time amount: float, forecast horizon from current time

        Returns
        ---------------------------------------------------------------------------------------
        tuple: (wave_height, wave_heading, wave_period)
        """
        # Compute target forecast datetime
        fcst_dt = current_time + timedelta(hours=time_amount)

        results = planning_interpolation(
            ds=self.forecast_data,
            current_dt=current_time,
            fcst_dt=fcst_dt,
            target_lat=lat,
            target_lon=lon,
        )

        wave_height = results.get("swh")[0]
        wave_direction = results.get("dirpw")[0]
        wave_period = results.get("perpw")[0]

        return wave_height, wave_direction, wave_period

    def round_datetime_to_nearest_hour(self, dt):
        """
        Rounds a datetime object to the nearest hour
        --------------------------------------------

        Parameters
        ---------------------------------------------
        :param dt: datetime object

        Returns
        ---------------------------------------------
        Rounded Datetime Object
        """
        discard = timedelta(
            minutes=dt.minute, seconds=dt.second, microseconds=dt.microsecond
        )
        dt -= discard
        if discard >= timedelta(minutes=30):
            dt += timedelta(hours=1)
        return dt
