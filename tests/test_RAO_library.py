from datetime import datetime
import numpy as np
import pytest
from src.Ocean.basespectra import Bretschneider
from src.Simulation.vessel_sim_engine import Waypoint, Mission, Destination, vesselFuel
from src.Simulation.machinery_fuel import (
    propulsionSimulationBase,
    PropellerPropulsionModel,
    RPM_EngineModel,
    RPM_Power_EngineModel,
    NPL_ResistanceModel,
)
from src.Ocean.RAO_library import RAO_Library
from pathlib import Path

script_dir = Path(__file__).resolve().parent
ocean_path = script_dir.parent / "src/Ocean" / "NOAA_NorthAtlantic.nc"


@pytest.fixture
def rao_lib():
    rao_lib = RAO_Library()
    return rao_lib


def test_integration(rao_lib):

    rao_correct = [
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.01,
        1.03,
        1.07,
        1.20,
        1.22,
        0.80,
        0.50,
        0.24,
        0.09,
        0.03,
        0.00,
        0.00,
    ]  # heave/amplitude

    spectrum_correct = [
        0,
        0,
        0,
        0,
        0.2,
        2.04,
        4.27,
        4.98,
        4.5,
        3.62,
        2.75,
        2.06,
        1.53,
        1.14,
        0.86,
        0.65,
        0.5,
    ]

    response_spectra = [
        0,
        0,
        0,
        0,
        0.2,
        2.081,
        4.53,
        5.702,
        6.48,
        5.388,
        1.760,
        0.515,
        0.088,
        0.009,
        0.0008,
        0,
        0,
    ]
    freq_correct = [
        0.107,
        0.166,
        0.229,
        0.295,
        0.365,
        0.439,
        0.517,
        0.598,
        0.682,
        0.770,
        0.862,
        0.958,
        1.057,
        1.16,
        1.267,
        1.367,
        1.491,
    ]

    integration = np.sqrt(np.trapezoid(response_spectra, x=freq_correct))
    assert integration == pytest.approx(1.3, 0.5)


# test_integration(RAO_Library())
