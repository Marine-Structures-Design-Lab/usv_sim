'''
test cases for machinery and fuel code
'''
from datetime import datetime
import pytest
from Simulation.vessel_sim_engine import  Waypoint, Mission, Destination, vesselFuel
from Simulation.machinery_fuel import propulsionSimulationBase,PropellerPropulsionModel,RPM_EngineModel,RPM_Power_EngineModel,NPL_ResistanceModel
from pathlib import Path
script_dir = Path(__file__).resolve().parent
ocean_path=script_dir.parent / 'Ocean/weather_2_nc/1205_1215_lat-10_5_lon50_60_fc350_ts9'
npl_path=script_dir.parent / 'Simulation' / 'Molland_NPL_Data_NumpyRead.csv'
model_path="Wave_Predictor/torch_data/model.pth"
@pytest.fixture
def fuel_model(): 
        propeller=PropellerPropulsionModel(3.40,1.39,0.67,4)
        res_model = NPL_ResistanceModel(1025.0, 1.225, 1.1395e-6, 0.0, npl_path)
        RPM_vals = [800, 1000, 1300, 1600, 1800, 2000, 2100, 2300]
        power_vals = [[46], [89], [195], [364], [518], [711], [823], [1081]]
        fuel_vals = [[256.8], [230.8], [217.3], [212.1], [208.9], [208.3], [212.2], [222.4]]
        engine = RPM_EngineModel( RPM_vals,power_vals, fuel_vals, 10)
        propSim1=propulsionSimulationBase(res_model,propeller,engine)
        return propSim1,res_model,propeller,engine
@pytest.fixture
def fuel_vessel():
    destinations=[Destination(Waypoint(3, 55), datetime(2024, 12, 8, 2, 0), 9), 
                    Destination(Waypoint(4, 55), datetime(2024, 12, 8, 17, 0), 7),
                     Destination(Waypoint(2.5, 54.5), datetime(2024,12, 9, 6, 0), 9)]    
    mission = Mission(destinations)
    vessel = vesselFuel("fuel_vessel","fuel_sim",Destination(Waypoint(3, 54.5),datetime(2024, 12, 7, 20, 0),0),Destination(Waypoint(3, 54.5),datetime(2024, 12, 9, 7, 0),0),mission,5,ocean_path,model_path=model_path,speed=10)
    return vessel 
def test_engine(fuel_vessel):
    RPM1 = [800, 1000, 1300, 1600, 1800, 2000, 2100, 2300]
    RPM2 = [800, 1000, 1200, 1400, 1600, 1800]
    power1 = [[46], [89], [195], [364], [518], [711], [823], [1081]]
    power2 = [[43, 49, 56, 66], [84, 96, 109, 128], [146, 166, 188, 221], [231, 263, 298, 351], [346, 393, 445, 524], [492, 559, 634, 746]]
    fuel1 = [[256.8], [230.8], [217.3], [212.1], [208.9], [208.3], [212.2], [222.4]]
    fuel2 = [[249.5, 241.4, 245.3, 222.4], [226.2, 224.0, 226.5, 211.5], [222.1, 214.2, 216.2, 208.5], [212.0, 207.5, 209.0, 208.9], [207.6, 205.8, 206.9, 203.9], [204.4, 201.8, 201.5, 201.0]]
    
    engine1 = RPM_EngineModel(RPM1, power1, fuel1, 10)
    print(f"RPM Engine Model, RPM=100: {engine1.getSFC(100, 90, fuel_vessel)}")
    assert engine1.getSFC(100, 90, fuel_vessel)==pytest.approx(230.8,0.001)
    print(f"RPM Engine Model, RPM=170: {engine1.getSFC(170, 90, fuel_vessel)}")
    assert engine1.getSFC(170, 90, fuel_vessel)==pytest.approx(210.51299,0.001)
    print(f"RPM Engine Model RPM=300(too large) {engine1.getSFC(300, 90, fuel_vessel)} ")
    assert engine1.getSFC(300, 90, fuel_vessel)==pytest.approx(222.4,0.001)
    print(f"RPM Engine Model, RPM=100(power too large): {engine1.getSFC(100, 92, fuel_vessel)}")
    assert engine1.getSFC(100, 92, fuel_vessel)==pytest.approx(230.8,0.001)
    engine2 = RPM_Power_EngineModel(RPM2, power2, fuel2, 10)
    print(f"RPM Power Engine Model, RPM=100,Power 96 kW: {engine2.getSFC(100, 96, fuel_vessel)}")
    assert engine2.getSFC(100, 96, fuel_vessel)==pytest.approx(224.5273,0.001)
    print(f"RPM Power Engine Model, RPM=170, Power 500 kW: {engine2.getSFC(170, 500, fuel_vessel)}") 
    assert engine2.getSFC(170, 500, fuel_vessel)==pytest.approx( 203.8945,0.001)
    print(f"RPM Power Engine Model, RPM=170, Power 700(too big)  kW: {engine2.getSFC(170, 700, fuel_vessel)}") 
    assert engine2.getSFC(170, 700, fuel_vessel )==pytest.approx(201.1986,0.001)                
    print(f"RPM Power Engine Model, RPM=7(too small), Power 50 kW: {engine2.getSFC(7, 50, fuel_vessel)}")
    assert engine2.getSFC(7, 50, fuel_vessel)==pytest.approx(242.8202,0.001)
    print(f"RPM Power Engine Model, RPM=7(too small), Power 500kW(too big) kW: {engine2.getSFC(7, 5000, fuel_vessel)}")
    assert engine2.getSFC(7, 5000, fuel_vessel)==pytest.approx(222.3999,0.001)
def test_propeller(fuel_vessel):
    res_model = NPL_ResistanceModel(1025.0, 1.225, 1.1395e-6, 0.0, npl_path)
    Rt, wt, t=res_model.estimateResistance(25.0, fuel_vessel)
    propeller=PropellerPropulsionModel(3.40,1.39,0.67,4)
    eta,delivered_power,RPM=propeller.estimatePropulsion(25.0,Rt,wt,t,fuel_vessel,res_model.rho)
    print(f"open water efficiency at 25 kts: {eta}")
    assert eta==pytest.approx(0.75612,0.001)
    print(f"delivered power at 25 kts: {delivered_power}")
    assert delivered_power==pytest.approx(919.95707,0.001) 
    print(f"RPM at 25 kts: {RPM}")
    assert RPM==pytest.approx(227.71765,0.001)
    eta,delivered_power,RPM=propeller.estimatePropulsion(15.0,Rt,wt,t,fuel_vessel,res_model.rho)
    print(f"open water efficiency at 15 kts: {eta}")
    assert eta==pytest.approx(0.71549,0.001)
    print(f"delivered power at 15 kts: {delivered_power}")
    assert delivered_power==pytest.approx(583.3168,0.001)
    print(f"RPM at 15 kts: {RPM}")
    assert RPM==pytest.approx(161.9026,0.001)
    eta,delivered_power,RPM=propeller.estimatePropulsion(10.0,Rt,wt,t,fuel_vessel,res_model.rho)
    print(f"open water efficiency at 10 kts: {eta}")
    assert eta==pytest.approx(0.6147,0.001)
    print(f"delivered power at 10 kts: {delivered_power}")
    assert delivered_power==pytest.approx(452.6137,0.001)
    print(f"RPM at 10 kts: {RPM}")
    assert RPM==pytest.approx(133.1168,0.001)
def test_resistance(fuel_vessel):
     #Create a resistance model using the NPL Series
    res_model = NPL_ResistanceModel(1025.0, 1.225, 1.1395e-6, 0.0, npl_path)
    
    #Get three drag values
    drag_25kts = res_model.estimateResistance(25.0, fuel_vessel)
    drag_20kts = res_model.estimateResistance(20.0, fuel_vessel)
    drag_10kts = res_model.estimateResistance(10.0, fuel_vessel)

    print ("Drag at 25 kts: " + str(drag_25kts))
    assert drag_25kts==(pytest.approx(45.1658,0.001), pytest.approx(0.10375,0.001), pytest.approx(0.1649,0.001))
    print ("Drag at 20 kts: " + str(drag_20kts))
    assert drag_20kts==(pytest.approx(29.173497,0.001), pytest.approx(0.10375,0.001), pytest.approx(0.1649,0.001))
    print ("Drag at 10 kts: " + str(drag_10kts))
    assert drag_10kts==(pytest.approx(7.52097,0.001), pytest.approx(0.10375,0.001), pytest.approx(0.1649,0.001))
destinations=[Destination(Waypoint(3, 55), datetime(2024, 12, 8, 2, 0), 9), 
                    Destination(Waypoint(4, 55), datetime(2024, 12, 8, 17, 0), 7),
                     Destination(Waypoint(2.5, 54.5), datetime(2024,12, 9, 6, 0), 9)]    
mission = Mission(destinations)
vessel = vesselFuel("fuel_vessel","fuel_sim",Destination(Waypoint(3, 54.5),datetime(2024, 12, 7, 20, 0),0),Destination(Waypoint(3, 54.5),datetime(2024, 12, 9, 7, 0),0),mission,5,ocean_path,model_path=model_path,speed=10)
test_engine(vessel)
