##traci sumo with python
import os, sys

import numpy as np
import traci
from random import randint
import math
import traci.constants as tc
import custom_env

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def executeSimulation(targetVehicleId):
    maxSteps = custom_env.MAX_STEPS
    print("Monitoring vehicle: " + targetVehicleId)
    carEntered = False
    carExited = False
    changingPeriod = 0
    carCollisoned = False
    for step in range(maxSteps):
        # print("step", step)
        traci.simulationStep()
        collisonList = traci.simulation.getCollidingVehiclesIDList()
        if len(collisonList) > 0:
            if targetVehicleId in collisonList:
                print("Vehicle has entered in collision with another car in step", step - 1, "!")
                carCollisoned = True
                break



if __name__ == "__main__":
    runGUI = True
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    traci.start([sumoBinary, "-c", "./data/lane_change_3.sumocfg"])
    maxCars = 1000
    targetVehicleId = "vehicle_" + str(randint(501, 1000))
    #targetVehicleId = "vehicle_995"
    executeSimulation(targetVehicleId)
    traci.close()














































# vehID = "vehicle_371"
# traci.start(["sumo", "-c", "./data/lane_change_3.sumocfg"])
# traci.vehicle.subscribe(vehID, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION))
# print(traci.vehicle.getSubscriptionResults(vehID))
# for step in range(300):
#    print("step", step)
#    traci.simulationStep()
#    if traci.vehicle.getSubscriptionResults(vehID):
#        print(traci.vehicle.getLaneID(vehID))
#        #traci.vehicle.changeLane(vehID, 2, 3)
# traci.close()
