import traci
import numpy as np
import traci
from random import randint
import math


class CustomEnv:

    def __init__(self):
        self.MAX_EPISODES = 100
        self.MAX_STEPS = 3000
        self.CAR_LENGTH = 5
        self.DISTANCE_THRESHOLD = 30
        self.INPUT_SHAPE = (9, 5)
        self.OUTPUT_SHAPE = (3)
        # 0 go left, 1 stay in place, 2 go right
        self.LEFT_ACTION = 0
        self.STAY_ACTION = 1
        self.RIGHT_ACTION = 2
        self.ACTIONS = (self.LEFT_ACTION, self.STAY_ACTION, self.RIGHT_ACTION)
        self.carEntered = False
        self.carExited = False
        self.carCollision = False
        self.isOnThreeLaneRoad = False
        self.CollisionPenalty = -1 * (self.MAX_STEPS) / 10
        self.ExitReward = (self.MAX_STEPS) / 10

    def reset(self):
        self.carEntered = False
        self.carExited = False
        self.carCollision = False
        self.isOnThreeLaneRoad = False
        currentState = np.zeros(self.INPUT_SHAPE)
        return currentState

    def getVehicleList(self, vehicleIdList):
        cars = {}
        for vehicleId in vehicleIdList:
            cars[vehicleId] = {
                "vehicle_id": vehicleId,
                "speed": traci.vehicle.getSpeed(vehicleId),
                "acceleration": traci.vehicle.getAcceleration(vehicleId),
                "road": traci.vehicle.getRoadID(vehicleId),
                "lane": traci.vehicle.getLaneIndex(vehicleId),
                "pos_x": traci.vehicle.getPosition(vehicleId)[0],
                "pos_y": traci.vehicle.getPosition(vehicleId)[1],
                "angle": traci.vehicle.getAngle(vehicleId)
            }
        return cars

    def getTargetVehicleInf(self, targetVehicleId, cars):
        if targetVehicleId in cars:
            return cars[targetVehicleId]
        return {}

    def getDistance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    def getApproximateCars(self, targetVehicleId, targetVehicleInf, cars):
        distanceThreshold = self.DISTANCE_THRESHOLD
        approximateCars = {}
        for carId in cars:
            if carId == targetVehicleId:
                continue
            carInfo = cars[carId]
            if carInfo["road"] == targetVehicleInf["road"]:
                distance = self.getDistance(targetVehicleInf["pos_x"], targetVehicleInf["pos_y"], carInfo["pos_x"],
                                            carInfo["pos_y"])
                # print(distance)
                # print(carInfo["lane"])
                if distance <= distanceThreshold:
                    carInfo["distance"] = distance
                    approximateCars[carId] = carInfo
        return approximateCars

    def convertCarsToTuples(self, approximateCarsByLane):
        listOfTuples = []
        for carId in approximateCarsByLane:
            listOfTuples.append((carId, approximateCarsByLane[carId]["distance"]))
        listOfTuples.sort(key=lambda x: x[1])
        return listOfTuples

    def getApproximateCarsByLane(self, approximateCars, lane):
        approximateCarsByLane = {}
        for carId in approximateCars:
            if approximateCars[carId]["lane"] == lane:
                approximateCarsByLane[carId] = approximateCars[carId]
        return approximateCarsByLane

    def getFrontCarByMinimumDistance(self, carsOnTargetLaneTuple, approximateCarsByLane, targetVehicleInf):
        for carTuple in carsOnTargetLaneTuple:
            if approximateCarsByLane[carTuple[0]]["pos_x"] > targetVehicleInf["pos_x"] + (self.CAR_LENGTH / 2.0):
                return approximateCarsByLane[carTuple[0]]
        return None

    def getBackCarByMinimumDistance(self, carsOnTargetLaneTuple, approximateCarsByLane, targetVehicleInf):
        for carTuple in carsOnTargetLaneTuple:
            if approximateCarsByLane[carTuple[0]]["pos_x"] < targetVehicleInf["pos_x"] - (self.CAR_LENGTH / 2.0):
                return approximateCarsByLane[carTuple[0]]
        return None

    def getBesideCarByMinimumDistance(self, carsOnTargetLaneTuple, approximateCarsByLane, targetVehicleInf):
        for carTuple in carsOnTargetLaneTuple:
            if (approximateCarsByLane[carTuple[0]]["pos_x"] <= targetVehicleInf["pos_x"] + (
                    self.CAR_LENGTH / 2.0)) and \
                    (approximateCarsByLane[carTuple[0]]["pos_x"] >= targetVehicleInf["pos_x"] - (
                            self.CAR_LENGTH / 2.0)):
                return approximateCarsByLane[carTuple[0]]
        return None

    def getArrayTable(self, targetVehicleInf, approximateCars):
        carsAroundTarget = [[None, None, None], [None, None, None], [None, None, None]]
        # Same lane cars
        targetLane = targetVehicleInf["lane"]
        carsOnTargetLane = self.getApproximateCarsByLane(approximateCars, targetLane)
        carsOnTargetLaneTuple = self.convertCarsToTuples(carsOnTargetLane)
        frontOnSameLane = self.getFrontCarByMinimumDistance(carsOnTargetLaneTuple, carsOnTargetLane, targetVehicleInf)
        backOnSameLane = self.getBackCarByMinimumDistance(carsOnTargetLaneTuple, carsOnTargetLane, targetVehicleInf)
        carsAroundTarget[2 - targetLane][1] = targetVehicleInf
        carsAroundTarget[2 - targetLane][2] = frontOnSameLane
        carsAroundTarget[2 - targetLane][0] = backOnSameLane
        # Left Lane
        if targetLane < 2:
            leftLane = targetLane + 1
            carsOnLeftLane = self.getApproximateCarsByLane(approximateCars, leftLane)
            carsOnLeftLaneTuple = self.convertCarsToTuples(carsOnLeftLane)
            # Left Front
            carFrontLeft = self.getFrontCarByMinimumDistance(carsOnLeftLaneTuple, carsOnLeftLane, targetVehicleInf)
            # Left
            carLeft = self.getBesideCarByMinimumDistance(carsOnLeftLaneTuple, carsOnLeftLane, targetVehicleInf)
            # Left Back
            carBackLeft = self.getBackCarByMinimumDistance(carsOnLeftLaneTuple, carsOnLeftLane, targetVehicleInf)
            carsAroundTarget[2 - leftLane][1] = carLeft
            carsAroundTarget[2 - leftLane][2] = carFrontLeft
            carsAroundTarget[2 - leftLane][0] = carBackLeft
        # Right Lane
        if targetLane > 0:
            rightLane = targetLane - 1
            carsOnRightLane = self.getApproximateCarsByLane(approximateCars, rightLane)
            carsOnRightLaneTuple = self.convertCarsToTuples(carsOnRightLane)
            # Right Front
            carFrontRight = self.getFrontCarByMinimumDistance(carsOnRightLaneTuple, carsOnRightLane, targetVehicleInf)
            # Right
            carRight = self.getBesideCarByMinimumDistance(carsOnRightLaneTuple, carsOnRightLane, targetVehicleInf)
            # Right Back
            carBackRight = self.getBackCarByMinimumDistance(carsOnRightLaneTuple, carsOnRightLane, targetVehicleInf)
            carsAroundTarget[2 - rightLane][1] = carRight
            carsAroundTarget[2 - rightLane][2] = carFrontRight
            carsAroundTarget[2 - rightLane][0] = carBackRight
        return carsAroundTarget

    def convertToDqnInput(self, systemState, targetVehicleId):
        # System state:
        # Array of 3 X 3 for the target and approximate cars
        # rows numbers:
        #       0           1           2
        #   -------------------------------------
        # 0 | row. 0    | row. 1    | row. 2    |
        #   -------------------------------------
        # 1 | row. 3    | row. 4    | row. 5    |
        #   -------------------------------------
        # 2 | row. 6    | row. 7    | row. 8    |
        #   -------------------------------------

        dqnInput = np.zeros((9, 5))
        # DQN input:
        # Rows:     as described in the previous array
        # Columns:  0           1           2           3           4
        #           Speed       Accel.      Dist.       carExists   targetVeh.
        # values:   double      double      double      (0 or 1)    (0 or 1)
        #                                               0: false, 1: true
        row = 0
        for i in range(3):
            for j in range(3):
                vehicleState = systemState[i][j]
                if vehicleState:
                    # -> if not None
                    dqnInput[row, 0] = vehicleState["speed"]
                    dqnInput[row, 1] = vehicleState["acceleration"]
                    if targetVehicleId == vehicleState["vehicle_id"]:
                        # -> if is target vehicle
                        dqnInput[row, 2] = 0
                        dqnInput[row, 3] = 0
                        dqnInput[row, 4] = 1
                    else:
                        # -> if is approximate vehicle
                        dqnInput[row, 2] = vehicleState["distance"]
                        dqnInput[row, 3] = 1
                        dqnInput[row, 4] = 0
                else:
                    # -> ONLY if is None
                    dqnInput[row, 0] = 0
                    dqnInput[row, 1] = 0
                    dqnInput[row, 2] = 0
                    dqnInput[row, 3] = 0
                    dqnInput[row, 4] = 0
                row += 1
        return dqnInput

    def isCollision(self, targetVehicleId):
        collisonList = traci.simulation.getCollidingVehiclesIDList()
        if len(collisonList) > 0:
            if targetVehicleId in collisonList:
                print("Target vehicle is in collision")
                return True
        return False

    def printAction(self, action):
        if action != None:
            if action == self.RIGHT_ACTION:
                print("### Choosing Right")
            elif action == self.LEFT_ACTION:
                print("### Choosing Left")
            else:
                print("### Choosing Same")

    def step(self, targetVehicleId, action):
        self.printAction(action)
        targetVehicleOutOfRoad = False
        if action != None and self.isOnThreeLaneRoad and self.carEntered and (not self.carExited):
            laneIndex = traci.vehicle.getLaneIndex(targetVehicleId)
            if laneIndex == 0 and action == self.RIGHT_ACTION:
                # if car is at rightmost and action is to go right---> penalise
                print("Target vehicle out of road")
                targetVehicleOutOfRoad = True
            elif laneIndex == 2 and action == self.LEFT_ACTION:
                # if car is at leftmost and action is to go right---> penalise
                print("Target vehicle out of road")
                targetVehicleOutOfRoad = True
            else:
                chosenLane = laneIndex
                if action == self.RIGHT_ACTION:
                    chosenLane = laneIndex - 1
                    print("### Going Right")
                elif action == self.LEFT_ACTION:
                    chosenLane = laneIndex + 1
                    print("### Going Left")
                else:
                    print("### Stay in place")
                print("Veh from lane", laneIndex, "to lane", chosenLane)
                traci.vehicle.changeLane(targetVehicleId, chosenLane, 0)

        if not targetVehicleOutOfRoad:
            traci.simulationStep()

        # If (our target vehicle is out of road) or (there was a collision due to Sumo simulation step)
        if targetVehicleOutOfRoad or self.isCollision(targetVehicleId):
            self.carCollision = True

        if self.carCollision:
            collisionreason= "accident"
            if targetVehicleOutOfRoad:
                collisionreason = "out_of_road"

            # When there is a collision or car out of road
            #   state -> array of -1
            #   penalty -> collision penalty
            #   episode done -> true
            return (np.ones(self.INPUT_SHAPE) * -1, self.CollisionPenalty, True, collisionreason)
        else:
            vehicleIdList = traci.vehicle.getIDList()
            if targetVehicleId not in vehicleIdList:
                # If the car's info is empty (means the car is not in the simulation)
                if self.carEntered and not self.carExited:
                    # If the car has already entered and it is not flagged as "exited"
                    #   then it should be flagged as Exited and the episode should terminate
                    print("the vehicle has exited")
                    self.carExited = True
                    # When the car arrives to distination
                    #   state -> array of 1
                    #   reward -> Exit reward
                    #   episode done -> true
                    return (np.ones(self.INPUT_SHAPE), self.ExitReward, True, "exited")
            else:
                cars = self.getVehicleList(vehicleIdList)
                targetVehicleInf = self.getTargetVehicleInf(targetVehicleId, cars)
                if not self.carEntered:
                    print("Target vehicle has entered")
                    # input("Press Enter to continue...")
                    self.carEntered = True
                    traci.vehicle.setColor(targetVehicleId, (255, 0, 0))
                    traci.vehicle.setMinGapLat(targetVehicleId, 0)
                    traci.vehicle.setMaxSpeed(targetVehicleId, randint(13,30))
                    # sumo only controls speed change
                    traci.vehicle.setLaneChangeMode(targetVehicleId, 0b000000010000)
                roadId = traci.vehicle.getRoadID(targetVehicleId)
                numberOfLanes = traci.edge.getLaneNumber(roadId)

                if numberOfLanes > 1:
                    self.isOnThreeLaneRoad = True
                    approximateCars = self.getApproximateCars(targetVehicleId, targetVehicleInf, cars)
                    systemState = self.getArrayTable(targetVehicleInf, approximateCars)
                    # print(systemState)
                    dqnInput = self.convertToDqnInput(systemState, targetVehicleId)
                    # When the car is on threelane road
                    #   state -> actual state
                    #   reward -> acceleration * 10
                    #   episode done -> false
                    return (dqnInput, targetVehicleInf["acceleration"] * 10, False, "")
                else:
                    # If vehicle is not on a three lane road, neither its state nor its reward are not
                    #   taken into account
                    return (np.zeros(self.INPUT_SHAPE), 0, False, "")
