import os
import sys
from random import random, randint

import traci

import custom_env
from dqn import DQN

# for each class that will be treated as main ,
# for importing environment variables in python
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def play(sumoBinary):
    env = custom_env.CustomEnv()
    steps = env.MAX_STEPS
    #this function use dqn class for choosing action to take
    dqn_agent = DQN(env=env)
    dqn_agent.load_model("success.model")
    dqn_agent.epsilon = 0
    dqn_agent.epsilon_min = 0

    traci.start([sumoBinary, "-c", "./data/lane_change_3.sumocfg"])
    cur_state = env.reset()

    targetVehicleId = "vehicle_" + str(randint(501, 1000))
    print("training vehicle: " + targetVehicleId)

    for step in range(steps):
        if env.carEntered and env.isOnThreeLaneRoad and not env.carExited:
            action = dqn_agent.action(cur_state)
            new_state, reward, done, _ = env.step(targetVehicleId,action)
            cur_state = new_state
        else:
            env.step(targetVehicleId,None)

    traci.close()
    print("play simulation has ended")

if __name__ == "__main__":
    runGUI = True
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    play(sumoBinary)


