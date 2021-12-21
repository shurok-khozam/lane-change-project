import math
import os
import sys
from random import randint

import numpy as np
import traci
import matplotlib.pyplot as plt

import custom_env
from dqn import DQN

# for each class that will be treated as main ,
# for importing environment variables in python
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def train(sumoBinary):
    env = custom_env.CustomEnv()
    episodes = env.MAX_EPISODES
    steps = env.MAX_STEPS
    dqn_agent = DQN(env=env)


    TargetVehicleStatistics = np.zeros((math.floor(episodes/10), 3))

    for episode in range(episodes):
        print("################ Starting episode", episode, "################")
        traci.start([sumoBinary, "-c", "./data/lane_change_3.sumocfg"])
        episodeIndex = math.floor(episode/10)

        cur_state = env.reset()
        targetVehicleId = "vehicle_" + str(randint(501, 1000))
        print("training vehicle: " + targetVehicleId)

        # randomizing lane's max speed
        laneIdList = traci.lane.getIDList()
        for laneId in laneIdList:
            traci.lane.setMaxSpeed(laneId, randint(13, 30))

        done_reason = ""
        # Episode's steps execution
        for step in range(steps):
            if env.carEntered and env.isOnThreeLaneRoad:
                action = dqn_agent.action(cur_state)
                new_state, reward, done, done_reason = env.step(targetVehicleId,action)
                dqn_agent.remember(cur_state, action, reward, new_state, done)

                dqn_agent.replay()  # internally iterates default (prediction) model
                dqn_agent.target_train()  # iterates target model

                cur_state = new_state
                if done:
                    break
            else:
                env.step(targetVehicleId,None)

        if done_reason == "exited":
            TargetVehicleStatistics[episodeIndex, 0] += 1
        elif done_reason == "accident":
            TargetVehicleStatistics[episodeIndex, 1] += 1
        elif done_reason == "out_of_road":
            TargetVehicleStatistics[episodeIndex, 2] += 1
        # after each episode save model
        dqn_agent.save_model("success.model")
        traci.close()
        print("################ Ending episode", episode, "################")
    print("################# Training simulation has ended #################")
    x_axis = list(range(0, math.floor(episodes / 10)))
    plt.plot(x_axis, TargetVehicleStatistics[:, 0], label="Exited")
    plt.plot(x_axis, TargetVehicleStatistics[:, 1], label="Accident")
    plt.plot(x_axis, TargetVehicleStatistics[:, 2], label="Out of road")
    plt.xlabel('Episodes / 10')
    plt.ylabel('Events')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    runGUI = False
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    train(sumoBinary)


