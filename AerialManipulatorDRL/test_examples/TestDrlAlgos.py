from stable_baselines import TD3,DDPG,TRPO,SAC,PPO2,A2C
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import AerialManipulatorDRL
import numpy as np
import matplotlib.pyplot as plt
import os
from gym_foo.controllers.pid_quad import PDpolicy
import Box2D as b2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "" # not use gpu since performance becomes worse
env = gym.make("quad-withArm_1Dof-v0")


#set and initial position to test the algorithm if not start the test at random pos
env.set_init_pos = [-0.68, 0.67]



forceMax=env.totalForce
env = DummyVecEnv([lambda: env])



#model = A2C.load("../benchmarkResults/A2C/best_modelbest_model7117969with_mean_rew1359.93819355")


#model = TRPO.load("../benchmarkResults/TRPO/best_modelbest_model7960149with_mean_rew1439.38035615")

#model = PPO2.load("../benchmarkResults/PPO/best_model1454735with_mean_rew911.7128571.pkl")


#model = DDPG.load("../benchmarkResults/DDPG/best_modelbest_model2061085with_mean_rew1158.69359315")


model = TD3.load("../benchmarkResults/TD3/best_modelbest_model2080313with_mean_rew1251.64180435")



#model = SAC.load("../benchmarkResults/SAC/best_modelbest_model2750875with_mean_rew1312.1644071")



# Enjoy trained agent
obs = env.reset()


obs_list = [obs]
act_list = [np.array([[0, 0,0,0]])]
actPD_list = [np.array([[0, 0,0,0]])]
sum_of_rew = 0
counter=0


counterLenght = 0
time_list=[]



for i in range(3002):
    start = time.time()
    action, _states = model.predict(obs, deterministic=True)
    end = time.time()
    duration = end-start
    time_list.append(duration)
    counterLenght =counterLenght+1
    obs, rewards, dones, info = env.step(action)

    sum_of_rew += rewards
    obs_list.append(obs)
    act_list.append(action)
    #actPD_list.append(actionPD)
    counter=counter+1
    env.render()
    if dones:
        print(info[0]["finstate"])
        obs = env.reset()
        print("RESET")
        break
print("Reward collected is:", sum_of_rew)
meanTime = np.mean(time_list)
print("Average Time for one step:", meanTime )
print(" episode lenght", counterLenght)

    


