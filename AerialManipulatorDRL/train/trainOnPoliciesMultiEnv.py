from stable_baselines.common.policies import MlpPolicy, register_policy, FeedForwardPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines import  PPO2,TRPO,A2C
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
import gym
import AerialManipulatorDRL
import numpy as np
import datetime
import os
import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback,CheckpointCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "" # not use gpu since performance becomes worse


policy_kwargs =dict(net_arch =[dict(pi=[128,128],vf=[128,128])])
log_dir = "../logs/"
n_env =4 
SelectPolicy =1


def make_env(env_id,logdir,i):

    def _init():
        env = gym.make(env_id)
        """
        #change the environment region and other parameters if needed
        env.init_dist_pos_low_x = -1
        env.init_dist_pos_high_x = 1 # 0.5
        env.init_dist_pos_low_y =  0.2 #-0.5 
        env.init_dist_pos_high_y = 1# 0.5 #0.5 # 0.5
        env.step_per_episode=400
        env.outOfBorderPos=4
        """
        if (i==1):
            #monitor only one environment
            env = Monitor(env, filename=logdir, allow_early_resets=True,info_keywords = ('finstate',))
        return env
    return _init


checkCallBack = CheckpointCallback(save_freq=200,save_path=log_dir,name_prefix='rl_model')

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-20:])
              #print(x[-1], 'timesteps')
              #print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
              #if self.verbose > 0:
                #print("Num timesteps: {}".format(self.num_timesteps))
                #print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path + 'best_model' + str(x[-1]) + 'with_mean_rew' + str(mean_reward))

        return True

def main():
    env = DummyVecEnv([make_env("quad-withArm_1Dof-v0",log_dir,i) for i in range(n_env)])
    callback = SaveOnBestTrainingRewardCallback(check_freq=200, log_dir=log_dir)

    if SelectPolicy == 0:
      model = A2C(MlpPolicy, env,policy_kwargs=policy_kwargs)
    else:
      model = PPO2(MlpPolicy, env,policy_kwargs=policy_kwargs,verbose=1) #used for 1 dof training 128 and 128 2 layer not merged mlp
    
    
    model.learn(total_timesteps=int(9e6),callback=callback,)



if __name__ == '__main__':
    main()
