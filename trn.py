import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import gym_duckietown
from gym_duckietown.simulator import Simulator

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Parallel environments
from stable_baselines3.common.callbacks import BaseCallback
save_path = "acciones_recompensasObtacles.txt"
class TimestepPrinterCallback(BaseCallback):
    def __init__(self, save_path, print_freq=1000, verbose=0):
        super(TimestepPrinterCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.save_path = save_path
        self.actions = []
        self.rewards = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print('------------------------------------')
            print(f"Timestep: {self.num_timesteps}")
            print('------------------------------------')
        self.actions.append(self.locals['actions'])
        self.rewards.append(self.locals['rewards'])
       
            
    def _on_training_end(self) -> None:
        with open(self.save_path, 'w') as f:
            for i in range(len(self.actions)):
                f.write(f"Timestep {i+1}:\n")
                f.write(f"Actions: {self.actions[i]}\n")
                f.write(f"Rewards: {self.rewards[i]}\n\n")     
    
        return True

    
    
def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name="mapas/recto", 
        max_steps=15000,  #cambiarlo
        domain_rand=1,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,  
        
        full_transparency=False,
        distortion=True,
        style='synthetic',
        draw_curve=True,
        start_pose=[[0.35, 0, 0.35], 0],
        user_tile_start=[0,1],
        goal=[9,1],
        

        # start_pose=[1.75, 0, 1.75],
        # full_transparency=True
    )

# Crear un entorno vectorizado para el entrenamiento
vec_env = make_vec_env(make_duckietown_env, n_envs=4, seed=123)

# vec_env = make_vec_env("CartPole-v1", n_envs=4)

timestep_printer = TimestepPrinterCallback(print_freq=10000, save_path=save_path)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", n_steps=4096, batch_size=1024)
# model.load("final/ModeloLineaRecta.zip")

model.learn(total_timesteps=4500000, callback=timestep_printer, log_interval=100000)
model.save("final/ModeloLineaRecta")

del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     # vec_env.render()
#     # print(rewards)