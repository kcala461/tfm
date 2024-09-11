

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


import gym_duckietown
from gym_duckietown.simulator import Simulator

import os
import torch
import matplotlib.pyplot as plt
print('aaaaa', max(-5000, -1.5))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Parallel environments

def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name="straight_road",
        max_steps=10000,  #cambiarlo
        domain_rand=1,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,  
        
        full_transparency=False,
        distortion=True,
        style='synthetic',
        draw_curve=True,
        start_pose=[[0.35, 0, 0.35], 0],
        user_tile_start=[0,0],
        goal=[10,0],
        

        # start_pose=[1.75, 0, 1.75],
        # full_transparency=True
    )

# Crear un entorno vectorizado para el entrenamiento
vec_env = make_vec_env(make_duckietown_env, n_envs=1, seed=123)

model = PPO.load("ModeloNuevoAngQuee")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
    # print(rewards)