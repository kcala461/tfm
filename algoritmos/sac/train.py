import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")
# Parallel environments
from stable_baselines3.common.callbacks import BaseCallback


class TimestepPrinterCallback(BaseCallback):
    def __init__(self, model_save_path, save_freq = 1, print_freq=1000, verbose=0):
        super(TimestepPrinterCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.model_save_path = model_save_path
 

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print('------------------------------------')
            print(f"Timestep: {self.num_timesteps}")
            print('------------------------------------')

        if self.num_timesteps % self.save_freq == 0:
            # Guardar el modelo
            model_save_path = f"{self.model_save_path}.zip"
            self.model.save(model_save_path)
            print(f"Modelo guardado en: {model_save_path}")
       
            



def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name=r"..\..\mapas\recto", 
        max_steps=5000,  #cambiarlo
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
    )

# Crear un entorno vectorizado para el entrenamiento
vec_env = make_vec_env(make_duckietown_env, n_envs=1, seed=123)

# Callback para guardar acciones, recompensas y el modelo
model_path = "model/ModeloLineaRecta"
save_freq = 100000  # Frecuencia para guardar el modelo


timestep_printer = TimestepPrinterCallback(print_freq=50000, save_freq=save_freq, model_save_path=model_path)

# Instanciar el agente SAC
model = SAC("MlpPolicy", vec_env, verbose=1, device='cuda', batch_size=1024)


model.learn(total_timesteps=4000000, callback=timestep_printer, log_interval=100000)
model.save(r"model/ModeloLineaRecta")