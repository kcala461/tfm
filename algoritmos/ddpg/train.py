import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import os
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")



class TimestepPrinterCallback(BaseCallback):
    def __init__(self, save_path, model_save_path, save_freq = 1, print_freq=1000, verbose=0):
        super(TimestepPrinterCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.save_path = save_path
        self.save_freq = save_freq
        self.model_save_path = model_save_path
        self.actions = []
        self.rewards = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print('------------------------------------')
            print(f"Timestep: {self.num_timesteps}")
            print('------------------------------------')
        # self.actions.append(self.locals['actions'])
        # self.rewards.append(self.locals['rewards'])
        
        if self.num_timesteps % self.save_freq == 0:
            # Guardar el modelo
            model_save_path = f"{self.model_save_path}.zip"
            self.model.save(model_save_path)
            print(f"Modelo guardado en: {model_save_path}")
       
            
    # def _on_training_end(self) -> None:
    #     with open(self.save_path, 'w') as f:
    #         for i in range(len(self.actions)):
    #             f.write(f"Timestep {i+1}:\n")
    #             f.write(f"Actions: {self.actions[i]}\n")
    #             f.write(f"Rewards: {self.rewards[i]}\n\n")     
    
        # return True

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
# print(42)
# 42 es la semilla que sirve pero estoy haciendo pruebas con 123
# Configuración de la exploración mediante ruido
n_actions = vec_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


# Callback para guardar acciones, recompensas y el modelo
save_path = "model/actions_rewardsRecto.txt"
model_path = "model/ModeloLineaRecta"
save_freq = 100000  # Frecuencia para guardar el modelo

timestep_printer = TimestepPrinterCallback(print_freq=50000, save_path=save_path, save_freq=save_freq, model_save_path=model_path)



model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, verbose=1, device='cuda', batch_size=1024)

model.learn(total_timesteps=2500000, callback=timestep_printer, log_interval=100000)
model.save(model_path)
