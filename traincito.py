import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym_duckietown
from gym_duckietown.simulator import Simulator
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Definir parámetros personalizados para PPO
ppo_params = {
    'batch_size': 1024,  # Tamaño del lote
    'n_steps': 4096,     # Múltiplo de 1024
}

# Callback para guardar acciones y recompensas en un archivo .txt
class SaveActionsRewardsCallback(BaseCallback):
    def __init__(self, save_path, save_model_path, save_freq, verbose=0, print_freq=100000):
        super(SaveActionsRewardsCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_model_path = save_model_path
        self.save_freq = save_freq
        self.actions = []
        self.print_freq = print_freq
        self.rewards = []

    def _on_step(self) -> bool:
        self.actions.append(self.locals['actions'])
        self.rewards.append(self.locals['rewards'])
        if self.num_timesteps % self.print_freq == 0:
            print('------------------------------------')
            print(f"Timestep: {self.num_timesteps}")
            print('------------------------------------')
        
        if self.num_timesteps % self.save_freq == 0:
            # Guardar el modelo
            model_save_path = f"{self.save_model_path}.zip"
            self.model.save(model_save_path)
            print(f"Modelo guardado en: {model_save_path}")
        
        return True

    def _on_training_end(self) -> None:
        with open(self.save_path, 'w') as f:
            for i in range(len(self.actions)):
                f.write(f"Timestep {i+1}:\n")
                f.write(f"Actions: {self.actions[i]}\n")
                f.write(f"Rewards: {self.rewards[i]}\n\n")

def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name="mapas/loop",
        max_steps=10000,
        domain_rand=1,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,
        full_transparency=False,
        distortion=True,
        style='synthetic',
        draw_curve=True,
        start_pose=[[0.35, 0, 0.35], 0],
        user_tile_start=[1,1],
        goal=[2,3],
    )

# Crear un entorno vectorizado para el entrenamiento
vec_env = make_vec_env(make_duckietown_env, n_envs=4, seed=123)

# Callback para guardar acciones, recompensas y el modelo
save_path = "actions_rewardsRectoNwe.txt"
save_model_path = "model_step"
save_freq = 100000  # Frecuencia para guardar el modelo

save_callback = SaveActionsRewardsCallback(save_path=save_path, save_model_path=save_model_path, save_freq=save_freq)

model = PPO("MlpPolicy", vec_env, verbose=1, device="cuda", **ppo_params)

model.load("final/modelitoCarril3.zip")

model.learn(total_timesteps=1800000, callback=save_callback, log_interval=100000)

# Guardar el modelo final después del entrenamiento
model.save("PorfaFuncionaCirculo")

del model # Eliminar el modelo para demostrar guardado y carga
