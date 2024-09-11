import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import torch
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")
# Verificar si CUDA está disponible y establecer el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name="mapas/recto",
        max_steps=7000,  #cambiarlo
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
        goal=[4,1],
        

        # start_pose=[1.75, 0, 1.75],
        # full_transparency=True
    )

# Crear un entorno vectorizado para el entrenamiento
vec_env = make_vec_env(make_duckietown_env, n_envs=1, seed=123)
# It will check your custom environment and output additional warnings if needed
# check_env(vec_env)
# Instanciar el agente PPO
model = PPO("MlpPolicy", vec_env, verbose=1, device=device, n_steps=4096, batch_size=1024)

# Entrenar el agente y registrar recompensas y acciones por paso
num_episodes = 100000
rewards_per_step = []
print(f'Estoy trabajando sin render')
save_path = r"C:\Users\karol\Documents\Master\gymduck\gym-duckietown\model"

for i in range(num_episodes):
    obs = vec_env.reset()
    episode_reward = 0
    step_count = 0
    # print(obs)

    
    while True:
        action, _states = model.predict(obs)
        
        obs, rewards, dones, info = vec_env.step(action)
        # print(obs)
        # print('pasé por aquí')
        # Registrar recompensa por paso
        # if rewards[0] < -1000:
        rewards_per_step.append(rewards[0])
        
        episode_reward += rewards[0]
        
        step_count += 1

                # Mostrar recompensa y acción por paso
        # print(f'Paso {step_count}: Acción = {action}, Recompensa = {rewards[0]}')

        if dones[0]:
            break

        vec_env.render()
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "peeeeee"))
    print(f"Episodio {i+1}: Recompensa = {episode_reward}, Pasos = {step_count}")


# Guardar el modelo entrenado
os.makedirs(save_path, exist_ok=True)
model.save(os.path.join(save_path, "peee"))

# Graficar recompensas por paso a través del tiempo
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(rewards_per_step) + 1), rewards_per_step, marker='o', linestyle='-')
plt.title("Recompensas por Paso")
plt.xlabel("Paso")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()
