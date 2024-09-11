import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import TD3
import torch
import os
import matplotlib.pyplot as plt

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

def make_duckietown_env(seed=None):
    return Simulator(
        seed=seed,  
        map_name="straight_road",  
        max_steps=80000,  
        domain_rand=0,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=1,  
        start_pose=[[0.35, 0, 0.35], 0],
        user_tile_start=[0, 0],
        style='synthetic',
    )

# Crear un entorno para pruebas
test_env = make_duckietown_env(seed=42)

# Cargar el modelo entrenado de DDPG
load_path = r"C:\Users\karol\Documents\Master\gymduck\gym-duckietown\algoritmos\td\model\modeloDDPG.zip"
model = TD3.load(load_path, device=device)

# Probar el modelo en un solo episodio
obs = test_env.reset()
episode_reward = 0
step_count = 0
rewards_per_step = []

while True:
    # Predecir acción usando el modelo DDPG entrenado
    action, _states = model.predict(obs, deterministic=True)
    
    # Tomar el paso en el entorno
    obs, rewards, done, info = test_env.step(action)
    episode_reward += rewards
    rewards_per_step.append(rewards)
    step_count += 1

    # Renderizar el entorno
    test_env.render()

    if done:
        break

# Mostrar el total de recompensas y pasos
print(f"Recompensa total del episodio: {episode_reward}")
print(f"Total de pasos: {step_count}")

# Eliminar la última recompensa si se rompe el bucle
if rewards_per_step:
    rewards_per_step.pop()

# Graficar recompensas por paso a través del tiempo
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(rewards_per_step) + 1), rewards_per_step, marker='o', linestyle='-')
plt.title("Recompensas por Paso en el Test")
plt.xlabel("Paso")
plt.ylabel("Recompensa")
plt.grid(True)
plt.show()

# Cerrar el entorno
test_env.close()
