import gym_duckietown
from gym_duckietown.simulator import Simulator
from stable_baselines3 import PPO
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
        user_tile_start=[0,0],
        style='synthetic',
        # user_tile_start=[0,0],
        # draw_curve=True,
        # draw_bbox=True,
        # start_pose=[[0.35, 0, 0.35], -1]
    )

# Crear un entorno para pruebas
test_env = make_duckietown_env(seed=42)

# Cargar el modelo entrenado  
# modeloAnguloDeg modeloConAnguloSinRetroceso50kst-1kEp modeloAnguloDeg
load_path = r"C:\Users\karol\Documents\Master\gymduck\gym-duckietown\algoritmos\ppo\model\modeloNuevoRetroDeg.zip"
model = PPO.load(load_path, device=device)

# Probar el modelo en un solo episodio
obs = test_env.reset()
episode_reward = 0
step_count = 0
rewards_per_step = []

while True:
    action, _states = model.predict(obs)
    
    obs, rewards, done, info = test_env.step(action)
    episode_reward += rewards
    rewards_per_step.append(rewards)
    step_count += 1

    test_env.render()

    if done:
        break

# Mostrar el total de recompensas y pasos
print(f"Recompensa total del episodio: {episode_reward}")
print(f"Total de pasos: {step_count}")

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
