#TP1 Marouane ACHARIFI
#partie 1 

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

#partie 2
#exercice 1

print(f"Espace d'actions : {env.action_space}")
print(f"Espace d'observation : {env.observation_space}")

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action : {action}, Observation : {observation}, Reward : {reward}")
    if done :
        env.reset()

env.close

#exercice 2

import gymnasium as gym 

env = gym.make("CartPole-v1", render_mode="human")

num_steps = 10  
observation, _ = env.reset() 

for step in range(num_steps):
    action = env.action_space.sample() 
    new_observation, reward, done, _, _ = env.step(action)  

    print(f"Étape {step + 1}:")
    print(f"Ancienne observation : {observation}")
    print(f"Action : {action}")
    print(f"Nouvelle observation : {new_observation}")
    print(f"Récompense : {reward}, Terminé ? {done}")
    print("-" * 40)

    observation = new_observation

    if done:
        print("L'épisode est terminé. Réinitialisation de l'environnement.")
        observation, _ = env.reset()  

env.close()  


#exercice 3

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, _ = env.reset(seed=42)

done = False
step_count = 0

print("Contrôle manuel : entrez 0 (gauche) ou 1 (droite) pour déplacer le chariot.")

while not done:
    env.render()

    action = input("Entrez votre action (0=gauche, 1=droite) : ")
    
    if action not in ["0", "1"]:
        print("Action invalide ! Entrez 0 ou 1.")
        continue
    
    action = int(action)

    observation, reward, done, _, _ = env.step(action)
    
    print(f"\nÉtape {step_count + 1}:")
    print(f"Action : {action}, Observation : {observation}, Reward : {reward}")
    print(f"  Terminé ? {done}")
    print("-" * 40)

    step_count += 1

print(f"\nL'épisode est terminé après {step_count} étapes.")

env.close()

#exercice 4

import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 10
episode_durations = []

for episode in range(num_episodes):
    observation, _ = env.reset()  
    done = False
    step_count = 0  

    while not done:
        action = env.action_space.sample() 
        observation, reward, done, _, _ = env.step(action)
        step_count += 1 

    episode_durations.append(step_count)  
    print(f"Épisode {episode + 1} terminé en {step_count} étapes.")


average_duration = np.mean(episode_durations)
std_duration = np.std(episode_durations) 
print("\nRésumé des performances de la politique aléatoire :")
print(f"  - Nombre d'épisodes : {num_episodes}")
print(f"  - Durée moyenne : {average_duration:.2f} étapes")
print(f"  - Écart-type : {std_duration:.2f}")

env.close() 