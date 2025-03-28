#TP2 Marouane ACHARIFI

#exercice 1

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

print(f"Espace d'actions: {env.action_space}")
print(f"Espace d'observations: {env.observation_space}")

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Observation: {observation}, Récompense: {reward}")
    if done:
        env.reset()

env.close()


#exercice 2

import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

n_actions = env.action_space.n
n_states = env.observation_space.n
Q = np.zeros((n_states, n_actions))

print("Q-table avant apprentissage :")
print(Q)




#exercice 3

import gymnasium as gym
import numpy as np

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
num_episodes = 5000

env = gym.make("FrozenLake-v1", is_slippery=True)

n_actions = env.action_space.n
n_states = env.observation_space.n
Q = np.zeros((n_states, n_actions))

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(Q[state])  
        
        next_state, reward, done, _, _ = env



#exercice 4 

import gymnasium as gym
import numpy as np

num_eval_episodes = 1000
success_count = 0

env = gym.make("FrozenLake-v1", is_slippery=True)

for episode in range(num_eval_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])
        
        next_state, reward, done, _, _ = env.step(action)

        state = next_state

    if reward == 1.0:
        success_count += 1

success_rate = success_count / num_eval_episodes * 100
print(f"Taux de succès après entraînement : {success_rate:.2f}%")


