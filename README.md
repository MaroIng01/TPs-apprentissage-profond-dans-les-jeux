# TPs-apprentissage-profond-dans-les-jeux

# TP1 :

## Objectif
L'objectif de ce TP est de se familiariser avec les outils essentiels du Reinforcement Learning (RL), notamment OpenAI Gym. Les étudiants vont explorer comment interagir avec un environnement RL et exécuter des actions avant d'implémenter un algorithme d'apprentissage dans les séances suivantes.

---

## A) Partie 1: Présentation des Bibliothèques Clés

### 1. OpenAI Gym
**OpenAI Gym** est une bibliothèque permettant de simuler des environnements interactifs pour tester des algorithmes de RL. Un environnement Gym est défini par :
- Un ensemble **d'états**
- Un ensemble **d'actions**
- Un **système de récompenses**
- Un critère de **fin d’épisode**

### 2. Installation de Gym
```bash
pip install --upgrade gymnasium pygame numpy
```

### 3. Création d'un environnement
```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

env.reset()
```

---

## B) Partie 2: Exercices Pratiques avec OpenAI Gym

### Exercice 1: Découverte et Exploration d'un Environnement Gym
**Objectif** : Comprendre la structure d'un environnement Gym en explorant ses propriétés et ses actions possibles.

✔ **Instructions** :
1. Afficher l'espace d'actions et l'espace d'observations.
2. Exécuter une boucle de simulation avec des actions aléatoires pendant 100 itérations.
3. Observer les valeurs des observations retournées.

```python

print(f"Espace d'actions : {env.action_space}")
print(f"Espace d'observation : {env.observation_space}")

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action : {action}, Observation : {observation}, Reward : {reward}")
    if done :
        env.reset()

env.close

```

---

### Exercice 2: Manipulation des Observations et Récompenses
**Objectif** : Comprendre comment récupérer les observations et les récompenses lors de l'interaction avec l'environnement.

✔ **Instructions** :
1. Prendre une action et récupérer les valeurs retournées (`observation`, `reward`, `done`).
2. Afficher ces valeurs et analyser leur signification.
3. Faire plusieurs essais et noter les variations des observations et des récompenses.

```python
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

```

---

### Exercice 3: Contrôle Manuel de l'Agent
**Objectif** : Permettre à l'utilisateur de contrôler manuellement l'agent pour mieux comprendre l'effet des actions.

✔ **Instructions** :
1. Demander à l'utilisateur d'entrer une action (`0` ou `1`).
2. Exécuter l'action dans l'environnement et afficher les nouvelles observations.
3. Répéter l'opération jusqu'à la fin de l'épisode.
4. Afficher la durée totale de l'épisode avant qu'il ne se termine.

```python
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
```

---

### Exercice 4: Évaluation des Performances d'une Politique Aléatoire
**Objectif** : Mesurer la durée moyenne d'un épisode lorsqu'un agent prend des actions aléatoires.

✔ **Instructions** :
1. Faire exécuter à l'agent des actions aléatoires pendant plusieurs épisodes (ex: 10 épisodes).
2. Calculer la durée moyenne avant que l'épisode ne se termine.
3. Comparer les résultats entre plusieurs exécutions.

```python
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
```
---
# TP2 :

### Exercice 1: Exploration de l'Environnement Frozen Lake
**Objectif** : Apprendre à interagir avec un environnement Gym, explorer les actions possibles et comprendre les observations et récompenses retournées par l'environnement.

✔ **Instructions** :

1. Charger l'environnement FrozenLake-v1 de OpenAI Gym.
2. Afficher les informations de l'espace d'états et d'actions.
3. Exécuter une boucle où l'agent prend des actions aléatoires pendant plusieurs épisodes.
4. Observer les observations et les récompenses obtenues.
```python
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
```

### Exercice 2: Implémentation de la Q-Table et Initialisation
**Objectif** : Créer une Q-Table et la remplir d'initialisations avant d'apprendre.

✔ **Instructions** :

1. Créer une Q-Table de dimension (nombre d'états x nombre d'actions), initialisée à 0.
2. Afficher la Q-Table avant l'apprentissage.
3. Vérifier que chaque état a une liste de valeurs associées aux actions possibles.
   
```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)

n_actions = env.action_space.n
n_states = env.observation_space.n
Q = np.zeros((n_states, n_actions))

print("Q-table avant apprentissage :")
print(Q)

```
### Exercice 3: Implémentation du Q-Learning avec Mise à Jour
**Objectif** : Implémenter l'algorithme Q-learning et mettre à jour la Q-table à chaque épisode.

✔ **Instructions** :

1. Définir les hyperparamètres : taux d'apprentissage (alpha), facteur de discount (gamma), epsilon pour l'exploration.
2. Mettre à jour la Q-table en appliquant la règle de mise à jour du Q-learning :
   Q(s,a)←Q(s,a)+α[R(s,a)+γ 
a 
′
 
max
​
 Q(s 
′
 ,a 
′
 )−Q(s,a)]
3. Exécuter plusieurs épisodes et observer l'évolution de la table.
```python
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

```
### Exercice 4: Évaluation du Q-Learning
**Objectif** : Tester la politique apprise en utilisant la Q-table après l'entraînement et mesurer la performance de l'agent.

✔ **Instructions** :

1. Lancer plusieurs épisodes en exploitant la politique apprise (choisir toujours l'action ayant la plus haute valeur Q).
2. Mesurer le taux de succès de l'agent sur N épisodes (ex. 1000 épisodes).
3. Afficher les résultats pour évaluer si l'agent a bien appris.
   
```python
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


```


---

## 📌 Auteurs
- **Marouane ACHARIFI**  

---

🚀 *Bon apprentissage du Reinforcement Learning !* 🎯

