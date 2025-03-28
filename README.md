# TPs-apprentissage-profond-dans-les-jeux


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

# Création de l'environnement CartPole-v1
env = gym.make("CartPole-v1", render_mode="human")

# Réinitialisation de l'environnement
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
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
print(f"Espace d'actions: {env.action_space}")
print(f"Espace d'observations: {env.observation_space}")

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, Observation: {observation}, Reward: {reward}")
    if done:
        env.reset()

env.close()
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
observation, _ = env.reset()
done = False
step_count = 0

while not done:
    action = input("Entrez votre action (0=gauche, 1=droite) : ")
    if action not in ["0", "1"]:
        print("Action invalide ! Entrez 0 ou 1.")
        continue
    
    action = int(action)
    observation, reward, done, _, _ = env.step(action)
    step_count += 1
    print(f"Étape {step_count} - Observation: {observation}, Récompense: {reward}, Terminé: {done}")

env.close()
print(f"L'épisode s'est terminé après {step_count} étapes.")
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

durations = []
for episode in range(num_episodes):
    observation, _ = env.reset()
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        observation, reward, done, _, _ = env.step(action)
        step_count += 1
    durations.append(step_count)
    print(f"Épisode {episode + 1} terminé en {step_count} étapes.")

print(f"Durée moyenne des épisodes : {np.mean(durations):.2f} étapes")
env.close()
```

---

## Conclusion
Ce TP permet une première immersion dans le Reinforcement Learning avec OpenAI Gym. Les exercices montrent comment interagir avec un environnement RL et comprendre les concepts clés (observations, actions, récompenses). Dans les prochaines sessions, nous implémenterons des algorithmes d’apprentissage comme **Q-Learning** et **Deep Q-Networks (DQN)** pour améliorer la performance de l'agent. 🚀

---

## 📌 Auteurs
- **Marouane ACHARIFI**  

---

🚀 *Bon apprentissage du Reinforcement Learning !* 🎯

