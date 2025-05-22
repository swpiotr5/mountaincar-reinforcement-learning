import gymnasium as gym
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import os
env = gym.make("MountainCar-v0")

#TEST 1 
alpha = 0.2   # Zwiększamy szybkość uczenia           
gamma = 0.99      
epsilon_decay = 0.999    # wolniejszy spadek eksploracji       


n_bins = 40  # Zwiększamy rozdzielczość przestrzeni stanów
episodes = 15000  # więcej epizodów na naukę
min_epsilon = 0.01       # Mniejsza minimalna eksploracja
epsilon = 1.0

algorithms = ['q', 'sarsa']
episode_rewards_all = {alg: [] for alg in algorithms}

# Skrajne wartości
obs_low = env.observation_space.low
obs_high = env.observation_space.high
# Rozmiar jednej strefy 
bin_size = (obs_high - obs_low) / n_bins

# Tablica Q:  liczba_stref_pozycji × liczba_stref_predkosci × liczba_akcji czyli 20x20x3 
q_table = np.zeros((n_bins, n_bins, env.action_space.n))

param_sets = {
    "Baseline":     {"alpha": 0.2, "gamma": 0.99,  "epsilon_decay": 0.995},
    "FastLearn":    {"alpha": 0.5, "gamma": 0.95,  "epsilon_decay": 0.99},
    "SlowThinker":  {"alpha": 0.1, "gamma": 0.99,  "epsilon_decay": 0.995},
    "FutureSeeker": {"alpha": 0.2, "gamma": 0.999, "epsilon_decay": 0.997},
    "Explorator":   {"alpha": 0.2, "gamma": 0.99,  "epsilon_decay": 0.999}
}

results = {}

for param_name, params in param_sets.items():
    for variant in ['default', 'custom']:
        for alg in algorithms:
            print(f"Trenuję: {alg.upper()} - {variant} - {param_name}")
            
            alpha = params["alpha"]
            gamma = params["gamma"]
            epsilon_decay = params["epsilon_decay"]

            if variant == 'default':
                rewards = train_default_reward(
                    env, n_bins, episodes, obs_low, bin_size,
                    alpha, gamma, min_epsilon, epsilon_decay,
                    algorithm=alg
                )
            else:
                rewards = train_custom_reward(
                    env, n_bins, episodes, obs_low, bin_size,
                    alpha, gamma, min_epsilon, epsilon_decay,
                    algorithm=alg
                )
            
            results[f"{alg}_{variant}_{param_name}"] = rewards

param_sets_used = set(label.split("_")[-1] for label in results.keys())

output_dir = "."

for param_name in param_sets_used:
    plt.figure(figsize=(10, 5))

    for variant in ['default', 'custom']:
        for alg in ['q', 'sarsa']:
            label = f"{alg}_{variant}_{param_name}"
            if label in results:
                rewards = results[label]
                rolling = [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))]
                plt.plot(rolling, label=label.upper())

    plt.xlabel("Epizod")
    plt.ylabel("Średnia nagroda (100 ep)")
    plt.title(f"Q vs SARSA – zestaw: {param_name}")
    plt.legend()
    plt.grid(True)

    filename = os.path.join(output_dir, f"comparison_{param_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Zapisano zestawowy wykres: {filename}")