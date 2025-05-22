import numpy as np
import random

def discretize(obs, obs_low, bin_size, n_bins):
    # Przekształcenie do numerów stref
    ratios = (obs - obs_low) / bin_size
    bins = np.floor(ratios).astype(int)

    bins = np.clip(bins, 0, n_bins - 1)

    return tuple(bins)

def choose_action(q_table, state, epsilon, n_actions):
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return np.argmax(q_table[state])
    
def update_q(q_table, state, action, reward, next_state, alpha, gamma):
    max_future_q = np.max(q_table[next_state])
    current_q = q_table[state][action]
    q_table[state][action] += alpha * (reward + gamma * max_future_q - current_q)

def update_sarsa(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    current_q = q_table[state][action]
    next_q = q_table[next_state][next_action]
    q_table[state][action] += alpha * (reward + gamma * next_q - current_q)

def train_custom_reward(env, n_bins, episodes, obs_low, bin_size, alpha, gamma, min_epsilon, epsilon_decay, algorithm='q'):
    q_table = np.zeros((n_bins, n_bins, env.action_space.n))
    epsilon = 1.0
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, obs_low, bin_size, n_bins)
        done = False
        total_reward = 0
        action = choose_action(q_table, state, epsilon, env.action_space.n)

        while not done:
            next_obs, _, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs, obs_low, bin_size, n_bins)
            done = terminated or truncated

            next_action = choose_action(q_table, next_state, epsilon, env.action_space.n)

            # Nagroda customowa
            position, velocity = next_obs
            reward = -1 + abs(velocity) * 0.5
            if position >= 0.5:
                reward = 100
            if -0.6 <= position <= -0.4 and abs(velocity) < 0.03:
                reward -= 1

            if algorithm == 'q':
                update_q(q_table, state, action, reward, next_state, alpha, gamma)
            elif algorithm == 'sarsa':
                update_sarsa(q_table, state, action, reward, next_state, next_action, alpha, gamma)

            state = next_state
            action = next_action if algorithm == 'sarsa' else choose_action(q_table, next_state, epsilon, env.action_space.n)
            total_reward += reward

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        rewards.append(total_reward)

    return rewards, q_table


def train_default_reward(env, n_bins, episodes, obs_low, bin_size, alpha, gamma, min_epsilon, epsilon_decay, algorithm='q'):
    q_table = np.zeros((n_bins, n_bins, env.action_space.n))
    epsilon = 1.0
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretize(obs, obs_low, bin_size, n_bins)
        done = False
        total_reward = 0
        action = choose_action(q_table, state, epsilon, env.action_space.n)

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_obs, obs_low, bin_size, n_bins)
            done = terminated or truncated

            next_action = choose_action(q_table, next_state, epsilon, env.action_space.n)

            if algorithm == 'q':
                update_q(q_table, state, action, reward, next_state, alpha, gamma)
            elif algorithm == 'sarsa':
                update_sarsa(q_table, state, action, reward, next_state, next_action, alpha, gamma)

            state = next_state
            action = next_action if algorithm == 'sarsa' else choose_action(q_table, next_state, epsilon, env.action_space.n)
            total_reward += reward

        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        rewards.append(total_reward)

    return rewards, q_table
