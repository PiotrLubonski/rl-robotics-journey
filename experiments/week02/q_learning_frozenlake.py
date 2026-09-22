"""Tabular Q-learning on FrozenLake-v1."""

import gymnasium as gym
import numpy as np

N_EPISODES = 20000
ALPHA = 0.1  # tempo uczenia
GAMMA = 0.95  # dyskontowanie
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999  # mnożnik po każdym epizodzie


def choose_action(q_table, state, epsilon, action_space):
    """Epsilon-greedy: losowo z prawdopodobienstwem epsilon, inaczej najlepsza akcja."""
    if np.random.random() < epsilon:
        return action_space.sample()
    row = q_table[state]
    best = np.flatnonzero(row == row.max())  # indeksy WSZYSTKICH najlepszych
    return int(np.random.choice(best))


def train(env, q_table):
    epsilon = EPS_START
    rewards = []

    for episode in range(N_EPISODES):
        state, info = env.reset()
        total_reward = 0.0

        while True:
            action = choose_action(q_table, state, epsilon, env.action_space)
            next_state, reward, terminated, truncated, info = env.step(action)

            # --- SERCE ALGORYTMU ---
            # cel = nagroda + GAMMA * najlepsze Q w nastepnym stanie
            # ale jesli terminated, to przyszlosc jest warta 0!
            best_next = np.max(q_table[next_state])  # TODO
            target = reward + GAMMA * best_next * (1.0 - float(terminated))  # TODO
            q_table[state, action] += ALPHA * (target - q_table[state, action])
            # -----------------------

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (episode + 1) % 200 == 0:
            recent = np.mean(rewards[-200:])
            print(
                f"Epizod {episode + 1:5d} | sukcesy: {recent:.2%} | eps: {epsilon:.3f}"
            )

    return rewards


def evaluate(env, q_table, n_episodes=100):
    """Bez eksploracji - sama wyuczona polityka."""
    wins = 0
    for _ in range(n_episodes):
        state, info = env.reset()
        while True:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                wins += int(reward > 0)
                break
    return wins / n_episodes


def main():
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    train(env, q_table)

    print("\nTabela Q (wiersz = stan, kolumna = lewo/dol/prawo/gora):")
    print(np.round(q_table, 3))

    print(f"\nSkutecznosc po treningu: {evaluate(env, q_table):.0%}")
    env.close()


if __name__ == "__main__":
    main()
