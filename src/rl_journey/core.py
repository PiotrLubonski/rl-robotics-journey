"""Male, czyste funkcje RL, ktore da sie przetestowac."""

import numpy as np


def td_target(reward, next_q_max, gamma, terminated):
    """Cel Q-learningu: r + gamma * max Q(s'). Na koncu epizodu samo r."""
    if terminated:
        return reward
    return reward + gamma * next_q_max


def q_update(q, target, alpha):
    """Mieszanie farb: (1 - alpha) * stare + alpha * cel."""
    return (1 - alpha) * q + alpha * target


def ppo_updates(n_envs, n_steps, batch_size, n_epochs):
    """Ile razy siec sie aktualizuje na jednej porcji danych PPO."""
    porcja = n_envs * n_steps  # TU: ile przejsc zbieraja wszystkie srodowiska razem
    paczki = porcja // batch_size  # TU: na ile paczek dzielimy porcje
    return paczki * n_epochs  # TU: ile aktualizacji przy n_epochs przejsciach


def smooth_penalty(prev_action, action, weight):
    """Kara za szarpanie: -weight * suma kwadratow zmian akcji."""
    return -weight * np.sum((action - prev_action) ** 2)  # TU: jedna linijka z numpy
