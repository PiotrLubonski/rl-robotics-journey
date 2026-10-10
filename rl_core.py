"""Male, czyste funkcje RL, ktore da sie przetestowac."""


def td_target(reward, next_q_max, gamma, terminated):
    """Cel Q-learningu: r + gamma * max Q(s'). Na koncu epizodu samo r."""
    if terminated:
        return reward
    return reward + gamma * next_q_max


def q_update(q, target, alpha):
    """Mieszanie farb: (1 - alpha) * stare + alpha * cel."""
    return (1 - alpha) * q + alpha * target
