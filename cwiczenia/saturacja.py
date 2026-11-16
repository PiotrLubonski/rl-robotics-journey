import numpy as np

KP, LIMIT = 998.0, 2.94  # sztywność serwa i limit silnika (N·m)


def sila(blad_stopnie):
    """Siła serwa przy danym błędzie kąta: kp · błąd, ale nie więcej niż limit."""
    blad_rad = np.deg2rad(blad_stopnie)
    return float(np.clip(KP * blad_rad, -LIMIT, LIMIT))


for b in [0.05, 0.1, 0.17, 1, 10, -10]:
    print(
        f"błąd {b:6.2f}° → wzór {KP * np.deg2rad(b):7.2f} N·m → serwo daje {sila(b):5.2f} N·m"
    )
