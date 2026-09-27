"""SAC kontra PPO na wahadle przy tym samym budzecie krokow."""

# ================= 1. NARZEDZIA =================
import argparse  # czytanie ustawien z terminala
import csv  # zapis tabeli wynikow
import time  # pomiar czasu
from pathlib import Path  # sciezki do plikow

import gymnasium as gym  # srodowiska
from stable_baselines3 import PPO, SAC  # algorytmy
from stable_baselines3.common.evaluation import evaluate_policy  # ocena

# ================= 2. USTAWIENIA =================
RUN = Path("runs/week04_pendulum")  # folder na wszystko z tego eksperymentu
RESULTS = RUN / "results.csv"  # wspolna tabela wynikow
ALGOS = {"sac": SAC, "ppo": PPO}  # nazwa -> klasa algorytmu


# ================= 3. WEJSCIE: czego chce uzytkownik =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS), required=True)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


# ================= 4. WYJSCIE: dopisz wynik do tabeli =================
def save_result(row):
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# ================= 5. EKSPERYMENT =================
def main():
    # 5a. przygotowanie
    args = parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    name = f"{args.algo}_{args.steps}_seed{args.seed}"

    # 5b. GDZIE — srodowisko                                   [RDZEN]
    env = gym.make("Pendulum-v1")

    # 5c. CZYM — model                                         [RDZEN]
    model = ALGOS[args.algo](
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=0,
        tensorboard_log=str(RUN / "tb"),
        device="cpu",
    )

    # 5d. JAK DLUGO — trening, z pomiarem czasu                [RDZEN]
    start = time.perf_counter()
    model.learn(total_timesteps=args.steps, tb_log_name=name, progress_bar=True)
    minutes = (time.perf_counter() - start) / 60

    # 5e. zapisz model, zeby dalo sie go obejrzec
    model.save(RUN / name)

    # 5f. JAK WYSZLO — ocena na osobnym srodowisku             [RDZEN]
    eval_env = gym.make("Pendulum-v1")
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    eval_env.close()

    # 5g. zapisz wynik do tabeli i pokaz go
    save_result(
        {
            "algo": args.algo,
            "steps": args.steps,
            "seed": args.seed,
            "mean": round(mean, 1),
            "std": round(std, 1),
            "minutes": round(minutes, 1),
        }
    )
    print(f"{name}: {mean:.1f} +/- {std:.1f}  ({minutes:.1f} min)")


# ================= 6. START =================
if __name__ == "__main__":
    main()
