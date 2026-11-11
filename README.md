# RL Robotics Journey

My 3-month self-study of reinforcement learning for robotics — from tabular Q-learning to PPO, heading toward robot manipulation in MuJoCo and a real low-cost arm.

![PPO agent landing on the Moon](media/lunar_ppo.gif)

## Progress

| Week | Topic | Result |
|------|-------|--------|
| 1 | Random agent on CartPole | baseline: 21 steps |
| 2 | Tabular Q-learning from scratch on FrozenLake | 100% (deterministic), 74% (slippery — near the theoretical ceiling) |
| 3 | PPO on LunarLander (Stable-Baselines3) | **212.4 ± 74.0** — solved (threshold 200) |

## Experiment: how the discount factor γ changes behaviour

PPO on LunarLander-v3, 300k steps (shorter than the main run), 3 seeds per setting. Hypothesis written before running.

| γ | Horizon 1/(1−γ) | Mean return | Min | Max | Behaviour |
|---|---|---|---|---|---|
| 0.999 | 1000 | 102.4 | −42.1 | 192.2 | lands (~650 steps) |
| 0.99 | 100 | 5.2 | −128.6 | 183.3 | unstable — depends on the seed |
| 0.9 | 10 | −136.0 | −167.5 | −103.8 | hovers until the time limit |

![Short-sighted agent hovering](media/lunar_gamma09.gif)

**Takeaway.** The higher γ, the more it pays off for the agent to land; with low γ it prefers to hover and never risks a crash. The fuel burned while hovering is spread over hundreds of steps, so with γ = 0.9 the agent perceives the cost of hovering *forever* as only ~1.7 points — while a crash (−100) is right in front of it.

The difference between γ = 0.999 and γ = 0.99 is within seed-to-seed variance; only γ = 0.9 is clearly worse.

## Run it

```bash
uv sync
uv run python experiments/week03/ppo_lunar.py
uv run python experiments/week03/watch.py runs/week03_ppo_lunar/ppo_lunar
Run the tests (17 tests; the reward wrappers are tested on a fake environment, no physics needed):
uv run pytest -v
```

## Repository layout

```
experiments/   one-off scripts, grouped by week
notes/         notes and analogies (in Polish)
src/           reusable code (custom environment coming in week 8)
media/         GIFs for this README
JOURNAL.md     learning log (in Polish)
```

### Week 5 — What makes tuned PPO work on HalfCheetah?

Same algorithm (PPO), same budget (1M steps), seed 0, same 50 evaluation episodes.
"Weak" = episode return < 1000.

| Variant | Mean of good runs | Weak runs |
|---|---|---|
| SB3 defaults | 1585 | 0% |
| + obs/reward normalization only | 1730 | 0% |
| + log_std_init = -2 only | 1385 | 42% (slow walking, no flips) |
| + both | 4567 | 0% |
| Full RL Zoo config (10 changes) | 6465 | 30% (flips onto its back) |

Takeaways:
- Each setting alone barely helps (or hurts); together they triple the score. An interaction, not a sum.
- The full Zoo config runs faster but flips in ~1/3 of episodes; the two-setting version never flipped. Speed vs. reliability.
- 10 evaluation episodes overestimated the Zoo model (5385 vs 4699 on 50). Single seed per variant, so results are preliminary.

### Week 5 — What makes tuned PPO work on HalfCheetah? (corrected in week 6)

Same algorithm (PPO), same budget (1M steps), seed 0, same 50 evaluation episodes.
Posture is measured directly from the torso angle (added in week 6), not guessed from the score.

| Variant | Mean return | On its back | Other failures |
|---|---|---|---|
| SB3 defaults | 1585 | 100% | — |
| + obs/reward normalization only | 1730 | ?% | ? |
| + log_std_init = -2 only | 1094 | 24% | 13 slow episodes |
| + both | 4567 | 0% | — |
| Full RL Zoo config (10 changes) | 4699 | 0% | 30% stuck tilted ~50° |

**Correction:** in week 5 I reported that the default policy "never flips" and the Zoo policy
"flips onto its back". Both were wrong. A flip detector showed that the default policy runs
on its back in every episode, and that the Zoo failures are the cheetah getting stuck tilted
~50°, not lying on its back. A stable score (1585 ± 61) said nothing about posture.

### Week 6 — Fixing the back-running cheetah with reward shaping

The HalfCheetah reward pays only for forward speed, so running on its back is a valid
solution (specification gaming). I added a posture penalty during training only:
`weight * (cos(torso_angle) - 1)`, which is 0 when level and −2 when upside down.
Evaluation uses the original reward.

| SB3 default PPO, 1M steps, seed 0 | Mean return (original reward) | On its back |
|---|---|---|
| no penalty (control) | 1585 | 100% |
| posture penalty, weight 1 | 2764 | 0% |
| + obs/reward normalization only | 1730 | 100% | — |

The penalty did more than forbid a behavior: it pushed the agent out of a local optimum.
Upright running scores 74% higher even on the original reward. Single seed, so preliminary.

### Week 5 — What makes tuned PPO work on HalfCheetah? (corrected in week 6)

Same algorithm (PPO), same budget (1M steps), seed 0, same 50 evaluation episodes.
Posture is measured directly from the torso angle (added in week 6), not guessed from the score.

| Variant | Mean return | On its back | Other failures |
|---|---|---|---|
| SB3 defaults | 1585 | 100% | — |
| + obs/reward normalization only | 1730 | 100% | — |
| + log_std_init = -2 only | 1094 | 24% | 13 slow episodes |
| + both | 4567 | 0% | — |
| Full RL Zoo config (10 changes) | 4699 | 0% | 30% stuck tilted ~50° |

**Correction:** in week 5 I reported that the default policy "never flips" and the Zoo policy
"flips onto its back". Both were wrong. A flip detector showed that the default policy runs
on its back in every episode, and that the Zoo failures are the cheetah getting stuck tilted
~50°, not lying on its back. A stable score (1585 ± 61) said nothing about posture.

### Week 6 — Fixing the back-running cheetah with reward shaping

The HalfCheetah reward pays only for forward speed, so running on its back is a valid
solution (specification gaming). I added a posture penalty during training only:
`weight * (cos(torso_angle) - 1)`, which is 0 when level and −2 when upside down.
Evaluation uses the original reward.

| SB3 default PPO, 1M steps, seed 0 | Mean return (original reward) | On its back |
|---|---|---|
| no penalty (control) | 1585 | 100% |
| posture penalty, weight 1 | 2764 | 0% |

The penalty did more than forbid a behavior: it pushed the agent out of a local optimum.
Upright running scores 74% higher even on the original reward. Single seed, so preliminary.
The new gait still pitches the torso a lot (up to ~70°, vs ~20° for the Zoo policy).

#### Smoothness penalty (action-rate penalty)

Added on top of the posture penalty during training: `smooth * sum((a_t - a_{t-1})^2)` over the 6 joints.
Jerk = mean |a_t − a_{t−1}| per joint (0 = smooth, 2 = jumping between −1 and +1 every step).
Return over 50 episodes; jerk and saturation over 10 episodes.

| smooth | Mean return | Jerk | Saturated actions | On its back | Final policy std |
|---|---|---|---|---|---|
| 0 | 2764 | 0.853 | 44% | 0% | 0.22 |
| 0.1 | 1383 | 0.575 | 32% | 44% | 0.21 |
| 0.3 | 1728 | 0.459 | 14% | 4% | 0.13 |
| reference: normalization + log_std_init = -2, no penalties | 4567 | 0.499 | 8% | 0% | — |

The penalty reduced jerk, but at a large cost in speed. At smooth = 0.3 the exploration noise collapsed much faster:
penalizing action changes also penalizes random exploration. The smoothest *and* fastest policy came from fixing
the cause (small initial std + normalization), not from penalizing the symptom.
Single seed; the 44% flips at smooth = 0.1 are not explained yet.

### Week 7 — Goal-conditioned RL and HER

Sparse reward: 0 within the threshold of the goal, −1 otherwise. SAC with and without
Hindsight Experience Replay (same code, only the replay buffer differs), 20k steps, seed 0,
50 evaluation episodes.

| Environment | Random agent | SAC | SAC + HER |
|---|---|---|---|
| FetchReach (5 cm threshold) | 4% | 94% | 100% |
| PointReach, my own env (threshold 0.05) | 0% | 4% | 100% |

HER matters most when random successes are rare. On FetchReach, 4% random hits were enough
for SAC to bootstrap. On PointReach, with 0% random hits, SAC without HER learned nothing,
while SAC + HER solved the task in the same 6.5 minutes.
`compute_reward` must work on batches (HER relabels whole batches at once); a test guards it.

## Tygodnie 8–9: ArmReach, własne ramię od zera

Własne środowisko Gymnasium: ramię z dwoma stawami (0,5 m + 0,5 m) na płaszczyźnie ma dotknąć celu końcówką.
Akcja to siła w stawach (−1…1), obserwacja to 10 liczb (cos i sin kątów, prędkości, cel, wektor końcówka→cel),
nagroda to −odległość − 0,01·Σa². Sukces: końcówka bliżej niż 5 cm na końcu epizodu (100 kroków).
Kod jest pakietem Pythona (`rl_journey`, `gym.make("rl_journey:ArmReach-v0")`), 59 testów pytest uruchamia CI przy każdym pushu.

Wszystkie wyniki: SAC, 1 seed treningu, ocena na 200 epizodach (seedy 1000+), chyba że napisano inaczej.

### Uczenie i punkt odniesienia

| Wariant | Sukces | Kraniec zasięgu (cele 0,80–0,91 m) |
|---|---|---|
| losowy agent | 0% (dotknął choć raz: 8%) | |
| SAC, 30 tys. kroków | 50% (50 epizodów) | |
| SAC, 50 tys. kroków | 65,0% | 20,8% (5 z 24) |
| SAC, 100 tys. kroków | 69,5% | 29,2% (7 z 24) |
| ręczny regulator (jakobian + hamowanie, bez uczenia) | 100,0% | 100% (24 z 24) |

### Co widzi ramię (50 tys. kroków)

| Obserwacja | Sukces |
|---|---|
| z prędkością | 65,0% |
| bez prędkości | 3,5% |
| bez prędkości, 2 ostatnie klatki | 19,0% |
| bez prędkości, 4 ostatnie klatki | 13,0% |

### Nagroda (50 tys. kroków)

| Nagroda | Sukces |
|---|---|
| −odległość | 65,0% |
| −odległość + premia +1 za każdy krok w progu 5 cm | 60,5% |

### Odporność na zmiany świata

![Odporność](docs/img/week09_robustness.png)

| Model | nominalny | długości ±10% | tarcie ×0,5–2 | silnik ±30% | szum 2 cm | wszystko naraz |
|---|---|---|---|---|---|---|
| ręczny regulator | 100,0% | 100,0% | 99,5% | 100,0% | 100,0% | 99,0% |
| SAC | 65,0% | 60,5% | 65,0% | 65,0% | 67,5% | 63,5% |
| SAC + losowa fizyka | 56,5% | 49,5% | 56,5% | 56,5% | 55,5% | 50,0% |
| SAC + losowa fizyka + szum | 33,5% | 34,0% | 34,5% | 33,0% | 35,0% | 33,5% |

### Wnioski

1. **Granica jest w treningu, nie w zadaniu.** Ręczny regulator, który zna fizykę ramienia, ma 100% (co najmniej 99% w każdym zmienionym świecie i 24 z 24 na krańcu zasięgu). SAC uczy się tego samego z prób i po 100 tys. kroków ma 69,5%, a na krańcu zasięgu 29,2%.
2. **Prędkość w obserwacji decyduje.** Bez niej ramię dotyka celu, ale nie umie się zatrzymać (3,5%). Pamięć dwóch klatek odzyskuje część (19%), bo sieć musi sama nauczyć się odejmować prawie równe liczby. Gdy prędkość da się zmierzyć, lepiej podać ją wprost.
3. **Premia za próg nie pomogła.** Ważniejsza lekcja: ten sam model miał 56% na 50 epizodach i 65% na 200, więc porównania robię tylko tym samym sprawdzianem na 200 epizodach.
4. **Polityka ze sprzężeniem zwrotnym jest odporna.** Zmiana długości o ±10%, tarcia ×0,5–2, siły silnika o ±30% i szum 2 cm mieszczą się w szumie oceny, zarówno dla SAC, jak i dla regulatora.
5. **Domain randomization nie jest darmowe.** Stary model nie był kruchy, więc losowy świat dał tylko koszt: −8,5 pp za losową fizykę i kolejne −23 pp za szum w treningu. Najpierw mierzę odporność, a zakres losowania dobieram do zmierzonej różnicy między symulacją a rzeczywistością (np. enkoder serwa STS3215 ma ok. 0,8 mm rozdzielczości na końcówce, a nie 2 cm).
6. **Kiedy RL, a kiedy klasyczne sterowanie.** Tu wygrywa regulator, bo model ramienia jest znany i prosty. RL ma sens tam, gdzie takiego modelu nie da się napisać: kontakt, miękkie materiały, sterowanie z obrazu.

### Ograniczenia

- Jeden seed treningu na wariant; różnice poniżej ok. 7 pp traktuję jako szum.
- Symulator jest łagodny: wektor końcówka→cel liczy się z prawdziwych długości ramienia. Na robocie liczy się go z modelu albo z kamery, więc błąd długości byłby też błędem obserwacji.

### Jak odtworzyć

```powershell
uv sync
uv run pytest
uv run python experiments/week09/robustness.py runs/week08/sac_arm_cossin_vel1_50000_seed0
uv run python experiments/week09/robustness.py --hand
uv run python experiments/week09/plot_robustness.py
```