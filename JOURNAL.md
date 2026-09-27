# Dziennik nauki RL w robotyce

## Tydzień 1

### Poniedziałek
- Setup: uv, git, GitHub CLI, Python 3.12
- Projekt + pierwszy commit
- Pytanie otwarte:

Episodes:  100
Mean:      21.0
Std:       9.4
Min / Max: 10 / 59
Terminated: 100, Truncated: 0

## Tydzień 2
is_slippery = True
alfa = 0.8

Tabela Q (wiersz = stan, kolumna = lewo/dol/prawo/gora):
[[0.44  0.171 0.172 0.171]
 [0.01  0.097 0.001 0.177]
 [0.023 0.061 0.07  0.072]
 [0.012 0.046 0.007 0.057]
 [0.352 0.    0.358 0.042]
 [0.    0.    0.    0.   ]
 [0.118 0.001 0.54  0.   ]
 [0.    0.    0.    0.   ]
 [0.248 0.001 0.013 0.339]
 [0.005 0.25  0.021 0.062]
 [0.124 0.803 0.117 0.076]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.104 0.145 0.858 0.168]
 [0.337 0.845 0.768 0.698]
 [0.    0.    0.    0.   ]]

Skutecznosc po treningu: 17%

## Tydzień 3

### Tydzień 3 — PPO na LunarLander-v3
- 500k kroków, 8 środowisk, seed 0
- Trening (ep_rew_mean): ~132 | Ewaluacja (deterministic): 212,4 ± 74
- ROZWIĄZANE (próg 200). Ląduje za każdym razem, bez przechyłów.
- Różnica trening/ewaluacja: losowanie akcji w treningu + opóźnienie średniej ze 100 epizodów
- ±74 to głównie różne warunki startowe (losowe pchnięcie, teren), nie pech
- explained_variance ~idealna, lekki spadek pod koniec — krytyk goni zmieniającą się politykę

### Tydzień 3, środa — eksperyment z gamma
Hipoteza (napisana PRZED uruchomieniem):

| gamma | horyzont 1/(1−γ) | przewidywany wynik ewaluacji | co będzie robił lądownik |
|-------|------------------|------------------------------|--------------------------|
| 0.999 |  1000            |      250                     | wyląduje precyzyjnie     |
| 0.99  |   100            |      120                     |  będzie nad lądowiskiem  |
| 0.9   |    10            |     80                       |    losowo będzie wisiał  |

Dlaczego tak myślę: no bo potrzebujemy ok 200 kroków do lądowania, 1000 to jest wystarczajaca liczba az nadto. 100 to jest troche za malo. 10 to bardzo malo i bedziemy sie opierac na losowych ruchach

## Checkpoint #1 — koniec tygodnia 3
- Umiem:rozróźniać gamma
- Sprawia mi trudność: czytanie tensorboarda
- Co chcę poprawić w sposobie nauki: trochę więcej tłumaczenia pythona i praktyki z nim.

## Tydzień 4, przykladowe liczby
=== Pendulum-v1 ===
obserwacja: Box([-1. -1. -8.], [1. 1. 8.], (3,), float32)
akcja:      Box(-2.0, 2.0, (1,), float32)
przykladowa akcja: [1.3106172]
losowy agent: -1174.6 +/- 266.7
epizody koncza sie przez: truncated

=== Reacher-v5 ===
obserwacja: Box(-inf, inf, (10,), float64)
akcja:      Box(-1.0, 1.0, (2,), float32)
przykladowa akcja: [-0.83242106 -0.79425025]
losowy agent: -43.0 +/- 4.3
epizody koncza sie przez: truncated

### Tydzień 4 — SAC kontra PPO na Pendulum-v1 (2 seedy, domyślne hiperparametry)
| Ustawienie | Średni wynik | Czas |
|---|---|---|
| SAC 20k | −132 | 4,3 min |
| PPO 20k | −1248 (losowy poziom) | 0,3 min |
| PPO 200k | −532 | 3,5 min |
Wniosek: SAC ~25× lepiej wykorzystuje każde doświadczenie (bufor), kosztem ~12× więcej obliczeń na krok.
Na prawdziwym robocie drogie są kroki, nie minuty → SAC.