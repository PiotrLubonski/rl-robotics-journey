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