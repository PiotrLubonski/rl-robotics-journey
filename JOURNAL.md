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

### Tydzień 5
=== HalfCheetah-v5 ===
obserwacja: Box(-inf, inf, (17,), float64)
akcja:      Box(-1.0, 1.0, (6,), float32)
przykladowa akcja: [-0.8027939  -0.09275231 -0.33667022  0.33393055  0.89608324  0.41529486]
losowy agent: -281.4 +/- 67.9
epizody koncza sie przez: truncated

### Tydzień 5, poniedziałek — PPO na HalfCheetah-v5 (domyślne ustawienia)
- 1M kroków, seed 0, 19,7 min → 1597,7 ± 45,7 (ok. 2 m/s)
- Biegnie na nogach, głową do przodu, ale szarpie.
- Powód: nagroda nie mówi nic o płynności → sterowanie „wszystko albo nic".
- Na prawdziwym robocie trzeba by dodać karę za gwałtowne zmiany akcji.

Wynik: 5385.0 ± 2458.8 (39.4 min). Zoo (v3, 150 ep.): 5819 ± 664.
- Wynik: potwierdzona (≥ 1800), ale niedoszacowana ~2,7×.
- Szarpanie: stary __% → nowy __%.
Wniosek: ten sam algorytm i kroki, inne ustawienia → 3,4× lepiej.
Ale rozrzut między epizodami ponad 50× większy (46 → 2459).

## 2026-10-10 — Ktore ustawienie Zoo robi roznice? (HalfCheetah-v5, 1M, seed 0)
| Wariant            | Udane (pkt) | Przewrotki | Dlaczego |
|--------------------|-------------|------------|----------|
| domyslne (mamy)    | 1585        | 0%         | —        |
| tylko normalizacja | 4000          | 10%        | bo liczby sa znormalizowane wiec akcje sa bardziej lagodne i plynne i ostrozniejsze wiec przewrotek jest mniej i w efekcie wolniej ale dojdziemy do lepszego wyniku        |
| tylko sigma 0,14   | 3000          | 15%         | mniej losuje wiec rzadziej wypadnie mu akcja w kierunku przewrotki ale moze miec problemy z dojsciem do celu       |
| pelne Zoo (mamy)   | 6465        | 30%        | —        |
Wynik: ppo_norm1_logstd0.0_seed0: srednia 1729.6, porazki 0%, udane 1729.6 (20.7 min) ppo_norm0_logstd-2.0_seed0: srednia 1094.0, porazki 42%, udane 1384.9 (19.9 min)
Wniosek:pojedyncze parametry nic nie daja trzeba sprobowac ich razem

Wynik (50 epizodow, te same starty):
| Wariant            | Udane | Przewrotki | Hipoteza   |
|--------------------|-------|------------|------------|
| domyslne           | 1585  | 0%         | —          |
| tylko normalizacja | 1730  | 0%         | 4000 / 10% |
| tylko sigma 0,14   | 1385  | 42%        | 3000 / 15% |
| pelne Zoo          | 6465  | 30%        | —          |
Hipoteza: obalona (wyniki duzo nizsze), ale kolejnosc trafiona.
Wniosek (wlasnymi slowami): ...

### Hipoteza: normalizacja + sigma 0,14 razem
Udane: 2500  Przewrotki: 10%   Dlaczego: normalizacja ulatwi prace gamma przez to wyniki beda lepsze

### Wynik: normalizacja + sigma 0,14
| Wariant               | Udane | Slabe (<1000) | Co widac                      |
|-----------------------|-------|---------------|-------------------------------|
| domyslne              | 1585  | 0%            | biega, szarpie                |
| tylko normalizacja    | 1730  | 0%            | prawie jak domyslne           |
| tylko sigma 0,14      | 1385  | 42%           | chodzi, bez przewrotek        |
| normalizacja + sigma  | 4567  | 0%            | biega                         |
| pelne Zoo             | 6465  | 30%           | najszybszy, przewrotki        |
Hipoteza (2500 / 10%): kierunek trafiony (interakcja), wielkosc niedoszacowana.
Wniosek (wlasnymi slowami): razem te dwa parametry dzialaja bardzo dobrze, nie ma sensu zmieniac innych

## 2026-10-12 — Prawdziwy licznik przewrotek (cos(kat) < 0 przez 1 s)
Stara miara: "porazka" = wynik < 1000.
| Model              | Stara miara | Przewrotki (hipoteza) | Przewrotki (wynik) |
|--------------------|-------------|-----------------------|--------------------|
| domyslne           | 0%          | brak                     |                    |
| tylko sigma 0,14   | 42%         | wolnoo chodzi                     |                    |
| normalizacja+sigma | 0%          | brak, dobrze chodzi                   |                    |
| pelne Zoo          | 30%         | przewrotki                     |                    |

### Sprawdzian nie przeszedl: domyslny 100% przewrotek, Zoo 0%
Moja hipoteza: winny jest B (nasz pewnik) — bo czasami jak ogladalem zachowanie geparda to wydawalo mi sie nietypowe a wyniki bylo ok
Domyslny gepard, kat na koncu epizodu: ok. 0 stopni
Slaby bieg Zoo, kat na koncu epizodu: ok. 90 stopni

Wynik: winny B. Domyslny gepard biega NA PLECACH (kat ~190°, 89–99% krokow do gory nogami).
Slabe biegi Zoo: nie na plecach, tylko utkniety przechylony ~50° (max 57°), ok. 570 pkt.
Wrapper mierzyl dobrze. Bledny byl nasz "pewnik". Maly rozrzut ≠ dobre zachowanie.

## 2026-10-13 — Kara za przechylenie tulowia (domyslne PPO, 1M, seed 0)
Kara: weight * (cos(kat) - 1) → 0 poziomo, -1 przy 90°, -2 na plecach
| Wariant        | Srednia (pkt) | Na plecach | Wolne (<1000) |
|----------------|---------------|------------|---------------|
| bez kary (0)   | 1500          | 3000          | 600             |
| z kara (1)     |     1200         | 400      | 600           |
Dlaczego: teraz epizody na plecach beda znacznie nizsze bo beda karane ale srednia mocno nie spadnie bo model nausczy sie w koncu prawidlowo chodzic

Wynik: bez kary 1585, 100% na plecach (kontrola = wynik z tyg. 5 co do przecinka).
Z kara 2764, 0% na plecach, 0 wolnych.
Hipoteza: mechanizm trafiony, kierunek sredniej odwrotny (przewidzialem spadek, jest +74%).
Wniosek: kara wypchnela geparda z pulapki (lokalne optimum). Bieg na nogach jest lepszy
nawet wedlug oryginalnej nagrody, tylko PPO samo do niego nie dochodzilo.

Weryfikacja: na podgladzie wydawalo sie "brzuch do gory", ale debug_angle:
kat -38..+70°, koniec ~0°, 0% krokow do gory nogami → biega na nogach.
Mocno kolysze tulowiem (do ~70°, Zoo tylko ~20°) → to pewnie "troche szarpie".
Sama normalizacja (tydz. 5): 1730 pkt, 100% na plecach.

## 2026-10-14 — Pomiar szarpania (10 epizodow, te same starty)
| Model                  | Kolysanie (st.) | Szarpanie (0-2) | Saturacja (%) |
|------------------------|-----------------|-----------------|---------------|
| bez kary (na plecach)  |       mocno          |       1          |     60%         |
| z kara (na nogach)     |       średnio          |       0          |     5%          |
| normalizacja + sigma   |          mało       |         1        |     10%          |
| pelne Zoo              |         dużo        |         2        |     70%          |
Hipoteza: najbardziej szarpie pelne zoo, najmniej z kara, bo pelne zoo idzie po "trupach do celu"

## 2026-10-14 — Kara za szarpanie (upright 1.0 + smooth, 1M, seed 0)
Kara: smooth * suma((a_t - a_{t-1})^2) po 6 stawach
| smooth    | Punkty | Szarpanie | Saturacja | Kolysanie | Na plecach |
|-----------|--------|-----------|-----------|-----------|------------|
| 0 (mamy)  | 2724   | 0.853     | 44%       | 16.7      | 0%         |
| 0.1       | 3000      | 0.5         | 30%         | 10         | 0%          |
| 0.3       | 4000     | 0.2         | 50%         | 5         | 0%          |
Dlaczego: im bardziej wygladzimy z pomoca smooth tym szarpanie sie zmniejszyi inne parametry tez. przy czym wydaje mi sie ze jak za duzo wygladzimy to saturacja moze zaczac wzrastac.
### Test mechanizmu: czy kara za szarpanie zabija ciekawosc?
Prognoza: sigma (train/std) spadnie najszybciej przy smooth = 0.3

Wynik: kara za szarpanie dziala na szarpanie (0.853 → 0.575 → 0.459), saturacja 44 → 32 → 14%,
ale punkty spadly (2764 → 1383 / 1728), a przy 0.1 wrocilo lezenie na plecach (44%).
Test mechanizmu (train/std): przy 0.3 sigma spadla duzo szybciej (0.13 na koncu, kontrola 0.22) → kara zgasila ciekawosc.
Przy 0.1 sigma prawie jak w kontroli, a mimo to 44% na plecach → niewyjasnione, moze przypadek seeda.
Wniosek: najplynniejszy i najszybszy gepard to normalizacja + mala sigma BEZ kar. Lecz przyczyne, nie objaw.