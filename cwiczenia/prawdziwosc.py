"""Ćwiczenie: co if uznaje za prawdę? Tekst: tylko czy jest pusty. Liczba: tylko czy jest zerem."""

for x in ["False", "", "0", " ", "  ", "None"]:
    print(f"tekst  {x!r:8} długość {len(x)}  →  {bool(x)}")
for x in [0, 0.0, 5, -1]:
    print(f"liczba {x!r:8}             →  {bool(x)}")
