from pathlib import Path

import mujoco
import numpy as np
import pytest

import rl_journey

SCENE = Path(rl_journey.__file__).parent / "assets" / "so101" / "scene.xml"


def wczytaj():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    return model, mujoco.MjData(model)


def test_tyle_silownikow_ile_serw():
    model, _ = wczytaj()
    assert model.nu == 6
    # ↑ ile serw ma jedno ramię SO-101 (follower)? plan nauki: „follower 6 × STS3215”


def test_sterowanie_to_kat_docelowy():
    model, _ = wczytaj()
    j = model.actuator_trnid[0, 0]  # staw, którym rusza siłownik 0
    assert model.actuator_ctrlrange[0] == pytest.approx(model.jnt_range[j], abs=1e-3)
    # ↑ siłownik pozycyjny dostaje kąt, więc zakres sterowania = zakres kąta stawu.
    #   explore_so101.py, sekcja STAWY: z jakiej tablicy model bierze zakres stawu?


def test_obrot_podstawy_przesuwa_chwytak_po_okregu():
    model, data = wczytaj()
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    mujoco.mj_forward(model, data)
    anchor, axis = data.xanchor[0].copy(), data.xaxis[0].copy()

    def wysokosc_promien(p):
        r = p - anchor
        h = r @ axis
        return h, np.linalg.norm(r - h * axis)

    h0, r0 = wysokosc_promien(data.site_xpos[tip].copy())
    data.qpos[model.jnt_qposadr[0]] = np.pi / 2  # podstawa o 90°
    mujoco.mj_forward(model, data)
    h90, r90 = wysokosc_promien(data.site_xpos[tip].copy())
    assert h90 == pytest.approx(h0, abs=1e-6)
    assert r90 == pytest.approx(r0, abs=1e-6)
    # ↑ zmienna z TEGO testu: co nie może się zmienić przy obrocie wokół osi? (Twoja H4)
