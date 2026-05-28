"""
Esempio: APC 14x8.5E in hovering
==================================
Simula il rumore dell'elica APC 14x8.5E a due pale in condizione di hovering
(velocità di avanzamento nulla, solo rotazione).

Condizioni operative:
  - V     = 0 m/s   (nessuna velocità di avanzamento)
  - RPM   = 9000
  - Tutti i meccanismi del rumore attivi

Come eseguire (dalla radice del progetto BPM/):
    python -m examples.apc14x85_hover
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src import bpm_equations

# ------------------------------------------------------------------
# PARAMETRI FISICI
# ------------------------------------------------------------------
nu = 1.48e-5        # Viscosità cinematica aria (m²/s)
c0 = 343.0          # Velocità del suono (m/s)

# ------------------------------------------------------------------
# CONDIZIONI OPERATIVE: APC 14x8.5E in hovering
# ------------------------------------------------------------------
B     = 2                                    # numero di pale
V     = 0.0                                  # hovering: nessun vento in ingresso
omega = 9000.0 * (2.0 * np.pi / 60.0)       # 3600 RPM → rad/s

# Posizione microfono (osservatore)
ox, oy, oz = 5.0, 2.0, 0.0

# ------------------------------------------------------------------
# GEOMETRIA DELLA PALA  (caricata dal CSV nella cartella BPM/)
# ------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "..", "BPM_geometry_hover.csv")
data  = np.loadtxt(csv_path, delimiter=",", skiprows=1)

r     = data[:, 0]
c     = data[:, 1]
c1    = data[:, 2]
psi   = data[:, 3]
h     = data[:, 4]
alpha = data[:, 5]

# ------------------------------------------------------------------
# FLAG DEI MECCANISMI DI RUMORE
# ------------------------------------------------------------------
laminar   = True
turbulent = True
blunt     = True
tip       = True
trip      = False
round_tip = False
weighted  = True
nbeta     = 36

f   = bpm_equations.DEFAULT_F
AdB = bpm_equations.DEFAULT_AD_B

# ------------------------------------------------------------------
# SIMULAZIONE
# ------------------------------------------------------------------
print("=== APC 14x8.5E — Condizione di HOVERING ===")
print(f"  V = {V} m/s  |  RPM = 9000  |  B = {B}")
print()

oaspl, spl = bpm_equations.sound_pressure_levels(
    ox=ox, oy=oy, oz=oz,
    V=V, omega=omega, B=B,
    r=r, c=c, c1=c1, h=h, alpha=alpha, psi=psi,
    nu=nu, c0=c0,
    laminar=laminar, turbulent=turbulent, blunt=blunt,
    tip=tip, trip=trip, round=round_tip,
    weighted=weighted, nbeta=nbeta,
    f=f, AdB=AdB,
    smooth=True,
)

print(f"  OASPL = {oaspl:.2f} dBA")
print()
print(f"  {'Frequenza (Hz)':>16}  {'SPL (dBA)':>10}")
print("  " + "-" * 30)
for freq, level in zip(f, spl):
    print(f"  {freq:>16.0f}  {level:>10.2f}")

# ------------------------------------------------------------------
# GRAFICO
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(f, spl, "r-o", markersize=4, linewidth=1.5, label=f"OASPL = {oaspl:.1f} dBA")
ax.set_xlabel("Frequenza (Hz)")
ax.set_ylabel("SPL (dBA)")
ax.set_title("APC 14x8.5E — Spettro in 1/3 d'ottava — Hovering (V=0 m/s, 9000 RPM)")
ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "apc14x85_hover_spectrum.png"), dpi=150)
print("\n  Grafico salvato in examples/apc14x85_hover_spectrum.png")
plt.show()
