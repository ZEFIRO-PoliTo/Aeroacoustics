"""
Esempio: APC 14x8.5E in volo di crociera
=========================================
Simula il rumore dell'elica APC 14x8.5E a due pale in condizione di crociera.

Condizioni operative:
  - V     = 20 m/s   (velocità di avanzamento)
  - RPM   = 5000
  - Numeri di Reynolds relativamente bassi
  - Tutti i meccanismi di rumore attivi, tranne ''turbulent''


Come eseguire (dalla radice del progetto BPM/):
    python -m examples.apc14x85_cruise
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
# CONDIZIONI OPERATIVE: APC 14x8.5E in crociera
# ------------------------------------------------------------------
B     = 2                                    # numero di pale
V     = 20.0                                 # velocità vento (m/s)
omega = 5000.0 * (2.0 * np.pi / 60.0)       # 5000 RPM → rad/s

# Posizione microfono (osservatore)
ox, oy, oz = 5.0, 5.0, 0.0

# ------------------------------------------------------------------
# GEOMETRIA DELLA PALA  (caricata dal CSV nella cartella BPM/)
# ------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "..", "BPM_geometry_cruise.csv")
data  = np.loadtxt(csv_path, delimiter=",", skiprows=1)

r     = data[:, 0]   # posizioni radiali (m)
c     = data[:, 1]   # corde (m)
c1    = data[:, 2]   # distanza pitch-axis → LE (m)
psi   = data[:, 3]   # wedge angle (deg)
h     = data[:, 4]   # spessore TE (m)
alpha = data[:, 5]   # angolo d'attacco (deg)

# ------------------------------------------------------------------
# FLAG DEI MECCANISMI DI RUMORE
# ------------------------------------------------------------------
laminar   = True    # LBL-VS: rumore da instabilità strato laminare
turbulent = False    # TBL-TE: rumore da turbolenza strato limite
blunt     = True    # TE bluntness: rumore da bordo d'uscita smussato
tip       = True    # Tip vortex: rumore da vortice di estremità
trip      = False   # nessun forzamento della transizione
round_tip = False   # punta squadrata (APC standard)
weighted  = True    # applica curva A-weighting
nbeta     = 36      # posizioni azimutali

f   = bpm_equations.DEFAULT_F
AdB = bpm_equations.DEFAULT_AD_B

# ------------------------------------------------------------------
# SIMULAZIONE
# ------------------------------------------------------------------
print("=== APC 14x8.5E — Condizione di CROCIERA ===")
print(f"  V = {V} m/s  |  RPM = 5000  |  B = {B}")
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
ax.semilogx(f, spl, "b-o", markersize=4, linewidth=1.5, label=f"OASPL = {oaspl:.1f} dBA")
ax.set_xlabel("Frequenza (Hz)")
ax.set_ylabel("SPL (dBA)")
ax.set_title("APC 14x8.5E — Spettro in 1/3 d'ottava — Crociera (V=20 m/s, 5000 RPM)")
ax.set_xticks([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 40000])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "apc14x85_cruise_spectrum.png"), dpi=150)
print("\n  Grafico salvato in examples/apc14x85_cruise_spectrum.png")
plt.show()
