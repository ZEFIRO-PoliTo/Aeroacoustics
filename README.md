# Aeroacoustics

Dipartimento di Aerodinamica & Aeroacustica del team ZEFIRO.
Domenico Inchingolo 
Luca De Bonis

# BPM Python

Traduzione Python del modello semi-empirico Broadband BPM per la predizione del rumore aerodinamico a larga banda generato da eliche, rotori e turbine.

Il codice implementa le equazioni BPM sviluppate da Brooks, Pope e Marcolini nel report NASA RP-1218 e prende come riferimento l'implementazione Julia del BYU FLOW Lab, oggi pubblicata come [`byuflowlab/BroadbandBPM.jl`](https://github.com/byuflowlab/BroadbandBPM.jl). La versione originale BYU descrive il pacchetto come un modello semi-empirico per l'acustica di propulsori e turbine, basato sulle equazioni di Brooks, Pope e Marcolini.

## Obiettivo

Il progetto calcola lo spettro del livello di pressione sonora, `SPL`, e il livello globale `OASPL` per una pala discretizzata radialmente. La geometria della pala viene letta da file CSV e il calcolo integra i contributi acustici sulle sezioni radiali, sulle pale e su piu posizioni azimutali.

Gli output principali sono:

- `spl`: spettro in bande di terzo d'ottava, da 100 Hz a 40 kHz;
- `oaspl`: Overall Sound Pressure Level ottenuto sommando energeticamente le bande;
- grafici dello spettro per i casi di esempio.

## Teoria

Il modello BPM e un modello aeroacustico semi-empirico. Non risolve direttamente il campo di moto come farebbe una simulazione CFD completa, ma usa correlazioni sperimentali per stimare il rumore generato dagli strati limite e dalle regioni caratteristiche della pala.

Le sorgenti di rumore considerate sono:

- rumore da vortici nello strato limite laminare;
- rumore da strato limite turbolento al bordo d'uscita;
- rumore da separazione e stallo;
- rumore da bordo d'uscita con spessore finito;
- rumore da vortice di estremita della pala.

Per ogni elemento radiale della pala il codice calcola grandezze locali come velocita relativa, numero di Mach, numero di Reynolds, spessori di strato limite e numeri di Strouhal. Le funzioni empiriche del modello trasformano queste grandezze in contributi spettrali in decibel. I contributi vengono accumulati in energia acustica, cioe in termini di `p^2 / p_ref^2`, e solo alla fine riconvertiti in dB.

La posizione dell'osservatore entra nelle funzioni di direttivita, che modificano il livello sonoro in base alla geometria sorgente-osservatore. Il codice puo inoltre applicare la pesatura A, utile quando si vuole esprimere lo spettro in dBA, cioe con una correzione legata alla sensibilita dell'orecchio umano.

## Origine del codice

Questa repo contiene una traduzione in Python/Numpy del codice Julia del BYU FLOW Lab:

- repository originale: [`byuflowlab/BroadbandBPM.jl`](https://github.com/byuflowlab/BroadbandBPM.jl);
- nome precedente del progetto BYU: `BPM.jl`;
- linguaggio originale: Julia;
- linguaggio di questa repo: Python.

Nel README originale BYU sono riportate anche le tappe storiche del codice: sviluppo iniziale al FLOW Lab, traduzione in Julia, refactoring, riscrittura con casi di verifica e rinomina in `BroadbandBPM.jl`.

Questa versione mantiene la logica del modello BPM, ma la espone con funzioni Python in `src/bpm_equations.py` e utility numeriche in `src/utils.py`.

## Struttura

```text
BPM/
+-- main.py
+-- src/
|   +-- bpm_equations.py
|   +-- utils.py
+-- examples/
|   +-- apc14x85_cruise.py
|   +-- apc14x85_hover.py
|   +-- BPM_geometry_cruise.csv
|   +-- BPM_geometry_hover.csv
```

## Installazione

Creare un ambiente Python e installare le dipendenze principali:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy matplotlib pytest
```

Su Linux/macOS l'attivazione dell'ambiente e:

```bash
source .venv/bin/activate
```

## Uso rapido

Eseguire il caso principale:

```bash
python main.py
```

Eseguire gli esempi con generazione dello spettro:

```bash
python -m examples.apc14x85_cruise
python -m examples.apc14x85_hover
```

Gli esempi usano una pala APC 14x8.5E a due pale e leggono la geometria dai file CSV:

- `BPM_geometry_cruise.csv`, per il caso in crociera;
- `BPM_geometry_hover.csv`, per il caso in hovering.

## API principale

La funzione centrale e `sound_pressure_levels`:

```python
from src import bpm_equations

oaspl, spl = bpm_equations.sound_pressure_levels(
    ox, oy, oz,
    V, omega, B,
    r, c, c1, h, alpha, psi,
    nu, c0,
    laminar, turbulent, blunt, tip,
    trip, round_tip, weighted,
    nbeta,
    bpm_equations.DEFAULT_F,
    bpm_equations.DEFAULT_AD_B,
    smooth=True,
)
```

Dove:

- `ox, oy, oz` sono le coordinate dell'osservatore;
- `V` e la velocita di avanzamento;
- `omega` e la velocita angolare in rad/s;
- `B` e il numero di pale;
- `r, c, c1, h, alpha, psi` descrivono la geometria e l'assetto aerodinamico delle sezioni di pala;
- `nu` e la viscosita cinematica dell'aria;
- `c0` e la velocita del suono;
- `laminar`, `turbulent`, `blunt`, `tip` attivano o disattivano i meccanismi di rumore;
- `weighted=True` applica la pesatura A;
- `nbeta` imposta il numero di posizioni azimutali considerate;
- `smooth=True` usa transizioni smussate per rendere continue alcune relazioni empiriche.


## Riferimenti

- Brooks, T. F., Pope, D. S., and Marcolini, M. A., *Airfoil Self-Noise and Prediction*, NASA Reference Publication 1218, 1989.
- Brooks, T. F., and Marcolini, M. A., "Airfoil Tip Vortex Formation Noise", *AIAA Journal*, 1986.
- Vargas, L., *Wind Turbine Noise Prediction*, Master's Thesis, Technical University of Lisbon, 2008.
- BYU FLOW Lab, [`BroadbandBPM.jl`](https://github.com/byuflowlab/BroadbandBPM.jl).
- BYU FLOW Lab code list, [`BroadbandBPM.jl`](https://flow.byu.edu/codes/).

# Hanson model


