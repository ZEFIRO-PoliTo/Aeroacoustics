'''Modificare i dati in input e poi runnare il codice'''
def main():
    print("=== Avvio Simulazione Aeroacustica BPM ===")

    # ---------------------------------------------------------
    # 1. PARAMETRI AMBIENTALI E CINEMATICI
    # ---------------------------------------------------------
    nu = 1.48e-5            # Viscosità cinematica dell'aria (m^2/s)
    c0 = 343.0              # Velocità del suono (m/s)

    # Posizione dell'osservatore (Microfono)
    ox = 0.0               # asse del vento
    oy = 0.0             # distanza laterale/verticale (m)
    oz = 0.0                # asse verticale

    # ---------------------------------------------------------
    # 2. GEOMETRIA DELLA PALA E PROFILO AERODINAMICO
    # ---------------------------------------------------------
    B = 0                   # Numero di pale
    V = 0.0                 # Velocità del vento in ingresso (m/s)
    omega = 0.0 * (2 * np.pi / 60.0)            # Velocità di rotazione del rotore (rad/s)
    
    # delimiter=',' dice a Python che i dati sono separati da virgola
    # skiprows=1 fa saltare la prima riga (se hai scritto i nomi delle colonne)
    dati = np.loadtxt('BPM_geometry_.csv', delimiter=',', skiprows=1)
    # Estrai le colonne (l'indice parte da 0)
    r = dati[:, 0]             # Vettore delle posizioni radiali delle sezioni della pala (m)
    c = dati[:, 1]                    # Lunghezza della corda per ciascuna sezione radiale (m)
    c1 = dati[:, 2]                   # Distanza dall'asse di rotazione (pitch axis) al bordo d'attacco per ogni sezione (m)
    psi = dati[:, 3]                  # Angolo di cuneo (wedge angle) del bordo d'uscita per ogni sezione (gradi)
    h = dati[:, 4]                        # Spessore del bordo d'uscita (trailing edge) per ogni sezione (m)
    alpha = dati[:, 5]                # Angolo d'attacco aerodinamico per ogni sezione (gradi)
    

    laminar = False         # Flag per calcolare il rumore da strato limite laminare.
    turbulent = True        # Flag per calcolare il rumore da strato limite turbolento.
    blunt = True            # Flag per calcolare il rumore da spessore del bordo d'uscita.
    tip = True             # Flag per calcolare il rumore da vortice di estremità (tip noise).
    trip = False            # Flag per forzare la transizione (tripped) dello strato limite.
    round = True            # Indica se la punta della pala è arrotondata (True) o squadrata (False).
    weighted = True        # Indica se applicare la curva di pesatura A (A-weighting) ai decibel.
    nbeta = 36              # Numero di posizioni azimutali (angoli di rotazione) da simulare.
    f = DEFAULT_F           # Frequenze da usare nell'analisi
    AdB = DEFAULT_AD_B      # Frequenze pesate con il metodo A-Weighted
    smooth = True          # Attiva lo smoothing per rendere continue e differenziabili le equazioni empiriche (utile per l'ottimizzazione tramite gradienti).

    # ---------------------------------------------------------
    # 4. ESECUZIONE DELLA SIMULAZIONE
    # ---------------------------------------------------------

    oaspl, spl = sound_pressure_levels(ox, oy, oz, V, omega, B, r, c, c1, h, alpha, psi, nu, c0, 
                          laminar, turbulent, blunt, tip, trip, round, weighted, nbeta, f, AdB, smooth)
  
    print(spl, oaspl)
