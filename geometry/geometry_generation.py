import os
import pandas as pd
import numpy as np


FILE_INPUT = "14x85E-PERF copia.txt" 
FILE_OUTPUT = "Geometria_BPM_" + FILE_INPUT.replace(".txt", ".csv").replace(".dat", ".csv")

TRANSITION_START = 1.82  # inizio della transizione del profilo
TRANSITION_END = 5.12    # fine della transizione del profilo

#inserisci i 2 profili presenti
PROFILO_RADICE = "E63"
PROFILO_PUNTA = "NACA4412"


def calcola_geometria_profilo(nome_profilo, tc, chord_in):
    """
    NOTA BENE 

    Inserisci qui le formule aerodinamiche per Wedge Angle (phi) e TE Thickness (h_in)
    basate sullo spessore relativo (t/c) e la corda. In generale la formula è WedgeAngle=2⋅arctan(K⋅t/c).
    K è la Costante di Pendenza del Bordo d'Uscita, è un numero che dipende unicamente dall'equazione 
    matematica che il progettista ha inventato per disegnare quel profilo. Non è altro che la derivata al TE del
    polinomio che genera il profilo.
    """
    if nome_profilo == "NACA4412" or nome_profilo == "NACA4-digit":
        phi = 2 * np.degrees(np.arctan(1.169 * tc))
        h_in = 0.021 * tc * chord_in
        return phi, h_in
        
    elif nome_profilo == "E63":
        phi = 2 * np.degrees(np.arctan(0.71 * tc)) 
        h_in = 0.005 * tc * chord_in 
        return phi, h_in
        
    elif nome_profilo == "CLARK_Y":
        # Valori empirici approssimati per la famiglia Clark Y
        phi = 2 * np.degrees(np.arctan(1.25 * tc))
        h_in = 0.015 * tc * chord_in
        return phi, h_in
        
    else:
        raise ValueError(f"ERRORE: Il profilo '{nome_profilo}' non esiste nella libreria!")

def elabora_file_apc():
    if not os.path.exists(FILE_INPUT):
        print(f"ERRORE: Il file '{FILE_INPUT}' non è stato trovato!")
        return

    with open(FILE_INPUT, 'r') as f:
        lines = f.readlines()

    R_in = None
    start_idx = -1
    
    for i, line in enumerate(lines):
        if "RADIUS:" in line:
            R_in = float(line.split()[1])
        if "STATION" in line and "CHORD" in line and "PITCH" in line:
            start_idx = i + 1

    if R_in is None or start_idx == -1:
        print("ERRORE: Impossibile trovare Raggio o inizio tabella nel file.")
        return

    data = []
    for line in lines[start_idx:]:
        if line.strip() == "": continue
        if "RADIUS:" in line or "TOTAL WEIGHT" in line: break
            
        parts = line.split()
        if len(parts) >= 11:
            try:
                data.append({
                    'r_in': float(parts[0]),
                    'chord_in': float(parts[1]),
                    'max_thick_in': float(parts[9])
                })
            except ValueError:
                pass 

    df = pd.DataFrame(data)
    if df.empty: return

    IN_TO_M = 0.0254
    df['r/R'] = df['r_in']*0.0254
    df['t/c'] = df['max_thick_in'] / df['chord_in']

    wedge_angles = []
    te_thicknesses_m = []
    c1_m = []
    chord_m = []

    for _, row in df.iterrows():
        r_in = row['r_in']
        tc = row['t/c']
        c_in = row['chord_in']
        
        c_m_val = c_in * IN_TO_M
        chord_m.append(c_m_val)
        c1_m.append(0.25 * c_m_val) # Asse di beccheggio standard al 25%
        
        # Estrae le formule dalla Libreria in base alle tue scelte
        phi_radice, h_radice_in = calcola_geometria_profilo(PROFILO_RADICE, tc, c_in)
        phi_punta, h_punta_in = calcola_geometria_profilo(PROFILO_PUNTA, tc, c_in)
        
        if r_in <= TRANSITION_START:
            phi, h_in = phi_radice, h_radice_in
        elif r_in >= TRANSITION_END:
            phi, h_in = phi_punta, h_punta_in
        else:
            w = (r_in - TRANSITION_START) / (TRANSITION_END - TRANSITION_START)
            phi = (1 - w) * phi_radice + w * phi_punta
            h_in = (1 - w) * h_radice_in + w * h_punta_in
            
        wedge_angles.append(phi)
        te_thicknesses_m.append(h_in * IN_TO_M)

    df['c_m'] = chord_m
    df['c1_m'] = c1_m
    df['psi_deg'] = wedge_angles
    df['h_m'] = te_thicknesses_m

    output = df[['r/R', 'c_m', 'c1_m', 'psi_deg', 'h_m']].copy()
    output.to_csv(FILE_OUTPUT, index=False, float_format='%.7f')
    
    print("=== ELABORAZIONE COMPLETATA ===")
    print(f"File Output: {FILE_OUTPUT}")
    print(f"Transizione: da {TRANSITION_START}\" a {TRANSITION_END}\"")
    print(f"Profili Usati: {PROFILO_RADICE} -> {PROFILO_PUNTA}")

if __name__ == "__main__":
    elabora_file_apc()