import pandas as pd
import numpy as np

def elabora_pala_vtk(file_input, file_output):
    # 1. Caricamento dati (saltando le prime 2 righe di intestazione del file VTK/CSV)
    try:
        # Usiamo engine='python' per gestire eventuali virgolette extra nel file
        df = pd.read_csv(file_input, skiprows=2, names=['alpha_geom'], quotechar='"')
    except Exception as e:
        print(f"Errore nella lettura del file: {e}")
        return

    # 2. Parametri della pala
    L_tot = 7.0         # Lunghezza totale in pollici
    x_trans_1 = 1.82    # Fine zona E63 pura
    x_trans_2 = 5.12    # Inizio zona 4412 pura
    
    # Valori assoluti degli angoli di zero-lift
    zl_E63 = abs(-2.1)  # 2.1°
    zl_4412 = abs(-1.0) # 1.0°

    # 3. Generazione delle stazioni (Assumendo distribuzione uniforme lungo i 7 pollici)
    n_punti = len(df)
    df['stazione_in'] = np.linspace(0, L_tot, n_punti)
    df['xi'] = df['stazione_in'] / L_tot # Stazione relativa (0 a 1)

    # 4. Funzione di calcolo dello zero lift con transizione lineare
    def calcola_zl(x):
        if x <= x_trans_1:
            return zl_E63
        elif x >= x_trans_2:
            return zl_4412
        else:
            # Calcolo dell'interpolazione lineare nella zona di transizione [1.82, 5.12]
            frazione = (x - x_trans_1) / (x_trans_2 - x_trans_1)
            return zl_E63 + frazione * (zl_4412 - zl_E63)

    # 5. Applicazione della logica
    df['zero_lift_da_sommare'] = df['stazione_in'].apply(calcola_zl)
    df['alpha_corretto'] = df['alpha_geom'] + df['zero_lift_da_sommare']

    # 6. Salvataggio e output
    df.to_csv(file_output, index=False)
    print(f"Processati {n_punti} punti.")
    print(f"File salvato come: {file_output}")
    
    # Mostriamo i primi e gli ultimi risultati per verifica
    print("\n--- Anteprima Risultati (Radice - E63) ---")
    print(df.head(5))
    print("\n--- Anteprima Risultati (Punta - 4412) ---")
    print(df.tail(5))

# Esecuzione
file_in = 'SCALARS-ThetaEffDeg-float.csv'
file_out = 'A_o_A.csv'
elabora_pala_vtk(file_in, file_out)