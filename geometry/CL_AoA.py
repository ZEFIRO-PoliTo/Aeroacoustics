import numpy as np
import csv
import os

def carica_e_pulisci_polare(nome_file):
    """
    il codice qui calcola alfa zero lift da sommare agli angoli geometrici delle polari
    """
    dati = np.loadtxt(nome_file)
    alpha = dati[:, 0]
    cl = dati[:, 1]
    
    idx_min = np.argmin(cl)
    idx_max = np.argmax(cl)
    
    cl_pulito = cl[idx_min:idx_max+1]
    alpha_pulito = alpha[idx_min:idx_max+1]
    

    alpha_L0 = np.interp(0.0, cl_pulito, alpha_pulito)
    
    # Restituiamo le curve e l'angolo di zero-lift
    return cl_pulito, alpha_pulito, alpha_L0

def calcola_alpha(r_su_R, R_totale_pollici, cl_target, polare_e63, polare_4412):
    """
    ilcodice calcola l'angolo d'attacco geometrico e assoluto.
    """
    r_pollici = r_su_R * R_totale_pollici
    
    # Scompattiamo i dati completi
    cl_e63, alpha_e63_curva, alpha_L0_e63 = polare_e63
    cl_4412, alpha_4412_curva, alpha_L0_4412 = polare_4412
    
    # 1. Calcolo degli angoli geometrici locali
    alpha_geom_e63 = np.interp(cl_target, cl_e63, alpha_e63_curva)
    alpha_geom_4412 = np.interp(cl_target, cl_4412, alpha_4412_curva)
    
    # 2. Calcolo degli angoli assoluti locali (Formula: geom - zero_lift)
    alpha_abs_e63 = alpha_geom_e63 - alpha_L0_e63
    alpha_abs_4412 = alpha_geom_4412 - alpha_L0_4412
    
    # 3. Logica di Blending (Miscelazione lungo la pala)
    if r_pollici <= 1.83:
        return alpha_geom_e63, alpha_abs_e63
        
    elif r_pollici >= 5.12:
        return alpha_geom_4412, alpha_abs_4412
        
    else:
        peso_4412 = (r_pollici - 1.83) / (5.12 - 1.83)
        peso_e63 = 1.0 - peso_4412
        
        alpha_geom_fuso = (alpha_geom_e63 * peso_e63) + (alpha_geom_4412 * peso_4412)
        alpha_abs_fuso = (alpha_abs_e63 * peso_e63) + (alpha_abs_4412 * peso_4412)
        
        return alpha_geom_fuso, alpha_abs_fuso

def elabora_file_csv(file_input, file_output, R_elica, polare_e63, polare_4412):
    if not os.path.exists(file_input):
        print(f"Errore: Il file {file_input} non esiste.")
        return

    risultati = [] 
    
    with open(file_input, mode='r', encoding='utf-8-sig') as f_in:
        reader = csv.reader(f_in, delimiter=';')
        
        for riga in reader:
            if not riga:
                continue
            try:
                r_su_R = float(riga[0])
                cl_target = float(riga[1])
                
                # Ora la funzione restituisce due valori
                alpha_geom, alpha_abs = calcola_alpha(r_su_R, R_elica, cl_target, polare_e63, polare_4412)
                
                risultati.append([r_su_R, cl_target, alpha_geom, alpha_abs])
                
            except (ValueError, IndexError):
                print(f"Saltata riga: {riga}")
                continue

    with open(file_output, mode='w', newline='') as f_out:
        writer = csv.writer(f_out)
        # Aggiornato l'header per riflettere i nuovi dati
        writer.writerow(['r/R', 'Cl', 'Alpha_Geom_deg', 'Alpha_Abs_deg'])
        for res in risultati:
            writer.writerow([f"{res[0]:.4f}", f"{res[1]:.4f}", f"{res[2]:.4f}", f"{res[3]:.4f}"])
            
    print(f"\nElaborazione completata. Generati {len(risultati)} risultati.")
    print(f"I dati sono stati salvati in: {file_output}")



if __name__ == "__main__":
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    
    file_polare_e63 = os.path.join(cartella_script, 'E63.txt')
    file_polare_4412 = os.path.join(cartella_script, '4412.txt')
    file_input_csv = os.path.join(cartella_script, '14x8.5E_9000RPM_0.0J_CL_CD .csv')
    file_output_csv = os.path.join(cartella_script, 'risultati_alpha_14x8.5E_9000RPM_0.0J_CL_CD.csv')
    
    R_elica = 7.0 
    
    print("Caricamento polari in corso...")
    dati_e63 = carica_e_pulisci_polare(file_polare_e63)
    dati_4412 = carica_e_pulisci_polare(file_polare_4412)
    
    # Stampiamo gli Alpha_L0 per tua conferma visiva
    print(f"-> Eppler E63 Alpha_{'{L=0}'} stimato: {dati_e63[2]:.3f}°")
    print(f"-> NACA 4412 Alpha_{'{L=0}'} stimato: {dati_4412[2]:.3f}°")
    
    print(f"\nLettura del file {os.path.basename(file_input_csv)} in corso...")
    elabora_file_csv(
        file_input=file_input_csv, 
        file_output=file_output_csv, 
        R_elica=R_elica, 
        polare_e63=dati_e63, 
        polare_4412=dati_4412
    )
