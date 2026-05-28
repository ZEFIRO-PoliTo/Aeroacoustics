import numpy as np
from .utils import trapz, quintic_blend, abs_smooth, aspect_ratio_correction, ksmin

"""
Frequenze standard in bande di terzo d'ottava (Hz). 

Sono frequenze standard che vanno da 100dB a 40000dB, da basse ad alte frequenze. 
L'orecchio umano percepisce le frequenze in modo logaritmico. 
Dividere lo spettro in queste bande permette di simulare come noi distinguiamo i vari toni
"""

DEFAULT_F = np.array([
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0,
    630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0,
    4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    25000.0, 31500.0, 40000.0
])
"""
Curva di ponderazione A (dBA)

Ad ogni valore della costante DEFAULT_AD_B corrisponde una  frequenza
presente nella costante DEFAULT_F, per permettere di al codice di 'abbassare' 
il volume calcolato perché quella frequenza è percepita meno dall'orecchio umano
"""
DEFAULT_AD_B = np.array([
    -19.145, -16.190, -13.244, -10.847, -8.675, -6.644, -4.774, -3.248,
    -1.908, -0.795, 0.0, 0.576, 0.993, 1.202, 1.271, 1.202,
    0.964, 0.556, -0.114, -1.144, -2.488, -4.250, -6.701, -9.341,
    -12.322, -15.694, -19.402
])


def sound_pressure_levels(ox, oy, oz, V, omega, B, r, c, c1, h, alpha, psi, nu, c0, 
                          laminar, turbulent, blunt, tip, trip, round, weighted, nbeta, f, AdB, smooth):

    #Input validation: se discretizzo la pala in 10 stazioni devo avere 10 valori in input!!!
    assert len(r) == len(c) == len(c1) == len(alpha) == len(psi), \
        "Errore: Gli array geometrici della pala (r, c, c1, alpha, psi) devono avere tutti la stessa lunghezza."
    assert not weighted or len(f) == len(AdB), \
        "Errore: Se la pesatura A è attiva (weighted=True), l'array delle frequenze 'f' deve avere la stessa lunghezza dell'array 'AdB'."

    #numero delle stazioni radiali
    nr = len(r)
    nf = len(f)

    #angolo d'attacco viene reso positivo dal momento che i test BPM sono stati eseguiti su profili NACA simmetrici
    #quindi strato limite che si genera a 5 o -5 gradi è lo stesso, quindi stesso rumore!!!
    if smooth:
        alpha = abs_smooth(alpha, 0.001)
    else:
        alpha = np.abs(alpha)

    #blade angles
    beta = np.linspace(0.0, 2 * np.pi/B, nbeta, endpoint=False)

    #calcolo AR
    area = trapz(r, c)
    cbar = area/(r[-1] - r[0])          #corda media
    aspect_ratio = r[-1]/cbar

    #Velocità al tip
    Vt = np.sqrt(V ** 2 + (omega * r[-1]) ** 2)

    #inizializzo gli array di spl e p2 = p^2/pref^2
    spl = np.zeros(nf)
    p2 = np.zeros(nf)

    for ibeta in range(nbeta):

        p2 = np.zeros(nf)

        for iB in range(B):

            cbeta = beta[ibeta] + iB * (2 * np.pi/B)

            if tip:
                # Trova la posizione dell'osservatore rispetto alla PUNTA della pala [-1]
                r_obs, theta_obs, phi_obs = observer_location(ox, oy, oz, c[-1], c1[-1], r[-1], cbeta)
                
                # Aggiunge il contributo del Tip Vortex
                p2 = tip_pressure(p2, f, r_obs, theta_obs, phi_obs, c[-1], alpha[-1], Vt, c0, aspect_ratio, round, smooth)

            # Loop su ciascuna sezione radiale della pala (i segmenti)
            for k in range(nr - 1):
                
                # Calcola le proprietà medie della sezione (Punto Medio)
                Lm = r[k+1] - r[k]                      # lunghezza segmento (m)
                rm = (r[k+1] + r[k]) / 2.0              # raggio segmento (m)
                cm = (c[k+1] + c[k]) / 2.0              # corda segmento (m)
                c1m = (c1[k+1] + c1[k]) / 2.0           # distanza bordo d'attacco (m)
                hm = (h[k+1] + h[k]) / 2.0              # spessore bordo d'uscita (m)
                am = (alpha[k+1] + alpha[k]) / 2.0      # angolo d'attacco (deg)
                pm = (psi[k+1] + psi[k]) / 2.0          # wedge angle (deg)
                Vm = np.sqrt(V ** 2 + (omega * rm) **2)   # velocità locale (m/s)

                # Posizione dell'osservatore rispetto a questo elemento della pala
                r_obs, theta_obs, phi_obs = observer_location(ox, oy, oz, cm, c1m, rm, cbeta)

                # Controlla se i flag sono Array o valori singoli
                is_tripped = trip[k] if isinstance(trip, (list, np.ndarray)) else trip
                is_laminar = laminar[k] if isinstance(laminar, (list, np.ndarray)) else laminar
                is_turbulent = turbulent[k] if isinstance(turbulent, (list, np.ndarray)) else turbulent
                is_blunt = blunt[k] if isinstance(blunt, (list, np.ndarray)) else blunt

                if is_laminar:
                    p2 = laminar_pressure(p2, f, r_obs, theta_obs, phi_obs, Lm, cm, am, Vm, c0, nu, smooth)

                if is_turbulent:
                    p2 = turbulent_pressure(p2, f, r_obs, theta_obs, phi_obs, Lm, cm, am, Vm, c0, nu, is_tripped, smooth)

                if is_blunt:
                    p2 = bluntness_pressure(p2, f, r_obs, theta_obs, phi_obs, Lm, cm, hm, am, pm, Vm, c0, nu, is_tripped, smooth)

        # Accumula l'energia per questa rotazione nello spettro totale
        spl += p2

    # --- POST PROCESSING ACUSTICO ---

    # Divide l'energia accumulata per il numero di scatti (Media temporale)
    spl = spl / nbeta

    # Calcola il Livello di Pressione Sonora (SPL) in Decibel: 10 * log10(p^2/pref^2)
    spl = 10.0 * np.log10(spl)

    # Applica l'A-weighting se richiesto
    if weighted:
        spl += AdB

    # Calcola il Livello di Pressione Sonora Globale (OASPL)
    # Somma l'energia di tutte le frequenze: p2sum = sum(10^(spl_i / 10))
    p2sum = np.sum(10.0 ** (spl / 10.0))
    oaspl = 10.0 * np.log10(p2sum)

    return oaspl, spl

def observer_location(xo, yo, zo, c, c1, d, beta):

    #Calcolo della posizione del Trailing Edge
    xs = np.sin(beta) * d - np.cos(beta) * (c - c1)
    zs = np.cos(beta) * d - np.sin(beta) * (c - c1)

    #Calcolo della posizione dell'osservatore rispetto al TE
    xe_d = xo - xs
    ze_d = zo - zs  

    #
    deg = np.pi - beta
    xe = np.cos(deg) * xe_d + np.sin(deg) * ze_d
    ze = -np.sin(deg) * xe_d + np.cos(deg) * ze_d

    #Calcola la distanza dall'osservatore e l'angolo di direttività
    r = np.sqrt(yo ** 2 + xe ** 2 + ze ** 2)
    theta = np.arctan2(np.sqrt(yo **2 + ze ** 2), xe)
    phi = np.arctan2(yo, ze)

    #Questa parte di codice è utile se phi è vicino a 0° o 180°,
    #in caso affermativo la funzione di direttività mi darevve valori nulli
    deg_to_rad = np.pi / 180.0
    rad_to_deg = 180.0 / np.pi
    if abs(phi) < 5.0 * deg_to_rad:
        sign = 1 if phi >= 0.0 else -1          #conservo il segno dell'angolo
        phi_er = abs(phi) * rad_to_deg
        phi_er = 0.1 * phi_er ** 2 + 2.5
        phi = sign * phi_er * deg_to_rad
    elif abs(phi) > 175.0 * deg_to_rad:
        sign = 1 if phi >= 0.0 else -1
        phi_er = abs(phi) * rad_to_deg
        phi_er = -0.1 * (phi_er - 180.0) ** 2 + 2.5
        phi = sign * phi_er * deg_to_rad
    return r, theta, phi   

#queste 4 funzioni mi ritornano p, un valore adimensionale che sarebbe il rapporto tra (p/pref)^2

def laminar_pressure(p, f, r, theta, phi, L, c, alpha, V, c0, nu, smooth):
    
    #Numero di Mach
    M  = V/c0

    #Numero  di Reynolds 
    Re = (V * c)/nu

    dp = boundary_thickness(Re, c, alpha)

    Dh = Dhfunc(M, theta, phi)

    St1p = St1p_func(Re, smooth)

    Stp_peak = St1p * 10.0 ** (-0.04 * alpha)

    Re0 = Re0_func(alpha, smooth)

    G2 = G2_func(Re/Re0, smooth)

    G3 = 171.04 - 3.03 * alpha

    #Funzione di scaling
    scale = 10.0 * np.log10((dp * M ** 5 * L * Dh)/r ** 2)

    #Calcolo dei pressure levels per ogni frequenza
    for i in range(len(f)):
        stp = f[i] * dp/V

        G1 = G1_func(stp/Stp_peak, smooth)

        spl_lam = G1 + G2 + G3 + scale

        p[i] += 10.0 ** (spl_lam/10.0)
    
    return p

def turbulent_pressure(p, f, r, theta, phi, L, c, alpha, V, c0, nu, trip, smooth,
                       pressure = True, suction = True, separation = False):

    '''
    I flag 'pressure', 'suction' e 'separation' permettono di isolare le singole sorgenti.
    '''

    #Numero di Mach
    M = V/c0

    #Numero di Reynolds
    Re = (V * c)/nu   

    dp, ds = displacement_thickness(Re, c, alpha, trip, smooth)

    Dl = Dlfunc(M, theta, phi)
    Dh = Dhfunc(M, theta, phi)

    #Numero di Re basato su spessore di spostamento
    Rp = V * dp/nu
    Rs = V * ds/nu

    St1 = 0.02 * M ** (-0.6)

    St2 = St2_func(alpha, St1, smooth)

    St1bar = (St1 + St2)/2.0

    a0 = a0_func(Re, smooth)
    Amin0 = Amin_func(a0, smooth)
    Amax0 = Amax_func(a0, smooth)
    A_ratio = (20.0 + Amin0)/(Amin0 - Amax0)

    ap0 = a0_func(3 * Re, smooth)
    Apmin0 = Amin_func(ap0, smooth)
    Apmax0 = Amax_func(ap0, smooth)
    Ap_ratio = (20.0 + Apmin0)/(Apmin0 - Apmax0)

    b0 = b0_func(Re, smooth)
    Bmin0 = Bmin_func(b0, smooth)
    Bmax0 = Bmax_func(b0, smooth)
    B_ratio = (20.0 + Bmin0)/(Bmin0 - Bmax0)

    K1 = K1_func(Re, smooth)

    delta_K1 =  delta_K1_func(Rp, alpha, smooth)

    K2 = K2_func(alpha, M, K1, smooth)
 
    gamma0 = 23.430 * M + 4.651

    #Angolo d'attacco per cui ho stallo
    if smooth:
        alpha_stall = ksmin([12.5, gamma0])
    else:
        alpha_stall = min(12.5, gamma0)

    #Calcolo dei pressure levels per ogni frequenza
    for i in range(len(f)):

        #Pressure side
        Stp = f[i] * dp/V
        ap = np.log10(Stp/St1)
        Apmin = Amin_func(ap, smooth)
        Apmax = Amax_func(ap, smooth)
        Ap = Apmin + A_ratio * (Apmax - Apmin)
        spl_p = 10.0 * np.log10((dp * M ** 5 * L * Dh)/r ** 2) + Ap + K1 -3 + delta_K1      #unstalled
        spl_p_stall = 10.0 * np.log10((dp * M ** 5 * L * Dl)/r ** 2)                        #stalled

        p_p_dx = 0.5
        p_p_f1 = 10.0 ** (spl_p/10.0)               #pressione sonora calcolata per un flusso unstalled
        p_p_f2 = 10.0 ** (spl_p_stall/10.0)         #pressione sonora calcolata per un flusso stallato

        if smooth:
            p_p = quintic_blend(p_p_f1, p_p_f2, alpha, alpha_stall, p_p_dx)
        else:
            p_p = p_p_f1 if alpha < alpha_stall else p_p_f2

        #Suction side
        Sts = f[i] * ds/V
        a_s = np.log10(Sts/St1bar)              #Riccardo se lo vedrai il codice Julia usava as per indicare il generico coeff a nella zona di depressione, io ho scritto a_s senno troppo casino con python
        Asmin = Amin_func(a_s, smooth)
        Asmax =  Amax_func(a_s, smooth)
        As = Asmin + A_ratio * (Asmax - Asmin)
        spl_s = 10.0 * np.log10((ds * M ** 5 * L * Dh)/r ** 2) +As + K1 - 3         #unstalled
        spl_s_stall = 10.0 * np.log10((ds * M ** 5 * L * Dl)/r ** 2)                #stalled

        p_s_dx = 0.5
        p_s_f1 = 10.0 ** (spl_s/10.0)
        p_s_f2 = 10.0 ** (spl_s_stall/10.0)

        if smooth:
            p_s = quintic_blend(p_s_f1, p_p_f2, alpha, alpha_stall, p_s_dx)
        else:
            p_s = p_s_f1 if alpha < alpha_stall else p_p_f2

        #Separazione
        b = np.log10(Sts/St2)
        Amin = Amin_func(b, smooth)
        Amax = Amax_func(b, smooth)
        Bmin = Bmin_func(b, smooth)
        Bmax = Bmax_func(b, smooth)
        Aa = Amin + Ap_ratio * (Amax - Amin)
        Ba = Bmin + B_ratio * (Bmax - Bmin)
        spl_a = 10.0 * np.log10((ds * M ** 5 * L * Dh)/r ** 2) + Ba + K2            #unstalled
        slp_a_stall = 10.0 * np.log10((ds * M ** 5 * L * Dl)/r ** 2) + Aa + K2      #stalled

        p_a_dx = 0.5
        p_a_f1 = 10.0 ** (spl_a/10.0)       
        p_a_f2 = 10.0 ** (slp_a_stall/10.0)

        if smooth:
            p_a = quintic_blend(p_a_f1, p_a_f2, alpha, alpha_stall, p_a_dx)
        else:
            p_a = p_a_f1 if alpha < alpha_stall else p_a_f2

        if pressure:
            p[i] += p_p
        if suction:
            p[i] += p_s
        if separation:
            p[i] += p_a
    return p

def bluntness_pressure(p, f, r, theta, phi, L, c, h, alpha, psi, V, c0, nu, trip, smooth):
    
    #Numero di Mach
    M = V/c0

    #Numero di Reynolds
    Re = (V * c)/nu

    dp, ds = displacement_thickness(Re, c, alpha, trip, smooth)

    dav = (dp + ds)/2.0             #spessore di spostamento medio al TE

    hdav = h/dav                    #ratio spessore TE su dav

    Dh = Dhfunc(M, theta, phi)

    Stpeak = Stpeak_func(hdav, psi, smooth)

    G4 = G4_func(hdav, psi, smooth)

    for i in range(len(f)):
    
        St = f[i] * h/V
        eta = np.log10(St/Stpeak)
        
        hdav_prime = 6.724 * hdav **2 - 4.019 * hdav + 1.107
        G5_0 = G5_func(hdav_prime, eta, smooth)

        G5_14 = G5_func(hdav, eta, smooth)

        G5 = G5_0 + 0.0714 * psi * (G5_14 - G5_0)

        scale = 10.0 * np.log10((M ** 5.5 * h * Dh * L)/r ** 2)

        spl_blunt = G4 + G5 + scale

        p[i] += 10.0 ** (spl_blunt/10.0)

    return p    

def tip_pressure(p, f, r, theta, phi, c, alpha, V, c0, aspect_ratio, round, smooth):

    if smooth:
        aratio_dx = 0.5
        if aspect_ratio < 2.0 + aratio_dx:
            aratio_f1 = 0.5
            aratio_f2 = aspect_ratio_correction(aspect_ratio)
            aratio = quintic_blend(aratio_f1, aratio_f2, aspect_ratio, 2.0, aratio_dx)
        elif aspect_ratio <= 24.0 + aratio_dx:
            aratio_f2 = aspect_ratio_correction(aspect_ratio)
            aratio_f3 = 1.0
            aratio = quintic_blend(aratio_f2, aratio_f3, aspect_ratio, 24.0, aratio_dx)
        else:
            aratio = 1.0
    else:
        if aspect_ratio < 2.0:
            aratio = 0.5
        elif 2.0 <= aspect_ratio <= 24.0:
            aratio = aspect_ratio_correction(aspect_ratio)
        else:
            aratio = 1.0

    alpha = aratio * alpha

    #Numero di Mach
    M = V/c0

    #Funzione di direttività
    Dh = Dhfunc(M, theta, phi)

    if round:                           #Se round=True, il tip è arrotondato, viceversa è piatto
        l = 0.008 * alpha * c
    else:
        if smooth:
            l_dx = 0.05
            l_f1 = (0.0230 + 0.0169 * alpha) * c
            l_f2 = (0.0378 + 0.0095 * alpha) * c
            l = quintic_blend(l_f1, l_f2, alpha, 2.0, l_dx)
        else:                           #tip piatto
            if alpha <= 2.0:
                l = (0.0230 + 0.0169 * alpha) * c
            else:
                l = (0.0378 + 0.0095 * alpha) * c
    
    Mmax = (1.0 + 0.036 * alpha) * M

    Vmax = Mmax * c0

    scale = 10.0 * np.log10((M ** 2 * Mmax ** 3 * l **2 * Dh)/r ** 2)

    for i in range(len(f)):
        
        St = (f[i] * l)/Vmax
        spl_tip = 126.0 - 30.5 * (np.log10(St) + 0.3) ** 2 + scale

        p[i] += 10.0 ** (spl_tip/10.0)
    
    return p

def boundary_thickness(Re, c, alpha):       #Calcolo dello spessore di strato limite delta

    d0 = c * 10.0 ** (1.6569 - 0.9045 * np.log10(Re) + 0.0596 * np.log10(Re) ** 2)      #Spessore di strato limite riscalato per c, ad angoli d'attacco effettivi nulli

    dp = d0 * 10 ** (-0.04175 * alpha + 0.00106 * alpha ** 2)                   #Spessore di strato limite riscalato per angoli d'attacco effettivi non nulli

    return dp

def displacement_thickness(Re, c, alpha, trip, smooth):                   #Calcolo dello spessore di spostamento delta*
    
    #Pressure side
    if trip:                                                #tripped boundary layer
        if smooth:
            d0_dx = 5.0
            d0_f1 = c * 0.0601 * Re ** (-0.114)
            d0_f2 = c * 10.0 ** (3.411 - 1.5397 * np.log10(Re) + 0.1059 * (np.log10(Re)) ** 2)
            d0 = quintic_blend(d0_f1, d0_f2, Re, 0.3e6, d0_dx)

        else:
            if Re <= 0.3e6:
                d0 = c * 0.0601 * Re ** (-0.114)
            else:
                d0 = c * 10.0 ** (3.411 - 1.5397 * np.log10(Re) + 0.1059 * (np.log10(Re)) ** 2)
    else:
        d0 = c * 10.0 ** (3.0187 - 1.5397 * np.log10(Re) + 0.1059 * (np.log10(Re)) ** 2)

    dp = d0 * 10.0 ** (-0.0432 * alpha + 0.00113 * alpha ** 2)

    #Suction side
    if trip:
        if smooth:
            ds_dx = 0.05
            if alpha < 5.0 + ds_dx:
                ds_f1 = d0 * 10.0 ** (0.069 * alpha)
                ds_f2 = d0 * 0.381 * 10.0 ** (0.1516 * alpha)
                ds = quintic_blend(ds_f1, ds_f2, alpha, 5.0, ds_dx)
            elif alpha < 15.5 + ds_dx:
                ds_f2 = d0 * 0.381 * 10.0 ** (0.1516 * alpha)
                ds_f3 = d0 * 14.296 * 10.0 ** (0.0258 * alpha)
                ds = quintic_blend(ds_f2, ds_f3, alpha, 12.5, ds_dx)
            else:
                ds = d0 * 14.296 * 10.0 ** (0.0258 * alpha)
        else:
            if alpha <= 5.0:
                ds = d0 * 10.0 ** (0.0679 * alpha)
            elif 5.0 < alpha <= 12.5:
                ds = d0 * 0.381 * 10.0 ** (0.1516 * alpha)
            else:
                ds = d0 * 14.296 * 10.0 ** (0.0258 * alpha)
    else:
        if smooth:
            ds_dx = 0.05 
            if alpha < 7.5 + ds_dx:
                ds_f1 = d0 * 10.0 ** (0.0679 * alpha)
                ds_f2 = d0 * 0.0162 * 10.0 ** (0.3066 * alpha)
                ds = quintic_blend(ds_f1, ds_f2, alpha, 7.5, ds_dx)
            elif alpha < 12.5 + ds_dx:
                ds_f2 = d0 * 0.0162 * 10.0 ** (0.3066 * alpha)
                ds_f3 = d0 * 52.42 * 10.0 ** (0.0258 * alpha)
                ds = quintic_blend(ds_f2, ds_f3, alpha, 12.5, ds_dx)
            else:
                ds = 52.42 * 10.0 ** (0.0258 * alpha)
        else:
            if alpha <= 7.5:
                ds = d0 * 10.0 ** (0.0679 * alpha)
            elif 7.5 < alpha <= 12.5:
                ds = d0 * 0.0162 * 10.0 ** (0.3066 * alpha)
            else:
                ds = d0 * 52.42 * 10.0 ** (0.0258 * alpha)
   
    return dp, ds

#Reynolds di riferimento, dipende da alpha e non da Vinf

def Re0_func(alpha, smooth):               
    
    if smooth:
        Re0_dx = 0.05 #gradi
        Re0_f1 = 10.0 ** (0.215 * alpha + 4.978)
        Re0_f2 = 10.0 ** (0.120 * alpha + 5.263)
        Re0 = quintic_blend(Re0_f1, Re0_f2, alpha,3.0, Re0_dx)
    else:
        if alpha <= 3.0:
            Re0 = 10.0 ** (0.215 * alpha + 4.978)
        else:
            Re0 = 10.0 ** (0.120 * alpha + 5.263)
    
    return Re0

'''Costanti di aggiustamento per il livello spettrale basate su Reynolds e angolo d'attacco.'''

def K1_func(Re, smooth):
    
    if smooth:
        K1_dx = 5.0
        if Re < 2.47e5 + K1_dx:
            K1_f1 = -4.31 * np.log10(Re) + 156.3
            K2_f2 = -9.0 * np.log10(Re) +181.6
            K1 = quintic_blend(K1_f1, K2_f2, Re, 2.47e5, K1_dx)
        elif  Re < 8.0e5 + K1_dx:
            K1_f2 = -9.0 * np.log10(Re) +181.6
            K1_f3 = 128.5
            K1 = quintic_blend(K1_f2, K1_f3, Re, 8.0e5, K1_dx)
        else:
            K1 = 128.5
    else:
        if Re < 2.47e5:
            K1 = -4.31 * np.log10(Re) + 156.3 
        elif 2.47e5 <= Re < 8.0e5:
            K1 = -9.0 * np.log10(Re) + 181.6 
        else:
            K1= 128.5
    
    return K1

def delta_K1_func(Rp, alpha, smooth):
    
    if smooth:
        delta_K1_dx = 0.5
        delta_K1_f1 = -alpha * (5.29 - 1.43 * np.log10(Rp))
        delta_K1_f2 = 0.0
        delta_K1 = quintic_blend(delta_K1_f1, delta_K1_f2, Rp, 5000.0, delta_K1_dx)
    else:
        if Rp <= 5000.0:
            delta_K1 = -alpha * (5.29 - 1.43 * np.log10(Rp))
        else:
            delta_K1 = 0.0
    
    return delta_K1     

def K2_func(alpha, M, K1, smooth):
    
    gamma = 27.094 * M + 3.31 
    gamma0 = 23.430 * M + 4.651 
    beta = 72.650 * M + 10.74
    beta0 = -34.190 * M - 13.820 

    if smooth:
        K2_dx = 0.05  
        if alpha < gamma0 - gamma + 2 * K2_dx:
            K2 = K1 - 1000.0
        elif alpha < gamma0 - gamma + 2 * K2_dx:
            K2_f1 = K1 - 1000.0
            K2_f2 = K1 + np.sqrt(beta ** 2 - (beta/gamma) ** 2 * (alpha-gamma0) ** 2) + beta0
            K2 = quintic_blend(K2_f1, K2_f2, alpha, gamma0 - gamma + K2_dx, K2_dx)
        elif alpha < gamma0 + gamma - 2 * K2_dx:
            K2 = K1 + np.sqrt(beta ** 2 - (beta/gamma) ** 2 * (alpha-gamma0) ** 2) + beta0
        elif alpha < gamma0 + gamma:
            K2_f2 = K1 + np.sqrt(beta ** 2 - (beta/gamma) ** 2 * (alpha-gamma0) ** 2) + beta0
            K2_f3 = K1 - 12.0
            K2 = quintic_blend(K2_f2, K2_f3, alpha, gamma0 + gamma - K2_dx, K2_dx)
        else:
            K2 = K1 - 12.0
    else:
        if alpha <= gamma0 - gamma:
            K2 = K1 - 1000.0
        elif gamma0 - gamma < alpha <= gamma0 + gamma:
            K2 = K1 + np.sqrt(beta ** 2 - (beta/gamma) ** 2 * (alpha-gamma0) ** 2) + beta0
        else:
            K2 = K1 - 12.0
    
    return K2

'''Equazioni per definire i parametri di frequenza (numero di Strouhal) necessari per 
scalare i diversi meccanisci di rumore, principalmente due, TBL per strato limite turbolento 
e LBL per strato limite laminare.'''

def St1p_func(Re,smooth):
    
    if smooth:
        St1p_dx = 5.0
        if Re < 1.3e5 + St1p_dx:
            St1p_f1 = 0.18
            St1p_f2 = 0.0017575566819517628 * Re ** 0.39311408564912614
            St1p = quintic_blend(St1p_f1, St1p_f2, Re, 1.3e5, St1p_dx)
        elif Re < 4.0e5 + St1p_dx:
            St1p_f2 =  0.0017575566819517628 * Re ** 0.39311408564912614
            St1p_f3 = 0.28
            St1p = quintic_blend(St1p_f2, St1p_f3, Re, 4.0e5, St1p_dx)
        else:
            St1p = 0.28
    else:
        if Re < 1.3e5:
            St1p = 0.18
        elif 1.3e5 <= Re < 4.0e5:
            St1p = 0.0017575566819517628 * Re ** 0.39311408564912614
        else:
            St1p = 0.28
    
    return St1p

def St2_func(alpha, St1, smooth):
    
    if smooth:
        St2_dx = 0.05
        if alpha < 1.333 + St2_dx:
            St2_f1 = St1
            St2_f2 = St1 * 10.0 ** (0.0054 * (alpha - 1.333) ** 2)
            St2 = quintic_blend(St2_f1, St2_f2, alpha, 1.333, St2_dx)
        elif alpha < 12.5 + St2_dx:
            St2_f2 = St1 * 10.0 ** (0.0054 * (alpha - 1.333) ** 2)
            St2_f3 = St1 * 4.72
            St2 = quintic_blend(St2_f2, St2_f3, alpha, 12.5, St2_dx)
        else:
            St2 = St1 * 4.72
    else:
        if alpha <= 1.333:
            St2 = St1
        elif 1.333 < alpha <= 12.5:
            St2 = St1 * 10.0 ** (0.0054 * (alpha - 1.333) ** 2)
        else:
            St2 = St1 * 4.72
  
    return St2

def Stpeak_func(hdav, psi, smooth):
    
    if smooth:
        Stpeak_dx = 0.001
        Stpeak_f1 = 0.1 * hdav +  0.095 - 0.00243 * psi
        Stpeak_f2 =(0.212 - 0.0045 * psi)/(1.0 + 0.235/hdav - 0.0132/hdav ** 2)
        Stpeak = quintic_blend(Stpeak_f1, Stpeak_f2, hdav, 0.2, Stpeak_dx)
    else:
        if 0.2 <= hdav:
            Stpeak = (0.212 - 0.0045 * psi)/(1.0 + 0.235/hdav - 0.0132/hdav ** 2)
        else:
            Stpeak = 0.1 * hdav +  0.095 - 0.00243 * psi
    
    return Stpeak

'''Funzioni spettrali per il rumore laminare.'''

def G1_func(e, smooth):
    
    if smooth:
        G1_dx = 0.001
        if e < 0.5974 - G1_dx:
            G1 = 39.8 * np.log10(e) - 11.12
        elif e < 0.5974 + G1_dx:
            G1_f1 = 39.8 * np.log10(e) - 11.12
            G1_f2 = 98.409 * np.log10(e) + 2.0
            G1 = quintic_blend(G1_f1, G1_f2, e, 0.5974, G1_dx)
        elif e < 0.8545 - G1_dx:
            G1 = 98.409 * np.log10(e) + 2.0
        elif e < 0.8545 + G1_dx:
            G1_f2 = 98.409 * np.log10(e) + 2.0
            G1_f3 = -5.076 + np.sqrt(2.484 - 506.25 * np.log10(e) ** 2)
            G1 = quintic_blend(G1_f2, G1_f3, e, 0.8545, G1_dx)
        elif e < 1.17 - G1_dx:
            G1 = -5.076 + np.sqrt(2.484 - 506.25 * np.log10(e) ** 2)
        elif e < 1.17 + G1_dx:
            G1_f3 = -5.076 + np.sqrt(2.484 - 506.25 * np.log10(e) ** 2)
            G1_f4 = -98.409 * np.log10(e) + 2.0
            G1 = quintic_blend(G1_f3, G1_f4, e, 1.17, G1_dx)
        elif e < 1.674 - G1_dx:
            G1 = -98.409 * np.log10(e) + 2.0
        elif e < 1.674 + G1_dx:
            G1_f4 = -98.409 * np.log10(e) + 2.0
            G1_f5 = -39.8 * np.log10(e) + 11.12   
            G1 = quintic_blend(G1_f4, G1_f5, e, 1.674, G1_dx)
        else:
            G1 = -39.8 * np.log10(e) + 11.12      
    else:
        if e <= 0.5974:
            G1 = 39.8 * np.log10(e) - 11.12
        elif 0.5974 < e <= 0.8545:
            G1 = 98.409 * np.log10(e) + 2.0
        elif 0.8545 < e <= 1.17:
            G1 = -5.076 + np.sqrt(2.484 - 506.25 * np.log10(e)**2)
        elif 1.17 < e <= 1.674:
            G1 = -98.409 * np.log10(e) + 2.0
        else:
            G1 = -39.8 * np.log10(e) + 11.12  
    
    return G1

def G2_func(d, smooth):
    
    if smooth:
        G2_dx = 0.001
        if d < 0.3237 + G2_dx:
            G2_f1 = 77.852 * np.log10(d) + 15.328
            G2_f2 = 65.188 * np.log10(d) + 9.125
            G2 = quintic_blend(G2_f1, G2_f2, d, 0.3237, G2_dx)
        elif d < 0.5689 + G2_dx:
            G2_f2 = 65.188 * np.log10(d) + 9.125
            G2_f3 = -114.052 * np.log10(d)**2
            G2 = quintic_blend(G2_f2, G2_f3, d, 0.5689, G2_dx)
        elif d < 1.7579 + G2_dx:
            G2_f3 = -114.052 * np.log10(d)**2
            G2_f4 = -65.188 * np.log10(d) + 9.125
            G2 = quintic_blend(G2_f3, G2_f4, d, 1.7579, G2_dx)
        elif d < 3.0889 + G2_dx:
            G2_f4 = -65.188 * np.log10(d) + 9.125
            G2_f5 = -77.852 * np.log10(d) + 15.328
            G2 = quintic_blend(G2_f4, G2_f5, d, 3.0889, G2_dx)
        else:
            G2 = -77.852 * np.log10(d) + 15.328
    else:
        if d <= 0.3237:
            G2 = 77.852 * np.log10(d) + 15.328
        elif 0.3237 < d <= 0.5689:
            G2 = 65.188 * np.log10(d) + 9.125
        elif 0.5689 < d <= 1.7579:
            G2 = -114.052 * np.log10(d)**2
        elif 1.7579 < d <= 3.0889:
            G2 = -65.188 * np.log10(d) + 9.125
        else:
            G2 = -77.852 * np.log10(d) + 15.328
    
    return G2   

'''Funzioni spettrali per il rumore da spessore del bordo d'uscita'''

def G4_func(hdav, psi, smooth):
    
    if  smooth:
        G4_f1 = 17.5 * np.log10(hdav) + 157.5 - 1.114 * psi    
        G4_f2 = 169.7 - 1.114 * psi
        G4 = quintic_blend(G4_f1, G4_f2, hdav, 5.0, 0.01)
    else:
        if hdav <= 5:
            G4 = 17.5 * np.log10(hdav) + 157.5 - 1.114 * psi
        else:
            G4 = 169.7 - 1.114 * psi
    
    return G4

def G5_func(hdav, eta, smooth):
    
    if smooth:
        mu_dx = 0.001
        if hdav < 0.25 + mu_dx:
            mu_f1 = 0.1221
            mu_f2 = -0.2175 * hdav + 0.1755
            mu = quintic_blend(mu_f1, mu_f2, hdav, 0.25, mu_dx)
        elif hdav < 0.62 + mu_dx:
            mu_f2 = -0.2175 * hdav + 0.1755
            mu_f3 = -0.0308 * hdav + 0.0596
            mu = quintic_blend(mu_f2, mu_f3, hdav, 0.62, mu_dx)
        elif hdav < 1.15 + mu_dx:
            mu_f3 = -0.0308 * hdav + 0.0596
            mu_f4 = 0.0242
            mu = quintic_blend(mu_f3, mu_f4, hdav, 1.15, mu_dx) 
        else:
            mu = 0.0242

        m_dx = 0.001
        if hdav < 0.02 + m_dx:
            m_f1 = 0.0
            m_f2 = 68.724 * hdav - 1.35
            m = quintic_blend(m_f1, m_f2, hdav, 0.02, m_dx)
        elif hdav < 0.5 + m_dx:
            m_f2 = 68.724 * hdav - 1.35
            m_f3 = 308.475 * hdav - 121.23
            m = quintic_blend(m_f2, m_f3, hdav, 0.5, m_dx)
        elif hdav < 0.62 + m_dx:
            m_f3 = 308.475 * hdav - 121.23
            m_f4 = 224.811 * hdav - 69.35
            m = quintic_blend(m_f3, m_f4, hdav, 0.62, m_dx)
        elif hdav < 1.15 + m_dx:
            m_f4 = 224.811 * hdav - 69.35
            m_f5 = 1583.28 * hdav - 1631.59
            m = quintic_blend(m_f4, m_f5, hdav, 1.15, m_dx)
        elif hdav < 1.2 + m_dx:
            m_f5 = 1583.28 * hdav - 1631.59
            m_f6 = 268.344
            m = quintic_blend(m_f5, m_f6, hdav, 1.2, m_dx)
        else:
            m = 268.344
    else:
        if hdav < 0.25:
            mu = 0.1221
        elif 0.25 <= hdav < 0.62:
            mu = -0.2175 * hdav + 0.1755
        elif 0.62 <= hdav < 1.15:
            mu = -0.0308 * hdav + 0.0596
        else:
            mu = 0.0242

        if hdav <= 0.02:
            m = 0.0
        elif 0.02 < hdav <= 0.5:
            m = 68.724 * hdav - 1.35
        elif 0.5 < hdav <= 0.62:
            m = 308.475 * hdav - 121.23
        elif 0.62 < hdav <= 1.15:
            m = 224.811 * hdav - 69.35
        elif 1.15 < hdav <= 1.2:
            m = 1583.28 * hdav - 1631.59
        else:
            m = 268.344
    
    eta_0 = -np.sqrt((m ** 2 * mu ** 4)/(6.35 + m ** 2 * mu ** 2))

    k = 2.5 * np.sqrt(1.0 - (eta_0/mu) ** 2) - 2.5 - m * eta_0

    if smooth:
        G5_dx = 0.001
        if eta < eta_0:
            G5 = m * eta + k
        elif eta < eta_0 + 2 * G5_dx:
            G5_f1 = m * eta + k
            G5_f2 = 2.5 * np.sqrt(1.0 - (eta_0/mu) ** 2) - 2.5
            G5 = quintic_blend(G5_f1, G5_f2, eta, eta_0 + G5_dx, G5_dx)
        elif eta < 0.0 - 2 * G5_dx:
            G5 = 2.5 * np.sqrt(1.0 - (eta_0/mu) ** 2) - 2.5
        elif eta < 0.0:
            G5_f2 = 2.5 * np.sqrt(1.0 - (eta_0/mu) ** 2) - 2.5
            G5_f3 = np.sqrt(1.5625 - 1194.99 * eta ** 2) - 1.25
            G5 = quintic_blend(G5_f2, G5_f3, eta, 0.0, G5_dx)
        elif eta < 0.03616 - 2 * G5_dx:
            G5 = np.sqrt(1.5625 - 1194.99 * eta ** 2) - 1.25
        elif eta < 0.03616:
            G5_f3 = np.sqrt(1.5625 - 1194.99 * eta ** 2) - 1.25
            G5_f4 = -155.543 * eta + 4.375
            G5 = quintic_blend(G5_f3, G5_f4, eta, 0.03616 - G5_dx, G5_dx)
        else:
            G5 = -155.543 * eta + 4.375
    else:
        if eta < eta_0:
            G5 = m * eta + k
        elif eta_0 <= eta < 0.0:
            G5 = 2.5 * np.sqrt(1.0 - (eta_0/mu) ** 2) - 2.5
        elif 0.0 <= eta < 0.03616:
            G5 = np.sqrt(1.5625 - 1194.99 * eta ** 2) - 1.25
        else:
            G5 = -155.543 * eta + 4.375
    
    return G5


'''Equazioni che definiscono le curve di interpolazione utilizzate per determinare 
la forma dello spettro del rumore dello strato limite turbolento 
al bordo d'uscita e del rumore di separazione.'''

def Amin_func(a, smooth):
    
    if smooth:
        a = abs_smooth(a, 0.001)
        a_dx = 0.001
        if a < 0.204 - a_dx:
            Amin = np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
        elif a < 0.204 + a_dx:
            Amin_f1 =  np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
            Amin_f2 = -32.665 * a + 3.981
            Amin = quintic_blend(Amin_f1, Amin_f2, a, 0.204, a_dx)
        elif a < 0.244 + a_dx:
            Amin = -32.665 * a + 3.981
        elif a < 0.244 + a_dx:
            Amin_f2 = -32.665 * a + 3.981
            Amin_f3 = -142.795 * a ** 3 + 103.656 * a ** 2 - 57.757 * a + 6.006
            Amin = quintic_blend(Amin_f2, Amin_f3, a, 0.244, a_dx)
        else:
            Amin = -142.795 * a ** 3 + 103.656 * a ** 2 - 57.757 * a + 6.006
    else:
        a = abs(a)
        if a < 0.204:
            Amin = np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
        elif 0.204 <= a <= 0.244:
            Amin = -32.665 * a + 3.981
        else:
            Amin = -142.795 * a ** 3 + 103.656 * a ** 2 - 57.757 * a + 6.006
    
    return Amin

def Amax_func(a, smooth):
    
    if smooth:
        a = abs_smooth(a, 0.001)
        a_dx = 0.001
        if a < 0.13 - a_dx:
            Amax = np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
        elif a < 0.13 + a_dx:
            Amax_f1 = np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
            Amax_f2 = -15.901 * a + 1.098
            Amax = quintic_blend(Amax_f1, Amax_f2, a, 0.13, a_dx)
        elif a < 0.321 - a_dx:
            Amax = -15.901 * a + 1.098
        elif a < 0.321 + a_dx:
            Amax_f2 = -15.901 * a + 1.098
            Amax_f3 = -4.669 * a ** 3 + 3.491 * a ** 2 - 16.669 * a + 1.149
            Amax = quintic_blend(Amax_f2, Amax_f3, a, 0.321, a_dx)
        else:
            Amax = -4.669 * a ** 3 + 3.491 * a ** 2 - 16.669 * a + 1.149
    else:
        a = abs(a)
        if a < 0.13:
            Amax = np.sqrt(67.552 - 886.788 * a ** 2) - 8.219
        elif 0.13 <= a <= 0.321:
            Amax = -15.901 * a + 1.098
        else:
            Amax = -4.669 * a ** 3 + 3.491 * a ** 2 - 16.669 * a + 1.149
    
    return Amax

def Bmin_func(b, smooth):
    
    if smooth:
        b = abs_smooth(b, 0.001)
        b_dx = 0.001
        if b < 0.13 - b_dx:
            Bmin = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
        elif b < 0.13 + b_dx:
            Bmin_f1 = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
            Bmin_f2 = -83.607 * b + 8.138
            Bmin = quintic_blend(Bmin_f1, Bmin_f2, b, 0.13, b_dx)
        elif b < 0.145 - b_dx:
            Bmin = -83.607 * b + 8.138
        elif b < 0.145 + b_dx:
            Bmin_f2 = -83.607 * b + 8.138
            Bmin_f3 = -817.810 * b ** 3 + 355.210 * b ** 2 -135.024 * b + 10.619
            Bmin = quintic_blend(Bmin_f2, Bmin_f3, b, 0.145, b_dx)
        else:
            Bmin = -817.810 * b ** 3 + 355.210 * b ** 2 -135.024 * b + 10.619
    else:
        b = abs(b)
        if b < 0.13:
            Bmin = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
        elif 0.13 <= b <= 0.145:
            Bmin = -83.607 * b + 8.138
        else:
            Bmin = -817.810 * b ** 3 + 355.210 * b ** 2 -135.024 * b + 10.619
    
    return Bmin 

def Bmax_func(b, smooth):
    
    if smooth:
        b = abs_smooth(b, 0.001)
        b_dx = 0.001
        if b < 0.10 - b_dx:
            Bmax = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
        elif b < 0.10 + b_dx:
            Bmax_f1 = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
            Bmax_f2 = -31.313 * b + 1.854 
            Bmax = quintic_blend(Bmax_f1, Bmax_f2, b, 0.10, b_dx)
        elif b < 0.187 - b_dx:
            Bmax = -31.313 * b + 1.854      #Nella teoria è -31.330 * b + 1.854
        elif b < 0.187 + b_dx:
            Bmax_f2 = -31.313 * b + 1.854
            Bmax_f3 = -80.541 * b ** 3 + 44.174 * b ** 2 - 39.381 * b + 2.344
            Bmax = quintic_blend(Bmax_f2, Bmax_f3, b, 0.187, b_dx)
        else:
            Bmax = -80.541 * b ** 3 + 44.174 * b ** 2 - 39.381 * b + 2.344
    else:
        b = abs(b)
        if b < 0.10:    
            Bmax = np.sqrt(16.888 - 886.788 * b ** 2) - 4.109
        elif 0.10 <= b <= 0.187:
            Bmax = -31.313 * b + 1.854           #Nella teoria è -31.330 * b + 1.854
        else:
            Bmax = -80.541 * b ** 3 + 44.174 * b ** 2 - 39.381 * b + 2.344
    
    return Bmax

def a0_func(Re, smooth):
    
    if smooth:
        a0_dx = 5.0
        if Re < 9.52e4 - a0_dx:
            a0 = 0.57
        elif Re < 9.52e4 + a0_dx:
            a0_f1 = 0.57    
            a0_f2 = -9.57e-13 * (Re - 8.57e5) ** 2 + 1.13
            a0 = quintic_blend(a0_f1, a0_f2, Re, 9.52e4, a0_dx)
        elif Re < 8.57e5 - a0_dx:
            a0 = -9.57e-13 * (Re - 8.57e5) ** 2 + 1.13
        elif Re < 8.57e5 + a0_dx:
            a0_f2 = -9.57e-13 * (Re - 8.57e5) ** 2 + 1.13
            a0_f3 = 1.13
            a0 = quintic_blend(a0_f2, a0_f3, Re, 8.57e5, a0_dx)
        else:
            a0 = 1.13
    else:
        if Re < 9.52e4:
            a0 = 0.57
        elif 9.52e4 <= Re <= 8.57e5:
            a0 = -9.57e-13 * (Re - 8.57e5) ** 2 + 1.13
        else:
            a0 = 1.13
    
    return a0

def b0_func(Re, smooth):
    
    if smooth:
        b0_dx = 5.0 
        if Re < 9.52e4 - b0_dx:
            b0 = 0.30
        elif Re < 9.52e4 + b0_dx:
            b0_f1 = 0.30
            b0_f2 = -4.48e-13 * (Re - 8.57e5) ** 2 + 0.56
            b0 = quintic_blend(b0_f1, b0_f2, Re, 9.52e4, b0_dx)
        elif Re < 8.57e5 - b0_dx:
            b0 = -4.48e-13 * (Re - 8.57e5) ** 2 + 0.56
        elif Re < 8.57e5 + b0_dx:
            b0_f2 = -4.48e-13 * (Re - 8.57e5) ** 2 + 0.56
            b0_f3 = 0.56
            b0 = quintic_blend(b0_f2, b0_f3, Re, 8.57e5, b0_dx)
        else:
            b0 = 0.56
    else:
        if Re < 9.52e4:
            b0 = 0.30
        elif 9.52e4 <= Re < 8.57e5:
            b0 = -4.48e-13 * (Re - 8.57e5) ** 2 + 0.56
        else:
            b0 = 0.56
    
    return b0

'''Calcola la funzione di direttività ad alta frequenza per la posizione dell'osservatore in input'''

def Dhfunc(M, theta, phi):
  
    #Numero di Mach assunto per convenzione
    Mc = 0.8 * M

    # Funzione di direttività
    Dh = (2 * np.sin(theta/2.0) ** 2 * np.sin(phi) ** 2)/((1 + M * np.cos(theta)) * (1 + (M - Mc) * np.cos(theta)) ** 2)

    return Dh

'''Calcola la funzione di direttività a bassa frequenza per la posizione dell'osservatore in input'''

def Dlfunc(M, theta, phi):

    # Funzione di direttività
    Dl = (np.sin(theta) ** 2 * np.sin(phi) ** 2)/(1 + M * np.cos(theta)) ** 4

    return Dl
