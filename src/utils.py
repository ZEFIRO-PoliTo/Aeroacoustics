"""Generic numerical utility functions."""

import numpy as np
from scipy.interpolate import Akima1DInterpolator

def trapz(x, y):            
    return np.trapezoid(y, x)

def quintic_blend(f1, f2, x, x0, dx):                           #Prende due valori (o funzioni) f1 e f2 e restituisce una transizione fluida tra di essi in un intervallo dx attorno a un punto critico x0.
    s = (x - (x0 - dx)) / (2 * dx)
    if s <= 0: return f1
    if s >= 1: return f2
    w = 10 * s**3 - 15 * s**4 + 6 * s**5
    return f1 + w * (f2 - f1)

def abs_smooth(x, delta):                                       #Evita di avere un punto di non derivabilità quando utilizzo l'operatore abs()
    return np.sqrt(x**2 + delta**2)

def aspect_ratio_correction(AR):                                   #Interpolazione con una spline che evita delle oscillazioni artificiali.
    aspect_data = np.array([2.0, 2.67, 4.0, 6.0, 12.0, 24.0])
    aratio_data = np.array([0.54, 0.62, 0.71, 0.79, 0.89, 0.95])
    interp = Akima1DInterpolator(aspect_data, aratio_data)
    return float(interp(AR))

def ksmin(values, rho=50.0):
    """
    Kreisselmeier-Steinhauser smooth minimum.
    Trova il minimo in un array di valori smussando gli angoli vivi.
    'rho' controlla quanto è "dura" la curva (valori più alti = più simile al min standard).
    """
    values = np.array(values)
    max_val = np.max(values)
    return max_val - (1.0 / rho) * np.log(np.sum(np.exp(rho * (max_val - values))))
