from time import time
import numpy as np
from scipy.optimize import minimize

def remove_background(spectral_axis, spectral_data, background, poly_order=3, max_iter=100, eps=0.1):
    """
    This function is based on the method described in: Beier, Brooke D., 
    and Andrew J. Berger. “Method for Automated Background Subtraction from 
    Raman Spectra Containing Known Contaminants.” Analyst 134, no. 6 (2009): 
    1198–1202. https://doi.org/10.1039/B821856K.
    """

    emp_scaling_factor = 0.6 # Empirical scaling factor for the background
    X = background

    S = spectral_data
    B = [S.copy()]
    
    if X is None:
        C = 0
        X = np.arange(len(S))
    else:
        p = np.polyfit(X, S, 1)
        C = p[0] * emp_scaling_factor

    err = np.inf
    i = 0

    start_time = time()
    while err > eps and i <= max_iter:
        def cost_function(C):
            return np.sum((B[i] - C * X - np.polyval(np.polyfit(spectral_axis, B[i] - C * X, poly_order), spectral_axis))**2)

        res = minimize(cost_function, C, method='Nelder-Mead')
        C = res.x[0]

        F = B[i] - C * X
        poly_model = np.polyval(np.polyfit(spectral_axis, F, poly_order), spectral_axis)
        Btd = C * X + poly_model

        B.append(np.minimum(B[i], Btd))
        i += 1

        err = np.sqrt(np.sum((B[-1] - B[-2])**2))
    
    # print(f"Final background scaling factor for spectrum {k + 1}: {C}")
    # print(f"Final error for background removal: {err}")
    # print(f"Time taken: {time() - start_time:.2f} seconds")

    RS = S - C * X - poly_model

    return RS