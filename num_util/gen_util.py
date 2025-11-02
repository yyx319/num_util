import numpy as np
import astropy.constants as c; import astropy.units as u

def is_unitless(x):
    """
    Return True if x is unitless, False otherwise.
    x can be either invividual or array.
    """
    if type(x) in [int, np.int64, float, np.float64, np.complex128, np.ndarray]:
        return True
    elif type(x) == u.quantity.Quantity:
        return x.unit == u.dimensionless_unscaled
    else:
        return Exception("Type not recognized")