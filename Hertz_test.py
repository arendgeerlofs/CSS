import numpy as np
from Bak_Tang_Wiesenfield import BTW

def Hertz(alpha, h):
    spikes = BTW(50, 0, 2000, h=h, alpha=alpha, ret='Param')[1][500:]
    s = np.sum(spikes)
    return s/1500 * 1000 / 2500