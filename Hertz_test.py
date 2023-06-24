import numpy as np
from Bak_Tang_Wiesenfield import BTW

def Hertz(alpha, h):
    spikes = BTW(50, 0, 2000, h=h, alpha=alpha, ret='Param')[1][500:]
    print(spikes)
    s = np.sum(spikes)
    print(s)
    return s/1500 * 1000 / 2500

print(Hertz(1, 0.0001))