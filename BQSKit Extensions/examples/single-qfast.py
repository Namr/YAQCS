import numpy as np
from qfast import synthesize
single = np.loadtxt( "single.unitary", dtype = np.complex128 )
print(synthesize(single))
