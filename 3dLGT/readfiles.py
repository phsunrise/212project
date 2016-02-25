import pickle
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt

T_array = []
U_array = []
C_array = []
W_array = []

for filename in glob.glob("T_*.pickle"):
    f = pickle.load(open(filename, 'r'))
    header = f['header']

    if header['model'] != '3D LGT':
        continue

    T = header['T']
    U_ave = f['U_ave']
    T_array.append(T)
    U_array.append(U_ave)
    
    C_array.append((np.mean(np.array(f['U'])**2) - U_ave**2) / T**2)
    #slope, intercept, r_val, p_val, std_err = stats.linregress(f['r'], \
    #                                            -np.log(f['s0sr']))
    #xi_array.append(1. / slope)

T1_array = -2/np.log(np.tanh(1./np.array(T_array)))

plt.figure()
plt.grid(True, which='both', axis='both')
plt.plot(T_array, U_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel("U")
plt.savefig("U.png")

plt.figure()
plt.grid(True, which='both', axis='both')
plt.plot(T_array, C_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel("C")
plt.savefig("C.png")

print "max from C: T =", T_array[C_array.index(max(C_array))]

## output results
with open("results.txt", 'w') as fout:
    fout.write("max from C: T = %.2f\n" %
        T_array[C_array.index(max(C_array))])
