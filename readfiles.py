import pickle
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt

T_array = []
U_array = []
M_array = []
xi_array = []
chi_array = []
C_array = []

for filename in glob.glob("T_*.pickle"):
    f = pickle.load(open(filename, 'r'))
    header = f['header']

    if header['model'] != '3D Ising':
        continue

    T = header['T']
    U_ave = f['U_ave']
    M_ave = f['M_ave']
    T_array.append(T)
    U_array.append(U_ave)
    M_array.append(M_ave)
    
    C_array.append((np.mean(np.array(f['U'])**2) - U_ave**2) / T**2)
    chi_array.append((np.mean(np.array(f['M'])**2) - M_ave**2) / T)
    slope, intercept, r_val, p_val, std_err = stats.linregress(f['r'], \
                                                -np.log(f['s0sr']))
    xi_array.append(1. / slope)

plt.figure()
plt.plot(T_array, U_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel("U")
plt.savefig("U.png")

plt.figure()
plt.plot(T_array, M_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel("M")
plt.savefig("M.png")

plt.figure()
plt.plot(T_array, xi_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel(r"$\xi$")
plt.savefig("xi.png")

plt.figure()
plt.plot(T_array, chi_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel(r"$\chi$")
plt.savefig("chi.png")

plt.figure()
plt.plot(T_array, C_array, 'bo')
plt.xlabel(r"$T/T_0$")
plt.ylabel("C")
plt.savefig("C.png")

print "max from xi: T =", T_array[xi_array.index(max(xi_array))]
print "max from chi: T =", T_array[chi_array.index(max(chi_array))]
print "max from C: T =", T_array[C_array.index(max(C_array))]

## output results
with open("results.txt", 'w') as fout:
    fout.write("max from xi: T = %.2f\n" %
        T_array[xi_array.index(max(xi_array))])
    fout.write("max from chi: T = %.2f\n"  % 
        T_array[chi_array.index(max(chi_array))])
    fout.write("max from C: T = %.2f\n" %
        T_array[C_array.index(max(C_array))])
