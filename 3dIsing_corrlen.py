import numpy as np
import itertools
import matplotlib.pyplot as plt
from metropolis import metropolis_3dI

# parameters
size_1d = 32
size = [size_1d, size_1d, size_1d]
Ns = np.prod(size) # number of sites
Kstart = 0.3
Kend = 0.3
Ksteps = 1
K_array = np.linspace(Kstart, Kend, Ksteps, endpoint=False)

# initialize lattice
lt = np.zeros(size)
for s in np.nditer(lt, op_flags=['readwrite']):
    rand = np.random.rand()
    if rand < 0.5:
        s[...] = -1.
    else:
        s[...] = 1.

# Metropolis algorithm
Ninit = 100000
Ncycles = 1000 
Nsteps = 100 # steps for each cycle

for K in K_array:
    U = 0 # internal energy, defined as sum of -s_i * s_j
    # initialize by doing Ninit steps
    for i_step in xrange(Ninit):
        metropolis_3dI(lt, K)
    #END initialization
    
    # START cycle loop; measure physical quantity after each cycle
    r_array = np.arange(16)
    s0sr_array = np.zeros(16)

    for i_cycle in xrange(Ncycles):
        for i_step in xrange(Nsteps):
            metropolis_3dI(lt, K) 
        #END step loop

        # measure physical quantities for this cycle
        for i_r in xrange(len(r_array)):
            r = r_array[i_r]
            # measure <s0*sr>
            s0sr = 0.
            for i, j, k in itertools.product(*map(xrange, size)):
                ipr = (i+r) % size[0]
                jpr = (j+r) % size[1]
                kpr = (k+r) % size[2]
                s0sr += lt[i,j,k]*(lt[ipr,j,k]+lt[i,jpr,k]+lt[i,j,kpr])
            s0sr /= Ns * 3. # 3 pairs for each site

            s0 = np.sum(lt)*1. / Ns
            #print "<s0sr> =", s0sr, "<s0> =", s0
            s0sr_array[i_r] = s0sr_array[i_r] + s0sr - s0**2
            #print "s0sr:", s0sr_array

    print zip(r_array, s0sr_array)
    s0sr_array = s0sr_array / (Ncycles * 1.)

    plt.figure()
    plt.plot(r_array, -np.log(abs(s0sr_array)))
    plt.xlabel("r")
    plt.ylabel(r"$-\ln(<\sigma_0\sigma_r>-<s_0><s_r>)$")
    plt.savefig("corrlen_K_%.2f.png" % K)
