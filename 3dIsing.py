import os
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt
from metropolis import metropolis_3dI

# parameters
size_1d = 32 
size = [size_1d, size_1d, size_1d]
Ns = np.prod(size) # number of sites
T_array = np.arange(4.32, 4.7, 0.02)

Ninit = 100 # in units of monte carlo per spin
Ncycles = 1000 
Nsteps = 1 # monte carlo per spin for each cycle


# file header
header = {'model':'3D Ising', 'size':size, 'T':None, \
          'Ninit':Ninit, 'Ncycles':Ncycles, 'Nsteps':Nsteps}

# initialize lattice
lt = np.zeros(size)
for s in np.nditer(lt, op_flags=['readwrite']):
    ## random spins
    #rand = np.random.rand()
    #if rand < 0.5:
    #    s[...] = -1.
    #else:
    #    s[...] = 1.
    
    # all spins up
    s[...] = 1.

# Metropolis algorithm
r_array = np.arange(5)
for T in T_array:
    K = 1./T

    # initialize by doing Ninit steps
    for i_step in xrange(Ninit):
        metropolis_3dI(lt, K)
    #END initialization
    
    # START cycle loop; measure physical quantity after each cycle
    s0sr_array = np.zeros(5)
    M = []
    U = []

    for i_cycle in xrange(Ncycles):
        for i_step in xrange(Nsteps):
            metropolis_3dI(lt, K) 
        #END step loop

        # measure physical quantities for this cycle
        M.append(np.sum(lt)*1.)
        s0 = np.sum(lt)*1. / Ns # average spin, only
        for i_r in xrange(len(r_array)):
            r = r_array[i_r]
            # measure <s0*sr>
            s0sr = 0.
            for i, j, k in itertools.product(*map(xrange, size)):
                ipr = (i+r) % size[0]
                jpr = (j+r) % size[1]
                kpr = (k+r) % size[2]
                s0sr += lt[i,j,k]*(lt[ipr,j,k]+lt[i,jpr,k]+lt[i,j,kpr])
            if r == 1:
                U.append(-s0sr*1.) # internal energy is defined as sum of
                                   # s_i*s_j for nn

            s0sr /= Ns * 3. # 3 pairs for each site
            
            #print "<s0sr> =", s0sr, "<s0> =", s0
            s0sr_array[i_r] = s0sr_array[i_r] + s0sr - s0**2
            #print "s0sr:", s0sr_array

    #print zip(r_array, s0sr_array)
    s0sr_array = s0sr_array / (Ncycles * 1.)
    s0 = np.mean(M)*1./Ns

    #plt.figure()
    #plt.plot(r_array, -np.log(abs(s0sr_array)))
    #plt.xlabel("r")
    #plt.ylabel(r"$-\ln(<\sigma_0\sigma_r>-<s_0><s_r>)$")
    #plt.savefig("corrlen_K_%.2f.png" % K)


    # saving data to file
    os.chdir("/home/phsun/212project/3dIsing/%d" % size_1d)
    f = open("T_%.2f.pickle" % T, 'w')
    header['T'] = T
    pickle.dump({'header':header, 'M':M, 'M_ave':np.mean(M),\
                 'U':U, 'U_ave':np.mean(U), 's0':s0, \
                 'r':r_array, 's0sr':s0sr_array}, f)
    f.close() 
