import os
import sys
import pickle
import numpy as np
import itertools
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/phsun/212project")
from metropolis import metropolis_4dLGT

# parameters
size_1d = 8 
N = 2 # Z(N) symmetry
size = [size_1d, size_1d, size_1d, size_1d, 4]
Ns = np.prod(size) # number of links 
T_array = np.concatenate((
          np.arange(0.02, 3.0, 0.02),
          np.arange(0.01, 3.0, 0.02)[::-1]))

Ninit = 10 # in units of monte carlo per spin
Ncycles = 100 
Nsteps = 1 # monte carlo per spin for each cycle

r_array = np.arange(1, 5, 1) # side length of wilson loop

# file header
header = {'model':'4D LGT', 'size':size, 'N': N, 'T':None, \
          'Ninit':Ninit, 'Ncycles':Ncycles, 'Nsteps':Nsteps}

# create an array to store the real part for an argument
Re = np.cos(np.arange(N) * 2*np.pi/N)

# initialize lattice
lt = np.zeros(size) # all arguments are 0
lt1 = np.zeros([size[0]*2, size[1]*2, size[2]*2, size[3]*2, 4])
## random initialization
#for s in np.nditer(lt, op_flags=['readwrite']):
#    rand = np.random.rand()
#    if rand < 0.5:
#        s[...] = -1.
#    else:
#        s[...] = 1.

# Metropolis algorithm
for T in T_array:
    K = 1./T

    # initialize by doing Ninit steps
    for i_step in xrange(Ninit):
        metropolis_4dLGT(lt, K, N=2)
    #END initialization
    
    # START cycle loop; measure physical quantity after each cycle
    U_array = []
    W_array = [] # Wilson loops

    for i_cycle in xrange(Ncycles):
        for i_step in xrange(Nsteps):
            metropolis_4dLGT(lt, K, N=2) 
        #END step loop

        ## measure physical quantities for this cycle
        # first extend the lattice in all three directions
        lt1[:size[0], :size[1], :size[2], :size[3], :] = lt
        lt1[size[0]:, :size[1], :size[2], :size[3], :] = lt
        lt1[:size[0], size[1]:, :size[2], :size[3], :] = lt
        lt1[:size[0], :size[1], size[2]:, :size[3], :] = lt
        lt1[:size[0], :size[1], :size[2], size[3]:, :] = lt

        W = np.zeros(len(r_array))
        for i, j, k, l in itertools.product(*map(xrange, size[:-1])):
            # Wilson loop
            for i_r, r in enumerate(r_array):
                # sum in all three directions
                arg = 0
                for p in xrange(r): # xy
                    arg += lt1[i+p, j, k, l, 0]
                    arg += lt1[i+r, j+p, k, l, 1]
                    arg -= lt1[i+p, j+r, k, l, 0]
                    arg -= lt1[i, j+p, k, l, 1]
                W[i_r] += Re[arg % N]
                arg = 0
                for p in xrange(r): # yz
                    arg += lt1[i, j+p, k, l, 1]
                    arg += lt1[i, j+r, k+p, l, 2]
                    arg -= lt1[i, j+p, k+r, l, 1]
                    arg -= lt1[i, j, k+p, l, 2]
                W[i_r] += Re[arg % N]
                arg = 0
                for p in xrange(r): # zx
                    arg += lt1[i, j, k+p, l, 2]
                    arg += lt1[i+p, j, k+r, l, 0]
                    arg -= lt1[i+r, j, k+p, l, 2]
                    arg -= lt1[i+p, j, k, l, 0]
                W[i_r] += Re[arg % N]
                arg = 0
                for p in xrange(r): # xw
                    arg += lt1[i+p, j, k, l, 0]
                    arg += lt1[i+r, j, k, l+p, 3]
                    arg -= lt1[i+p, j, k, l+r, 0]
                    arg -= lt1[i, j, k, l+p, 3]
                W[i_r] += Re[arg % N]
                arg = 0
                for p in xrange(r): # wy
                    arg += lt1[i, j, k, l+p, 3]
                    arg += lt1[i, j+p, k, l+r, 1]
                    arg -= lt1[i, j+r, k, l+p, 3]
                    arg -= lt1[i, j+p, k, l, 1]
                W[i_r] += Re[arg % N]
                arg = 0
                for p in xrange(r): # zw
                    arg += lt1[i, j, k+p, l, 2]
                    arg += lt1[i, j, k+r, l+p, 3]
                    arg -= lt1[i, j, k+p, l+r, 2]
                    arg -= lt1[i, j, k, l+p, 3]
                W[i_r] += Re[arg % N]

        # record physical quantities
        U_array.append(-float(W[r_array==1])) # internal energy is W(r) for r=1
        W = W * 1. / (Ns/4*6)  
        W_array.append(W)
        
        #s0 = np.sum(lt)*1. / Ns # average spin, only
        #for i_r in xrange(len(r_array)):
        #    r = r_array[i_r]
        #    # measure <s0*sr>
        #    s0sr = 0.
        #    for i, j, k in itertools.product(*map(xrange, size)):
        #        ipr = (i+r) % size[0]
        #        jpr = (j+r) % size[1]
        #        kpr = (k+r) % size[2]
        #        s0sr += lt[i,j,k]*(lt[ipr,j,k]+lt[i,jpr,k]+lt[i,j,kpr])
        #    if r == 1:
        #        U.append(-s0sr*1.) # internal energy is defined as sum of
        #                           # s_i*s_j for nn

        #    s0sr /= Ns * 3. # 3 pairs for each site
        #    
        #    #print "<s0sr> =", s0sr, "<s0> =", s0
        #    s0sr_array[i_r] = s0sr_array[i_r] + s0sr - s0**2
        #    #print "s0sr:", s0sr_array

    #print zip(r_array, s0sr_array)
    #s0sr_array = s0sr_array / (Ncycles * 1.)
    #s0 = np.mean(M)*1./Ns

    # saving data to file
    os.chdir("/home/phsun/212project/4dLGT_Z%d/%d" % (N, size_1d))
    f = open("T_%.2f.pickle" % T, 'w')
    header['T'] = T
    pickle.dump({'header':header, 'U':U_array, 'U_ave':np.mean(U_array), \
                 'r':r_array, 'W':W_array, 'lt_final':lt}, f)
    f.close() 
