import numpy as np
import itertools
import matplotlib.pyplot as plt

# parameters
size = [50, 50, 50]
Ns = np.prod(size) # number of sites
K = 10 

# initialize lattice
lt = np.zeros(size)
for s in np.nditer(lt, op_flags=['readwrite']):
    rand = np.random.rand()
    if rand < 0.5:
        s[...] = -1
    else:
        s[...] = 1

# Metropolis algorithm
Nsteps = 1000
U_array = [] # store internal energy
M_array = [] # store total magnetization

for i_cycle in xrange(10000):
    U = 0 # internal energy, defined as sum of -s_i * s_j
    M = 0 # magnetization, defined as sum of s_i
    for i_step in xrange(Nsteps):
        # choose a random site
        i = np.random.randint(0, size[0])
        j = np.random.randint(0, size[1])
        k = np.random.randint(0, size[2])
        
        # trail flip
        new = -lt[i,j,k]
        
        # compute H
        ip1 = (i+1) % size[0]
        im1 = (i-1) % size[0]
        jp1 = (j+1) % size[1]
        jm1 = (j-1) % size[1]
        kp1 = (k+1) % size[2]
        km1 = (k-1) % size[2]
        H_old = K*lt[i,j,k]*(lt[ip1,j,k]+lt[im1,j,k]
                            +lt[i,jp1,k]+lt[i,jm1,k]
                            +lt[i,j,kp1]+lt[i,j,km1])
        H_new = K*new      *(lt[ip1,j,k]+lt[im1,j,k]
                            +lt[i,jp1,k]+lt[i,jm1,k]
                            +lt[i,j,kp1]+lt[i,j,km1])

        # accept or decline
        if H_new >= H_old:
            lt[i,j,k] = -lt[i,j,k]
        else:
            rand = np.random.rand()
            if rand < np.exp(H_new - H_old):
                lt[i,j,k] = -lt[i,j,k]
    #END step loop

    # measure physical quantities
    for i, j, k in itertools.product(*map(xrange, size)):
        ip1 = (i+1) % size[0]
        jp1 = (j+1) % size[1]
        kp1 = (k+1) % size[2]
        U += -lt[i,j,k]*(lt[ip1,j,k]+lt[i,jp1,k]+lt[i,j,kp1])
        M += lt[i,j,k]
    
    U_array.append(U*1.0)
    M_array.append(M*1.0)

plt.figure()
plt.plot(U_array/Ns)
plt.savefig("InternalEnergy.png")

plt.figure()
plt.plot(M_array/Ns)
plt.savefig("Magnetization.png")

np.savez("50_lattice_100_step_1000_cycles.npz", U=U_array, M=M_array)
