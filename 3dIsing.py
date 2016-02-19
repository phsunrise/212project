import numpy as np
import itertools
import matplotlib.pyplot as plt

# parameters
size_1d = 32
size = [size_1d, size_1d, size_1d]
Ns = np.prod(size) # number of sites
Kstart = 0.
Kend = 2.
Ksteps = 1000
K_array = np.append(np.linspace(Kstart, Kend, Ksteps, endpoint=False),\
                    np.linspace(Kend, Kstart, Ksteps, endpoint=False))

# initialize lattice
lt = np.zeros(size)
for s in np.nditer(lt, op_flags=['readwrite']):
    rand = np.random.rand()
    if rand < 0.5:
        s[...] = -1
    else:
        s[...] = 1

# Metropolis algorithm
Nsteps = 100000
U_array = [] # store internal energy
M_array = [] # store total magnetization

for K in K_array:
    U = 0 # internal energy, defined as sum of -s_i * s_j
    #M = 0 # magnetization, defined as sum of s_i
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
        #M += lt[i,j,k]
    
    U_array.append([K, U])
    #M_array.append(M*1.0)

#plt.figure()
#plt.plot(U_array/Ns)
#plt.savefig("InternalEnergy.png")

#plt.figure()
#plt.plot(M_array/Ns)
#plt.savefig("Magnetization.png")

plt.figure()
U_array = np.array(U_array)
plt.plot(U_array[:,0], U_array[:,1])
plt.savefig("thermalcycle_%d_lattice.png" % size_1d)
np.savez("thermalcycle_%d_lattice.npz" % size_1d, U=U_array)
