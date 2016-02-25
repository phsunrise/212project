import numpy as np
import itertools


size_1d = 2 
N = 2 # Z(N) symmetry
size = [size_1d, size_1d, size_1d, 3]
Ns = np.prod(size)

Re = np.cos(np.arange(N) * 2*np.pi/N)

lt = np.zeros(size) # all arguments are 0
lt1 = np.zeros([size[0]*2, size[1]*2, size[2]*2, 3])
for i_entry in xrange(Ns/2):
    i = np.random.randint(0, size[0])
    j = np.random.randint(0, size[1])
    k = np.random.randint(0, size[2])
    d = np.random.randint(0, 3)
    lt[i,j,k,d] = np.random.randint(0, N)

lt1[:size[0], :size[1], :size[2], :] = lt
lt1[size[0]:, :size[1], :size[2], :] = lt
lt1[:size[0], size[1]:, :size[2], :] = lt
lt1[:size[0], :size[1], size[2]:, :] = lt
U = 0.
for i, j, k in itertools.product(*map(xrange, size[:-1])):
    print "[%d,%d,%d]" % (i,j,k), lt[i,j,k]
    # sum in all three directions
    arg = 0
    r = 1
    for p in xrange(r):
        arg += lt1[i+p, j, k, 0]
        arg += lt1[i+r, j+p, k, 1]
        arg -= lt1[i+p, j+r, k, 0]
        arg -= lt1[i, j+p, k, 1]
    U += Re[arg % N]
    arg = 0
    for p in xrange(r):
        arg += lt1[i, j+p, k, 1]
        arg += lt1[i, j+r, k+p, 2]
        arg -= lt1[i, j+p, k+r, 1]
        arg -= lt1[i, j, k+p, 2]
    U += Re[arg % N]
    arg = 0
    for p in xrange(r):
        arg += lt1[i, j, k+p, 2]
        arg += lt1[i+p, j, k+r, 0]
        arg -= lt1[i+r, j, k+p, 2]
        arg -= lt1[i+p, j, k, 0]
    U += Re[arg % N]

print U
