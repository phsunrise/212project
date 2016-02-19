import numpy as np

def metropolis_3dI(lt, K):
    size = lt.shape
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
# END 3d Ising metropolis 
