import numpy as np

debug = False 

# do metropolis algorithm for 1 MC per spin
def metropolis_3dI(lt, K):
    size = lt.shape
    Ns = np.prod(size)

    for i_step in xrange(Ns):
        # choose a random site
        i = np.random.randint(0, size[0])
        j = np.random.randint(0, size[1])
        k = np.random.randint(0, size[2])
        
        # trail flip
        new = -lt[i,j,k]
        
        # compute H (actually -beta*H)
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



def metropolis_3dLGT(lt, K, N=2):
    size = lt.shape
    Ns = np.prod(size)
    
    # create an array to store the real part for an argument
    Re = np.cos(np.arange(N) * 2*np.pi/N)

    for i_step in xrange(Ns):
        # choose a random link
        i = np.random.randint(0, size[0])
        j = np.random.randint(0, size[1])
        k = np.random.randint(0, size[2])
        d = np.random.randint(0, size[3])
        
        # trail rotation
        new = (lt[i,j,k,d]+np.random.randint(1, N)) % N 
        
        # compute H (actually -beta*H)
        ip1 = i+1
        if ip1 == size[0]:
            ip1 = 0
        im1 = i-1
        if im1 == -1:
            im1 = size[0]-1
        jp1 = j+1
        if jp1 == size[1]:
            jp1 = 0
        jm1 = j-1
        if jm1 == -1:
            jm1 = size[1]-1
        kp1 = k+1
        if kp1 == size[2]:
            kp1 = 0
        km1 = k-1
        if km1 == -1:
            km1 = size[2]-1
        
        # H conventions:
        # xy plane: +(i,j,k,0) +(ip1,j,k,1) -(i,jp1,k,0) -(i,j,k,1)
        # yz plane: +(i,j,k,1) +(i,jp1,k,2) -(i,j,kp1,1) -(i,j,k,2)
        # zx plane: +(i,j,k,2) +(i,j,kp1,0) -(ip1,j,k,2) -(i,j,k,0)
        if d == 0:
            H_old = (Re[(+ lt[i,j,k,0]     - lt[i,j,k,1] 
                         + lt[ip1,j,k,1]   - lt[i,jp1,k,0]) % N]
                    +Re[(+ lt[i,j,k,2]     - lt[i,j,k,0]    
                         + lt[i,j,kp1,0]   - lt[ip1,j,k,2]) % N]
                    +Re[(+ lt[i,jm1,k,0]   - lt[i,jm1,k,1] 
                         + lt[ip1,jm1,k,1] - lt[i,j,k,0])   % N]
                    +Re[(+ lt[i,j,km1,2]   - lt[i,j,km1,0]   
                         + lt[i,j,k,0]     - lt[ip1,j,km1,2]) % N]) * K
            H_new = (Re[(+ new             - lt[i,j,k,1] 
                         + lt[ip1,j,k,1]   - lt[i,jp1,k,0]) % N]
                    +Re[(+ lt[i,j,k,2]     - new   
                         + lt[i,j,kp1,0]   - lt[ip1,j,k,2]) % N]
                    +Re[(+ lt[i,jm1,k,0]   - lt[i,jm1,k,1] 
                         + lt[ip1,jm1,k,1] - new          ) % N]
                    +Re[(+ lt[i,j,km1,2]   - lt[i,j,km1,0]   
                         + new             - lt[ip1,j,km1,2]) % N]) * K
        elif d == 1:
            H_old = (Re[(+ lt[i,j,k,1]     - lt[i,j,k,2] 
                         + lt[i,jp1,k,2]   - lt[i,j,kp1,1]) % N]
                    +Re[(+ lt[i,j,k,0]     - lt[i,j,k,1]    
                         + lt[ip1,j,k,1]   - lt[i,jp1,k,0]) % N]
                    +Re[(+ lt[i,j,km1,0]   - lt[i,j,km1,2] 
                         + lt[i,jp1,km1,2] - lt[i,j,k,1])   % N]
                    +Re[(+ lt[im1,j,k,0]   - lt[im1,j,k,1]   
                         + lt[i,j,k,1]     - lt[im1,jp1,k,0]) % N]) * K
            H_new = (Re[(+ new             - lt[i,j,k,2] 
                         + lt[i,jp1,k,2]   - lt[i,j,kp1,1]) % N]
                    +Re[(+ lt[i,j,k,0]     - new    
                         + lt[ip1,j,k,1]   - lt[i,jp1,k,0]) % N]
                    +Re[(+ lt[i,j,km1,0]   - lt[i,j,km1,2] 
                         + lt[i,jp1,km1,2] - new        )   % N]
                    +Re[(+ lt[im1,j,k,0]   - lt[im1,j,k,1]   
                         + new             - lt[im1,jp1,k,0]) % N]) * K
        elif d == 2:
            H_old = (Re[(+ lt[i,j,k,2]     - lt[i,j,k,0] 
                         + lt[i,j,kp1,0]   - lt[ip1,j,k,2]) % N]
                    +Re[(+ lt[i,j,k,1]     - lt[i,j,k,2]    
                         + lt[i,jp1,k,2]   - lt[i,j,kp1,1]) % N]
                    +Re[(+ lt[im1,j,k,2]   - lt[im1,j,k,0] 
                         + lt[im1,j,kp1,0] - lt[i,j,k,2])   % N]
                    +Re[(+ lt[i,jm1,k,1]   - lt[i,jm1,k,2]   
                         + lt[i,j,k,2]     - lt[i,jm1,kp1,1]) % N]) * K
            H_new = (Re[(+ new             - lt[i,j,k,0] 
                         + lt[i,j,kp1,0]   - lt[ip1,j,k,2]) % N]
                    +Re[(+ lt[i,j,k,1]     - new    
                         + lt[i,jp1,k,2]   - lt[i,j,kp1,1]) % N]
                    +Re[(+ lt[im1,j,k,2]   - lt[im1,j,k,0] 
                         + lt[im1,j,kp1,0] - new        )   % N]
                    +Re[(+ lt[i,jm1,k,1]   - lt[i,jm1,k,2]   
                         + new             - lt[i,jm1,kp1,1]) % N]) * K
        
        # debugging
        if debug == True:
            print lt
            print "H_old=", H_old, "H_new=", H_new
            raw_input("press any key to continue...")

        # accept or decline
        if H_new >= H_old:
            lt[i,j,k,d] = new 
        else:
            rand = np.random.rand()
            if rand < np.exp(H_new - H_old):
                lt[i,j,k,d] = new
# END 3d LGT metropolis 



def metropolis_4dLGT(lt, K, N=2):
    size = lt.shape
    Ns = np.prod(size)
    
    # create an array to store the real part for an argument
    Re = np.cos(np.arange(N) * 2*np.pi/N)

    for i_step in xrange(Ns):
        # choose a random link
        i = np.random.randint(0, size[0])
        j = np.random.randint(0, size[1])
        k = np.random.randint(0, size[2])
        l = np.random.randint(0, size[3])
        d = np.random.randint(0, size[4])
        
        # trail rotation
        new = (lt[i,j,k,l,d]+np.random.randint(1, N)) % N 
        
        # compute H (actually -beta*H)
        ip1 = i+1
        if ip1 == size[0]:
            ip1 = 0
        im1 = i-1
        if im1 == -1:
            im1 = size[0]-1
        jp1 = j+1
        if jp1 == size[1]:
            jp1 = 0
        jm1 = j-1
        if jm1 == -1:
            jm1 = size[1]-1
        kp1 = k+1
        if kp1 == size[2]:
            kp1 = 0
        km1 = k-1
        if km1 == -1:
            km1 = size[2]-1
        lp1 = l+1
        if lp1 == size[3]:
            lp1 = 0
        lm1 = l-1
        if lm1 == -1:
            lm1 = size[3]-1
        
        # H conventions:
        # xy plane: +(i,j,k,l,0) +(ip1,j,k,l,1) -(i,jp1,k,l,0) -(i,j,k,l,1)
        # yz plane: +(i,j,k,l,1) +(i,jp1,k,l,2) -(i,j,kp1,l,1) -(i,j,k,l,2)
        # zx plane: +(i,j,k,l,2) +(i,j,kp1,l,0) -(ip1,j,k,l,2) -(i,j,k,l,0)
        # xw plane: +(i,j,k,l,0) +(ip1,j,k,l,3) -(i,j,k,lp1,0) -(i,j,k,l,3)
        # wy plane: +(i,j,k,l,3) +(i,j,k,lp1,1) -(i,jp1,k,l,3) -(i,j,k,l,1)
        # zw plane: +(i,j,k,l,2) +(i,j,kp1,l,3) -(i,j,k,wp1,2) -(i,j,k,l,3)
        if d == 0:
            H_old = (Re[(+ lt[i,j,k,l,0]     - lt[i,j,k,l,1]          # xy
                         + lt[ip1,j,k,l,1]   - lt[i,jp1,k,l,0]) % N]  
                    +Re[(+ lt[i,j,k,l,2]     - lt[i,j,k,l,0]          # zx
                         + lt[i,j,kp1,l,0]   - lt[ip1,j,k,l,2]) % N]
                    +Re[(+ lt[i,j,k,l,0]     - lt[i,j,k,l,3]          # xw
                         + lt[ip1,j,k,l,3]   - lt[i,j,k,lp1,0]) % N]
                    +Re[(+ lt[i,jm1,k,l,0]   - lt[i,jm1,k,l,1] 
                         + lt[ip1,jm1,k,l,1] - lt[i,j,k,l,0])   % N]
                    +Re[(+ lt[i,j,km1,l,2]   - lt[i,j,km1,l,0]   
                         + lt[i,j,k,l,0]     - lt[ip1,j,km1,l,2]) % N]
                    +Re[(+ lt[i,j,k,lm1,0]   - lt[i,j,k,lm1,3] 
                         + lt[ip1,j,k,lm1,3] - lt[i,j,k,l,0])   % N]) * K
            H_new = (Re[(+ new               - lt[i,j,k,l,1] 
                         + lt[ip1,j,k,l,1]   - lt[i,jp1,k,l,0]) % N]
                    +Re[(+ lt[i,j,k,l,2]     - new   
                         + lt[i,j,kp1,l,0]   - lt[ip1,j,k,l,2]) % N]
                    +Re[(+ new               - lt[i,j,k,l,3]          # xw
                         + lt[ip1,j,k,l,3]   - lt[i,j,k,lp1,0]) % N]
                    +Re[(+ lt[i,jm1,k,l,0]   - lt[i,jm1,k,l,1] 
                         + lt[ip1,jm1,k,l,1] - new          ) % N]
                    +Re[(+ lt[i,j,km1,l,2]   - lt[i,j,km1,l,0]   
                         + new               - lt[ip1,j,km1,l,2]) % N])
                    +Re[(+ lt[i,j,k,lm1,0]   - lt[i,j,k,lm1,3] 
                         + lt[ip1,j,k,lm1,3] - new          ) % N]) * K
        elif d == 1:
            H_old = (Re[(+ lt[i,j,k,l,1]     - lt[i,j,k,l,2]         # yz 
                         + lt[i,jp1,k,l,2]   - lt[i,j,kp1,l,1]) % N]
                    +Re[(+ lt[i,j,k,l,0]     - lt[i,j,k,l,1]         # xy
                         + lt[ip1,j,k,l,1]   - lt[i,jp1,k,l,0]) % N]
                    +Re[(+ lt[i,j,k,l,3]     - lt[i,j,k,l,1]         # wy
                         + lt[i,j,k,lp1,1]   - lt[i,jp1,k,l,3]) % N]
                    +Re[(+ lt[i,j,km1,l,0]   - lt[i,j,km1,l,2] 
                         + lt[i,jp1,km1,l,2] - lt[i,j,k,l,1])   % N]
                    +Re[(+ lt[im1,j,k,l,0]   - lt[im1,j,k,l,1]   
                         + lt[i,j,k,l,1]     - lt[im1,jp1,k,l,0]) % N]
                    +Re[(+ lt[i,j,k,lm1,3]   - lt[i,j,k,lm1,1]       # wy
                         + lt[i,j,k,l,1]     - lt[i,jp1,k,lm1,3]) % N]) * K
            H_new = (Re[(+ new               - lt[i,j,k,l,2] 
                         + lt[i,jp1,k,l,2]   - lt[i,j,kp1,l,1]) % N]
                    +Re[(+ lt[i,j,k,l,0]     - new    
                         + lt[ip1,j,k,l,1]   - lt[i,jp1,k,l,0]) % N]
                    +Re[(+ lt[i,j,k,l,3]     - new                   # wy
                         + lt[i,j,k,lp1,1]   - lt[i,jp1,k,l,3]) % N]
                    +Re[(+ lt[i,j,km1,l,0]   - lt[i,j,km1,l,2] 
                         + lt[i,jp1,km1,l,2] - new        )   % N]
                    +Re[(+ lt[im1,j,k,l,0]   - lt[im1,j,k,l,1]   
                         + new               - lt[im1,jp1,k,l,0]) % N])
                    +Re[(+ lt[i,j,k,lm1,3]   - lt[i,j,k,lm1,1]       # wy
                         + new               - lt[i,jp1,k,lm1,3]) % N]) * K
        elif d == 2:
            H_old = (Re[(+ lt[i,j,k,l,2]     - lt[i,j,k,l,0]         # zx 
                         + lt[i,j,kp1,l,0]   - lt[ip1,j,k,l,2]) % N]
                    +Re[(+ lt[i,j,k,l,1]     - lt[i,j,k,l,2]         # yz
                         + lt[i,jp1,k,l,2]   - lt[i,j,kp1,l,1]) % N]
                    +Re[(+ lt[i,j,k,l,2]     - lt[i,j,k,l,3]         # zw 
                         + lt[i,j,kp1,l,3]   - lt[i,j,k,lp1,2]) % N]
                    +Re[(+ lt[im1,j,k,l,2]   - lt[im1,j,k,l,0] 
                         + lt[im1,j,kp1,l,0] - lt[i,j,k,l,2])   % N]
                    +Re[(+ lt[i,jm1,k,l,1]   - lt[i,jm1,k,l,2]   
                         + lt[i,j,k,l,2]     - lt[i,jm1,kp1,l,1]) % N]
                    +Re[(+ lt[i,j,k,lm1,2]   - lt[i,j,k,lm1,3]       # zw 
                         + lt[i,j,kp1,lm1,3] - lt[i,j,k,l,2]) % N]) * K
            H_new = (Re[(+ new               - lt[i,j,k,l,0] 
                         + lt[i,j,kp1,l,0]   - lt[ip1,j,k,l,2]) % N]
                    +Re[(+ lt[i,j,k,l,1]     - new    
                         + lt[i,jp1,k,l,2]   - lt[i,j,kp1,l,1]) % N]
                    +Re[(+ new               - lt[i,j,k,l,3]         # zw 
                         + lt[i,j,kp1,l,3]   - lt[i,j,k,lp1,2]) % N]
                    +Re[(+ lt[im1,j,k,l,2]   - lt[im1,j,k,l,0] 
                         + lt[im1,j,kp1,l,0] - new            ) % N]
                    +Re[(+ lt[i,jm1,k,l,1]   - lt[i,jm1,k,l,2]   
                         + new               - lt[i,jm1,kp1,l,1]) % N]
                    +Re[(+ lt[i,j,k,lm1,2]   - lt[i,j,k,lm1,3]       # zw 
                         + lt[i,j,kp1,lm1,3] - new            ) % N]) * K
         elif d == 3:
            H_old = (Re[(+ lt[i,j,k,l,0]     - lt[i,j,k,l,3]         # xw
                         + lt[ip1,j,k,l,3]   - lt[i,j,k,lp1,0]) % N]
                    +Re[(+ lt[i,j,k,l,3]     - lt[i,j,k,l,1]         # wy
                         + lt[i,j,k,lp1,1]   - lt[i,jp1,k,l,3]) % N]
                    +Re[(+ lt[i,j,k,l,2]     - lt[i,j,k,l,3]         # zw 
                         + lt[i,j,kp1,l,3]   - lt[i,j,k,lp1,2]) % N]
                    +Re[(+ lt[im1,j,k,l,0]   - lt[im1,j,k,l,3]       # xw
                         + lt[i,j,k,l,3]     - lt[i,j,k,l,0]) % N]
                    +Re[(+ lt[i,jm1,k,l,3]   - lt[i,jm1,k,l,1]       # wy
                         + lt[i,jm1,k,lp1,1] - lt[i,j,k,l,3]) % N]
                    +Re[(+ lt[i,j,km1,l,2]   - lt[i,j,km1,l,3]       # zw 
                         + lt[i,j,k,l,3]     - lt[i,j,km1,lp1,2]) % N]) * K
            H_new = (Re[(+ lt[i,j,k,l,0]     - new                   # xw
                         + lt[ip1,j,k,l,3]   - lt[i,j,k,lp1,0]) % N]
                    +Re[(+ new               - lt[i,j,k,l,1]         # wy
                         + lt[i,j,k,lp1,1]   - lt[i,jp1,k,l,3]) % N]
                    +Re[(+ lt[i,j,k,l,2]     - new                   # zw 
                         + lt[i,j,kp1,l,3]   - lt[i,j,k,lp1,2]) % N]
                    +Re[(+ lt[im1,j,k,l,0]   - lt[im1,j,k,l,3]       # xw
                         + new               - lt[i,j,k,l,0]) % N]
                    +Re[(+ lt[i,jm1,k,l,3]   - lt[i,jm1,k,l,1]       # wy
                         + lt[i,jm1,k,lp1,1] - new            ) % N]
                    +Re[(+ lt[i,j,km1,l,2]   - lt[i,j,km1,l,3]       # zw 
                         + new               - lt[i,j,km1,lp1,2]) % N]) * K
        
        # debugging
        if debug == True:
            print lt
            print "H_old=", H_old, "H_new=", H_new
            raw_input("press any key to continue...")

        # accept or decline
        if H_new >= H_old:
            lt[i,j,k,l,d] = new 
        else:
            rand = np.random.rand()
            if rand < np.exp(H_new - H_old):
                lt[i,j,k,l,d] = new
# END 4d LGT metropolis 
