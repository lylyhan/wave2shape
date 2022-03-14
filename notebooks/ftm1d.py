def getsigma(m,p,s11):        
    sigma = s11 * (1 + p * (m**2 - 1))
    return sigma

def getomega(m,p,D,w11,s11):
    omega_sq = D**2 * w11**2 * m**2 + (s11**2 * (1 - p)**2 + w11**2 * (1 - D**2)) - s11**2 * (1 - p)**2
    return np.sqrt(omega_sq)
    

def getk(m,omega,f,x):
    k = f * np.sin(m * np.pi*x)/omega #assuming x1,x2 at center of the surface
    #print(np.sum(k))
    return k

#calculate integral and approximate excitation function with gaussian distribution
def getf(m,l,tau,mode='gaus'): #calculate the f_m which takes spatial component of excitation.
    #trapezoid rule to integrate f(x)sin(mpix) from 0 to l
    #(f(a)+f(b))*(b-a)/2
    integral = 0
    h = l/tau
    if mode == "gaus":
        x = approxnorm(l,l/2,0.4,tau) #f(x)
    elif mode == "tri":
        x = np.arange(0,tau+1,1)*h
        x = np.minimum(2-2/l*x,2*x/l)
    elif mode == "delta":
        x = np.zeros(tau+1)
        x[tau//2] = 1
    elif mode == "random":
        x = np.random.rand(tau+1)

    for i in range(tau):
        #x(i+2)
        #print(x.shape,x[0],x[0,1])
        integral = integral + (x[i] * np.sin(m * np.pi * i * h/l) + x[i+1] * np.sin(m * np.pi * (i + 1) * h/l))*h/2
    integral = integral*2/l
    return integral


def approxnorm(l,mu,s,tau): #normal distribution simulating fx 
    h = l/tau
    #x = np.zeros((1,tau + 1))
    x = []
    for i in range(tau+1):
        #x[i] = 1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2)
        x.append(1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2))

    return x