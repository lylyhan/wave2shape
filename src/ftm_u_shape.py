import numpy as np
import soundfile

#ending audio would be 2**15 length, sampling rate is 22050

## helper function 
def approxnorm(l,mu,s,tau):
    h = l/tau
    #x = np.zeros((1,tau + 1))
    x = []
    for i in range(tau+1):
        #x[i] = 1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2)
        x.append(1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2))
    return x

## calculate excitation function
#calculate integral and approximate excitation function with gaussian distribution
def get_gaus_f(m,l,tau,r):
    #trapezoid rule to integrate f(x)sin(mpix) from 0 to l
    #(f(a)+f(b))*(b-a)/2
    integral = 0
    x = approxnorm(l,l*r,0.4,tau)
    h = l/tau
    for i in range(tau):
        #x(i+2)
        #print(x.shape,x[0],x[0,1])
        integral = integral + (x[i] * np.sin(m * np.pi * i * h/l) + x[i+1] * np.sin(m * np.pi * (i + 1) * h/l))*h/2
    integral = integral*2/l
    return integral

## calculate omega, sigma, K
def getsigma(m1,m2,alpha,p,s11):        
    beta = alpha + 1/alpha
    sigma = s11 * (1 + p * ((m1**2 * alpha + m2**2/alpha) - beta))
    return sigma

def getomega(m1,m2,alpha,p,D,w11,s11):
    interm = m1**2 * alpha + m2**2/alpha
    beta = alpha + 1/alpha
    omega_sq = D**2 * w11**2 * interm**2 + interm * (s11**2 * (1 - p * beta)**2/beta + w11**2 * (1 - D**2 * beta**2)/beta) - s11**2 * (1 - p * beta)**2
  
    return np.sqrt(omega_sq)

#k is dependent on striking position as well
def get_gaus_k(m1,m2,omega,f1,f2, x1,x2,l,alpha):
    l2 = l * alpha
    if x1 > l:
        x1 = 0
    if x2 > l2:
        x2 = 0
    k = f1 * f2 * np.sin(m1 * np.pi * x1/l) * np.sin(m2 * np.pi * x2/l2)/ omega #assuming x1,x2 at center of the surface
    k = np.round(k,10)
    return k

## calculate the ending sound 
def getsounds_imp_gaus(m1,m2,r1,r2,w11,tau11,p,D,alpha,sr):
    """
    Calculate sound of a FTM 2D rectangular drum of shape {w,tau,p,D,alpha}
    excitation function:(Laplace) spatially gaussian, temporally delta
    modes: m1, m2 along each direction
    excitation position: r1, r2 
    """

    l = np.pi
    l2 = l*alpha
    s11 = -1/tau11

    sigma=np.zeros((m1,m2))
    omega=np.zeros((m1,m2))
    k=np.zeros((m1,m2))

    x1 = l*r1
    x2 = l2*r2


    for i in range(m1):
        for j in range(m2):
            sigma[i,j] = getsigma(i+1,j+1,alpha,p,s11)
            omega[i,j] = getomega(i+1,j+1,alpha,p, D,w11,s11)
            k[i,j] = get_gaus_k(i+1,j+1,omega[i,j],get_gaus_f(i+1,1,300,r1),get_gaus_f(j+1,alpha,300,r2),x1,x2,l,alpha) # the covered striking length is 1

    dur = 2**15

    y = []
    for t in range(dur):
        y.append(np.sum(np.sum(k * np.exp(sigma * t/sr) * np.sin(omega * t/sr))))

    y = y/max(y)


    return y

if __name__ == '__main__':
    sr=22050
    y = getsounds_imp_gaus(20,20,0.5,0.5,2000,0.1,0.2,0.1,0.7,sr)
    soundfile.write("./sound.wav",y,sr)






