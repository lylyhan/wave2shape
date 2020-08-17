import numpy as np


#get omega, sigma, K
def getsigma(m1,m2,alpha,p,s11):        
    beta = alpha + 1/alpha
    sigma = s11 * (1 + p * ((m1**2 * alpha + m2**2/alpha) - beta))
    return sigma

def getomega(m1,m2,alpha,p,D,w11,s11):
    interm = m1**2 * alpha + m2**2/alpha
    beta = alpha + 1/alpha
    omega_sq = D**2 * w11**2 * interm**2 + interm * (s11**2 * (1 - p * beta)**2/beta + w11**2 * (1 - D**2 * beta**2)/beta) - s11**2 * (1 - p * beta)**2
  
    return np.sqrt(omega_sq)
    

def getk(m1,m2,omega,f1,f2):
    k = f1 * f2 * np.sin(m1 * np.pi/2) * np.sin(m2 * np.pi/2)/omega #assuming x1,x2 at center of the surface
    #print(np.sum(k))
    return k

#calculate integral and approximate excitation function with gaussian distribution
def getf(m,l,tau):
    #trapezoid rule to integrate f(x)sin(mpix) from 0 to l
    #(f(a)+f(b))*(b-a)/2
    integral = 0
    x = approxnorm(l,l/2,0.4,tau)
    h = l/tau
    for i in range(tau):
        #x(i+2)
        #print(x.shape,x[0],x[0,1])
        integral = integral + (x[i] * np.sin(m * np.pi * i * h/l) + x[i+1] * np.sin(m * np.pi * (i + 1) * h/l))*h/2
    integral = integral*2/l
    return integral

def get_impf(m,l,tau):
    return np.sin(m/2)*2/l

def approxnorm(l,mu,s,tau):
    h = l/tau
    #x = np.zeros((1,tau + 1))
    x = []
    for i in range(tau+1):
        #x[i] = 1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2)
        x.append(1/(s * np.sqrt(2*np.pi)) * np.exp(-0.5 * (i * h - mu)**2/s**2))
    return x

def getsounds_imp(m1,m2,w11,tau11,p,D,alpha,sr):
    l = np.pi
    s11 = -1/tau11

    sigma=np.zeros((m1,m2))
    omega=np.zeros((m1,m2))
    k=np.zeros((m1,m2))

    x1 = 1
    x2 = l*alpha/2


    for i in range(m1):
        for j in range(m2):
            sigma[i,j] = getsigma(i+1,j+1,alpha,p,s11)
            omega[i,j] = getomega(i+1,j+1,alpha,p, D,w11,s11)
            k[i,j] = getk(i+1,j+1,omega[i,j],getf(i+1,1,300),getf(j+1,alpha,300))
            #k[i,j] = get_del_k(i+1,j+1,omega[i,j],x1,x2,l,alpha)

            #print(get_impf(i+1,1,300),getf(i+1,1,300))
    #sr = 44100
    #print(omega,sigma,k)
    dur = 2**16

    y = []
    for t in range(dur):
        y.append(np.sum(np.sum(k * np.exp(sigma * t/sr) * np.sin(omega * t/sr))))
    return y

def getsounds_dif(m1,m2,w11,tau11,p,D,alpha,sr):
    l = np.pi
    s11 = -1/tau11

    sigma=np.zeros((m1,m2))
    omega=np.zeros((m1,m2))
    k=np.zeros((m1,m2))

    x1 = 1
    x2 = l*alpha/2


    for i in range(m1):
        for j in range(m2):
            sigma[i,j] = getsigma(i+1,j+1,alpha,p,s11)
            omega[i,j] = getomega(i+1,j+1,alpha,p, D,w11,s11)
            k[i,j] = getk(i+1,j+1,omega[i,j],getf(i+1,1,300),getf(j+1,alpha,300))
            #k[i,j] = get_del_k(i+1,j+1,omega[i,j],x1,x2,l,alpha)
    #print(omega,sigma,k)



    dur = 2**16
    y_1 = np.zeros((m1,m2))
    y_2 = np.zeros((m1,m2))
    x_1 = 0.0
    x_0 = 1
    y_iii = []
    for n in range(sr):
        
        #each sigma, omega corresponds to a matrix of modes, thus when updating need a matrix of y values
        ytemp = 2*np.exp(sigma/sr)*np.cos(omega/sr)*y_1-np.exp(2*sigma/sr)*y_2+np.exp(sigma/sr)*np.sin(omega/sr)*k*x_1
        #ytemp = 2*np.exp(sigma/sr)*np.cos(omega/sr)*y_1-np.exp(2*sigma/sr)*y_2+np.exp(sigma/sr)/omega*np.sin(omega/sr)*x_1 #with impulse
        y_iii.append(np.sum(np.sum(ytemp)))
     
        x_1 = x_0
        x_0 = 0.0
        y_2 = y_1
        y_1 = ytemp
            
    return y_iii

def getsounds_bil(m1,m2,w11,tau11,p,D,alpha,sr):
    l = np.pi
    s11 = -1/tau11

    sigma=np.zeros((m1,m2))
    omega=np.zeros((m1,m2))
    k=np.zeros((m1,m2))

    x1 = 1
    x2 = l*alpha/2


    for i in range(m1):
        for j in range(m2):
            sigma[i,j] = getsigma(i+1,j+1,alpha,p,s11)
            omega[i,j] = getomega(i+1,j+1,alpha,p, D,w11,s11)
            k[i,j] = getk(i+1,j+1,omega[i,j],getf(i+1,1,300),getf(j+1,alpha,300))
            #k[i,j] = get_del_k(i+1,j+1,omega[i,j],x1,x2,l,alpha)
    #print(omega,sigma,k)



    dur = 2**16
    y_1 = 0.0
    y_2 = 0.0
    x_1 = 0.0
    x_2 = 0.0
    sr= 44100
    c = sr*2

    a1 = c**2+sigma*sigma - 2*sigma*c + omega*omega
    a2 = 2*omega*omega - 2*c**2 + 2*sigma*sigma
    a3 = c**2 + sigma*sigma + 2*c*sigma + omega*omega
    x_0 = 1
    y_bt = []
    for n in range(dur):
        #each sigma, omega corresponds to a matrix of modes, thus when updating need a matrix of y values
        #ytemp = 1/(1-sigma+omega**2)*(x_0+2*x_1+x_2-(2*omega**2+2*sigma**2-2)*y_1-((sigma+1)**2+omega**2)*y_2)
        ytemp = (x_0 + 2*k*x_1 + x_2 - a2*y_1 - a3*y_2)/a1
        
        y_bt.append(np.sum(np.sum(ytemp)))
        x_2 = x_1
        x_1 = x_0
        x_0 = 0.0
        y_2 = y_1
        y_1 = ytemp

            
    return y_bt


def get_gaus_k(m1,m2,omega,f1,f2, x1,x2,l,alpha):
    l2 = l * alpha
    if x1 > l:
        x1 = 0
    if x2 > l2:
        x2 = 0
    k = f1 * f2 * np.sin(m1 * np.pi * x1/l) * np.sin(m2 * np.pi * x2/l2)/ omega #assuming x1,x2 at center of the surface
    k = np.round(k,10)
    return k


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

def getsounds_imp_gaus(m1,m2,r1,r2,w11,tau11,p,D,alpha,sr):
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

    #sr = 44100
    #print(omega,sigma,k)
    dur = 2**16

    y = []
    for t in range(dur):
        y.append(np.sum(np.sum(k * np.exp(sigma * t/sr) * np.sin(omega * t/sr))))
    return y



