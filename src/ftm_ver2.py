"""
This script implements Rabenstein's linear drum model
1. drum model with normalized side length ratio/ side length, in difference equation form
2. drum model without normalization, in difference equation form
3. drum model without normalization, in impulse form
"""
import numpy as np
import math

def getsounds_dif_linear_normlized(m1,m2,x1,x2,h,tau11,w11,p,D,l0,alpha_side,sr):
    """
    This implements ivan's drum model with normalized side length and side 
    length ratio. The inverse SLT operation is done at the end of each second-
    -order filter, instead of being implied in the excitatino function yi. This 
    has proven to make a difference in the resulting sounds.
    """
    l2 = l0*alpha_side
    beta_side = alpha_side + 1/alpha_side

    mu = np.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = np.arange(1,m2+1)
    dur = 2**16
    Ts = 1/sr
    tau = 1/sr*np.arange(1,dur+1)

    n = mu**2*alpha_side+mu2**2/alpha_side #eta 
    n2 = n**2#(mu)**4+(mu2/alpha_side)**4
    s11 = -1/tau11
    alpha = -s11*(1+p*(n-beta_side))
    omega = np.sqrt(np.abs(D**2*w11**2*n2+n*(s11**2*(1-p*beta_side)**2 \
                                             /beta_side+w11**2*(1-D**2*beta_side**2)/ \
                                            beta_side) - s11**2*(1-p*beta_side)**2))
    K = np.sin(mu*x1*l0)*np.sin(mu2*x2*l2) #mu pi x / l 

    
    N = l0*l2/4
    c0 = -np.exp(-2*alpha*Ts)
    c1 = 2*np.exp(-alpha*Ts)*np.cos(omega*Ts)
    a1 = 1 #term a1 neither

    a0 = np.exp(-alpha*Ts)*np.sin(omega*Ts)/omega
    
    yi = h*np.sin(mu*x1*l0)*np.sin(mu2*x2*l2)
    

    y_i = np.zeros(c0.shape)
    y_ii = np.zeros(c0.shape)
    y = []
    summed = 0
    node1 = 0
    node2 = a1*yi+node1
    dur_excit = 1
    for i in range(dur):
        #signal forward pass
        y.append(np.sum(node2*K/N)) #sum over all mode

        #update
        y_ii = y_i
        y_i = node2  #continuing with excitation
        if i<dur_excit: #calculated for the current sample, but manifest in the next loop
            node1 = a0*yi+c0*y_ii+c1*y_i
        elif i>=dur_excit:
            a1 = 0
            node1 = c0*y_ii+c1*y_i


        node2 = node1
      
    return y


def getsounds_dif_linear_nonorm(m1,m2,x1,x2,h,tau11,w11,p,D,l0,alpha_side,sr):
    """
    This implements Rabenstein's drum model. The inverse SLT operation is done at the end of each second-
    -order filter, no normalization on length and side length ratio is done
    """
    l2 = l0*alpha_side
    beta_side = alpha_side + 1/alpha_side
    S = l0/np.pi*((D*w11*alpha_side)**2 + (p*alpha_side/tau11)**2)**0.25
    c_sq = (alpha_side*(1/beta_side-p**2*beta_side)/tau11**2 + alpha_side*w11**2*(1/beta_side-D**2*beta_side))*(l0/np.pi)**2
    T = c_sq 
    d1 = 2*(1-p*beta_side)/tau11
    d3 = -2*p*alpha_side/tau11*(l0/np.pi)**2 

    EI = S**4 
        
    mu = np.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = np.arange(1,m2+1)
    dur = 2**16
    Ts = 1/sr
    tau = 1/sr*np.arange(1,dur+1)

    n = (mu*np.pi/l0)**2+(mu2*np.pi/l2)**2 #eta 
    n2 = n**2 #(mu*np.pi/l0)**4+(mu2*np.pi/l2)**4 
    K = np.sin(mu*math.pi*x1)*np.sin(mu2*math.pi*x2) #mu pi x / l
    
    beta = EI*n2 + T*n #(m,1)
    alpha = (d1-d3*n)/2 # nonlinear
    omega = np.sqrt(np.abs(beta - alpha**2))
    N = l0*l2/4
    c0 = -np.exp(-2*alpha*Ts)
    c1 = 2*np.exp(-alpha*Ts)*np.cos(omega*Ts)
    a1 = 1 #term a1 neither

    a0 = np.exp(-alpha*Ts)*np.sin(omega*Ts)/omega
    #o_a = mu*math.pi*x1
    #o_a2 = mu2*math.pi*x2
    
    #derive mu domain of the plucked string yx extended to 2D

    #yi = (h/(mu*math.pi)*np.cos(o_a)-h/(mu*math.pi)**2/x1*np.sin(o_a)-2*h/(1-x1)/(mu*math.pi)*np.cos(mu*math.pi) + \
    # h*x1/(1-x1)/(mu*math.pi)*np.cos(o_a)-h/(1-x1)/(mu*math.pi)**2*np.sin(o_a)+h/(1-x1)/(mu*math.pi)*np.cos(o_a)) \
    #*(h/(mu2*math.pi)*np.cos(o_a2)-h/(mu2*math.pi)**2/x1*np.sin(o_a2)-2*h/(1-x2)/(mu2*math.pi)*np.cos(mu2*math.pi) + \
    # h*x2/(1-x2)/(mu2*math.pi)*np.cos(o_a2)-h/(1-x2)/(mu2*math.pi)**2*np.sin(o_a2)+h/(1-x2)/(mu2*math.pi)*np.cos(o_a2))
    
    yi = h*np.sin(mu*np.pi*x1)*np.sin(mu2*np.pi*x2)
     
    y_i = np.zeros(c0.shape)
    y_ii = np.zeros(c0.shape)
    y = []
    summed = 0
    node1 = 0
    node2 = a1*yi+node1
    dur_excit = 1
    for i in range(dur):
        #signal forward pass
        y.append(np.sum(node2*K/N)) #sum over all mode

        #update
        y_ii = y_i
        y_i = node2  #continuing with excitation
        if i<dur_excit: #calculated for the current sample, but manifest in the next loop
            node1 = a0*yi+c0*y_ii+c1*y_i
        elif i>=dur_excit:
            a1 = 0
            node1 = c0*y_ii+c1*y_i


        node2 = node1
      
    return y

def getsounds_imp_linear_nonorm(m1,m2,x1,x2,h,tau11,w11,p,D,l0,alpha_side,sr):
    l2 = l0*alpha_side
    beta_side = alpha_side + 1/alpha_side
    S = l0/np.pi*((D*w11*alpha_side)**2 + (p*alpha_side/tau11)**2)**0.25
    c_sq = (alpha_side*(1/beta_side-p**2*beta_side)/tau11**2 + alpha_side*w11**2*(1/beta_side-D**2*beta_side))*(l0/np.pi)**2
    T = c_sq 
    d1 = 2*(1-p*beta_side)/tau11
    d3 = -2*p*alpha_side/tau11*(l0/np.pi)**2 

    EI = S**4 

    mu = np.arange(1,m1+1) #(0,1,2,..,m-1)
    mu2 = np.arange(1,m2+1)
    dur = 2**16
    Ts = 1/sr
    tau = 1/sr*np.arange(1,dur+1)

    n = (mu*np.pi/l0)**2+(mu2*np.pi/l2)**2 #eta 
    n2 = n**2 #(mu*np.pi/l0)**4+(mu2*np.pi/l2)**4 
    K = np.sin(mu*math.pi*x1)*np.sin(mu2*math.pi*x2) #mu pi x / l

    beta = EI*n2 + T*n #(m,1)
    alpha = (d1-d3*n)/2 # nonlinear
    omega = np.sqrt(np.abs(beta - alpha**2))
    N = l0*l2/4
    yi = h*np.sin(mu*np.pi*x1)*np.sin(mu2*np.pi*x2)/omega


    time_steps = np.linspace(0,dur,dur)/sr
    y = np.exp(-alpha[:,None]*time_steps[None,:])*np.sin(omega[:,None]*time_steps[None,:]) 
    y = yi[:,None]*y #(m,) * (m,dur)
    y = np.sum(y*K[:,None]/N,axis=0) #impulse response itself

    return y