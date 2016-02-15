import numpy as np
import theano.tensor as T
import theano.sandbox.linalg as Tlina


#normalise vector x
def normed(x):
    return x / np.linalg.norm(x)


#project v onto u
def project(u,v):
    return v.T.dot(u) / u.T.dot(u) * u


#Gram-Schmidt orthonormalise n x d matrix V
def gsortho(V):
    d = V.shape[1]
    for i in range(0,d):
        #normalise
        V[:,i] = normed(V[:,i])
        #remove components in i-th direction
        for j in range(i+1,d):
            V[:,j] = V[:,j] - project(V[:,i],V[:,j])
    return V


#returns basis of A's null space
def null(A, eps=1e-15):
    A = A.T
    #svd
    u, s, v = np.linalg.svd(A)
    #dimension of null space
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    #select columns/rows corresponding to v
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = np.compress(null_mask, v, axis=0)
    return null_space.T


#complements w to a full basis
def complementbasis(w):
    V = np.concatenate((w,null(w)), axis=1)
    return gsortho(V) #assumes provided vectors w were already orthonormal


#angle between u and v in radians
def angle(u, v):
    return np.asscalar(np.arccos(np.clip(np.dot(normed(u).T, normed(v)), -1 , 1)))


#linesearch to hand over to pymanopt
class linesearch(object):
    def __init__(self, minstepsize=1e-16):
        self._contraction_factor = .7
        self._initial_stepsize = 1
        self._minstepsize = minstepsize

    def search(self, objective, man, x, d, f0, df0):
        norm_d = man.norm(x, d)
        alpha = self._initial_stepsize / norm_d

        newx = man.retr(x, alpha * d)
        newf = objective(newx)

        #while there is no decrease, i.e. step too large
        while newf > f0 and alpha * norm_d > self._minstepsize:
            alpha *= self._contraction_factor
            newx = man.retr(x, alpha * d)
            newf = objective(newx)
        if newf > f0:
            alpha = 0
            newx = x
        stepsize = alpha * norm_d
        return stepsize, newx


#generate timeseries tensor Ftw from given data matrix F (cf. Section III.A.)
def genToyTimeseriesTensor(F,fs,n,omega1,omega2):
    #random (time)series
    d,m = F.shape
    Ftw = np.random.randn(d,m,n)

    #frequency range
    a = min([k for k in range(1,int(np.ceil(n/2))+1) if k*fs/n > omega1])-1
    b = max([k for k in range(1,int(np.ceil(n/2))+1) if k*fs/n <= omega2])

    #loop through trials x channels and match log-bandpower
    for trial in range(0,m):
        for channel in range(0,d):
            x = Ftw[channel,trial,:]
            #center, Hanning window, fft
            hanning = 0.5*(1-np.cos( (2*np.pi*np.arange(0,n)) / (n-1) ))
            prevmean = np.mean(x)
            x = (x - prevmean) * hanning
            fft = np.fft.rfft(x)

            curbp = np.mean( np.log( np.abs(fft[a:b]) ) - np.log(n) )
            aimbp = F[channel,trial]

            lbd = np.exp(aimbp-curbp)

            #bump each coefficient by factor lbd
            for f in range(0,len(fft)):
                fft[f] = lbd*fft[f]

            hanning[hanning == 0] = 1
            Ftw[channel,trial,:] = np.fft.irfft(fft) / hanning + prevmean

    return Ftw


#set up MERLiNbp objective function and its gradient as theano graph
#(cf. Algorithm 4; optional plus=True for MERLiNbpicoh variant)
#s    m x 1
#vFr  m x n'
#vFi  m x n'
#Fr   d x (m*n')
#Fi   d x (m*n')
def getMERLiNbpObjective(s,vFi,vFr,Fi,Fr,n,plus=False):
    w = T.matrix('w') #d x 1
    m = s.shape[0]

    #linear combination
    wFr = T.reshape(w.T.dot(Fr), (m,-1)) #m x n'
    wFi = T.reshape(w.T.dot(Fi), (m,-1)) #m x n'

    #replace zeros, since we're taking logs
    unzero = lambda x: T.switch(T.eq(x, 0), 1, x)

    #bandpower
    bp = lambda re, im: T.reshape(T.mean(T.log( unzero(T.sqrt(re*re + im*im)) )-T.log(n), axis=1), (m,1))
    wFbp = bp(wFr,wFi) #m x 1
    vFbp = bp(vFr,vFi) #m x 1

    #centering matrix
    I = T.eye(m,m)
    H = I - T.mean(I)
    #column-centered data
    X = H.dot( T.concatenate([s,vFbp,wFbp], axis=1) ) #m x 3

    #covariance matrix
    S = X.T.dot(X) / (m-1)
    #precision matrix
    prec = Tlina.matrix_inverse(S)

    #objective and gradient for MERLiNbpicoh
    if plus:
        #complex row-wise vdot
        #(x+yi)(u+vi) = (xu-yv)+(xv+yu)i
        #vdot i.e. -v instead of +v
        vdot  = lambda x,y,u,v:  x*u+y*v
        vdoti = lambda x,y,u,v: -x*v+y*u
        cross  = lambda x,y,u,v: T.sum(vdot(x,y,u,v), axis=0) / m
        crossi = lambda x,y,u,v: T.sum(vdoti(x,y,u,v), axis=0) / m
        sqrtcross = lambda x,y: T.sqrt(cross(x,y,x,y)+crossi(x,y,x,y))
        icoherency = crossi(vFr,vFi,wFr,wFi) / ( sqrtcross(vFr,vFi)*sqrtcross(wFr,wFi) ) #n'
        objective = T.abs_(T.sum(icoherency))*T.abs_(prec[1,2])-T.abs_(prec[0,2])
    #objective and gradient for MERLiNb
    else:
        objective = T.abs_(prec[1,2])-T.abs_(prec[0,2])

    #return compiled function
    return objective, w


#set up MERLiNbpicoh objective function and its gradient as theano graph
#(cf. Algorithm 5)
def getMERLiNbpicohObjective(s,vFi,vFr,Fi,Fr,n):
    return getMERLiNbpObjective(s,vFi,vFr,Fi,Fr,n,plus=True)
