from MERLiN_helper import gsortho, complementbasis, normed, angle, linesearch, getMERLiNbpicohObjective, getMERLiNbpObjective
import numpy as np
from scipy.special import betainc
import theano.tensor as T
from theano import function

from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent


#performance measures (cf. Section III.B.)
'''
Input to the following two functions
    wG0: ground truth vector to compare against
    w: vector to assess
Output
    probability of a better vector than w
        or
    angular distance between wG0 and w
'''
#probability of a better vector
def pobv(wG0,w):
    a = (wG0.shape[0]-1)/2
    b = 0.5
    #ensure both vectors are normed -> r=1
    wG0 = normed(wG0)
    w = normed(w)
    h = 1 - np.abs( wG0.T.dot(w) )
    x = h*(2-h)
    #betainc(a,b,x) computes the regularized incomplete beta function I_x(a,b)
    #http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.special.betainc.html#scipy.special.betainc
    return betainc(a,b,x)[0,0]
#angular distance
def andi(wG0, w):
    return min(angle(wG0,w), angle(-wG0,w))


'''
maximise using pymanopt
Input
    f, fprime: objective function and its gradient either both as
               regular functions or both as theano functions
    arg: None or theano variable that is the argument of f
    w0: initial vector
Optional input
    maxiter: maximum number of conjugate gradient steps
    tol: terminate if stepsize < tol
Output
    w: found solution after maxiter steps or meeting the stopping criterion
    converged: whether the stopping criterion was met
    curob: value of f at w
'''
def maximise(f,fprime,arg,w0,tol=1e-16,maxiter=500):
    manifold = Sphere(w0.shape[0])

    #MERLiNbp and MERLiNbpicoh
    if fprime is None:
        problem = Problem(man=manifold, cost=-f, arg=arg, verbosity=0)
    #MERLiN
    else:
        cost = lambda x: -f(x)
        egrad = lambda x: -fprime(x)
        problem = Problem(man=manifold, cost=cost, egrad=egrad, verbosity=0)

    solver = SteepestDescent(maxtime=float('inf'),maxiter=maxiter,mingradnorm=0,minstepsize=tol,
                             linesearch=linesearch(minstepsize=tol),logverbosity=1)
    w, info = solver.solve(problem, x=w0)
    converged = maxiter != info['final_values']['iterations']
    curob = -float(info['final_values']['f(x)'])
    return w, converged, curob


'''
MERLiN (cf. Algorithm 2)
Input
    S: (m x 1) vector of samples of S
    F: (d x m) matrix of linear mixture samples
    v: (d x 1) vector corresponding to C1 in S->C1
Output
    w: found (d x 1) vector
    converged: whether the stopping criterion was met
    curob: value of f at w
'''
def MERLiN(S,F,v):
    #set C
    C = F.T.dot(v)
    #replace F, i.e. remove v signal
    P = complementbasis(v)[:,1:].T
    F = P.dot(F)

    #set O,Q,R
    m = S.shape[0]
    H = np.eye(m) - np.ones((m,m))/m
    O = (( S.T.dot(H.dot(C.dot(C.T))) - C.T.dot(H.dot(C.dot(S.T))) ).dot(H)).dot(F.T)
    Q = (( S.T.dot(H.dot(C.dot(S.T))) - S.T.dot(H.dot(S.dot(C.T))) ).dot(H)).dot(F.T)
    r1 = (S.T.dot(H.dot(S.dot(C.T.dot(H.dot(C))))))*np.eye(m)
    r2 = S.T.dot(H.dot(C))*C.dot(S.T)
    r3 = S.T.dot(H.dot(C))*S.dot(C.T)
    r4 = C.T.dot(H.dot(C))*S.dot(S.T)
    r5 = (S.T.dot(H.dot(C))**2)*np.eye(m)
    r6 = S.T.dot(H.dot(S))*C.dot(C.T)
    R = F.dot(((H.dot( r1 + r2 + r3 - r4 - r5 - r6 )).dot(H)).dot(F.T))

    #set objective function and derivative
    f = lambda w: ( np.abs(Q.dot(w)) - np.abs(O.dot(w)) ) / np.abs( w.T.dot(R.dot(w)) )
    fprime = lambda w: np.asarray( ( np.abs(w.T.dot(R.dot(w)))*( np.sign(Q.dot(w))*Q.T-np.sign(O.dot(w))*O.T ) - np.sign(w.T.dot(R.dot(w)))*( np.abs(Q.dot(w)) - np.abs(O.dot(w)) )*(R+R.T).dot(w) ) / (np.abs(w.T.dot(R.dot(w)))**2) )

    #maximise
    w0 = normed(np.random.randn(v.shape[0]-1,1))
    w, converged, curob = maximise(f,fprime,None,w0)

    return P.T.dot(w), converged, curob


'''
preprocessing for bp algorithms (cf. Algorithm 3)
Input
    Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
    v: (d x 1) vector corresponding to C1 in S->C1
    fs: sampling rate
    omega1, omega2: low/high limit of desired frequency band
Output
    Vi, Vr: (m x n') matrix of imaginary/real part of the relevant n' fourier coefficients
            of the signal corresponding to v in m trials
    Fi, Fr: (d x m x n') tensor of imaginary/real part of the relevant n' fourier coefficients
            of each of the d channels in m trials
'''
def bpPreprocessing(Ftw,v,fs,omega1,omega2):
    d,m,n = Ftw.shape

    #frequency range
    a = min([k for k in range(1,int(np.ceil(n/2))+1) if k*fs/n > omega1])-1
    b = max([k for k in range(1,int(np.ceil(n/2))+1) if k*fs/n <= omega2])

    Vi = np.zeros([m,b-a])
    Vr = np.zeros([m,b-a])
    Fi = np.zeros([d,m,b-a])
    Fr = np.zeros([d,m,b-a])

    #hanning window
    hanning = 0.5*(1-np.cos( (2*np.pi*np.arange(0,n)) / (n-1) ))

    #loop through trials
    for trial in range(0,m):
        F = Ftw[:,trial,:] #d x n

        #extract v signal
        V = v.T.dot(F)

        #center, hanning window, fft
        V = (V - np.mean(V)) * hanning
        V = np.fft.rfft(V)[0,a:b]
        Vi[trial,:] = np.imag(V)
        Vr[trial,:] = np.real(V)

        #remove v signal
        P = complementbasis(v)[:,1:].T
        F = P.T.dot(P.dot(F))

        #loop through channels
        for channel in range(0,d):
            x = F[channel,:]
            x = (x - np.mean(x)) * hanning
            x = np.fft.rfft(x)[a:b]
            Fi[channel,trial,:] = np.imag(x)
            Fr[channel,trial,:] = np.real(x)

    return Vi, Vr, Fi, Fr


'''
MERLiNbp (cf. Algorithm 4)
Input
    S: (m x 1) vector of samples of S
    Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
    v: (d x 1) vector corresponding to C1 in S->C1
    fs: sampling rate
    omega1, omega2: low/high limit of desired frequency band
Optional input
    preprocessed: list of already preprocessed data Vi, Vr, Fi, Fr, n
Output
    w: found (d x 1) vector
    converged: whether the stopping criterion was met
    curob: value of f at w
'''
def MERLiNbp(S,Ftw,v,fs,omega1,omega2,preprocessed = False):
    if preprocessed:
        Vi, Vr, Fi, Fr, n = preprocessed
    else:
        Vi, Vr, Fi, Fr = bpPreprocessing(Ftw,v,fs,omega1,omega2)
        n = Ftw.shape[2]

    #rearrange (d x m x n') tensors Fi/Fr into (d x m*n') matrices
    d = Fi.shape[0]
    Fi = Fi.reshape(d,-1)
    Fr = Fr.reshape(d,-1)

    #get objective function and its argument as theano variables
    f, w = getMERLiNbpObjective(S,Vi,Vr,Fi,Fr,n)

    #random initial vector in orthogonal complement
    P = complementbasis(v)[:,1:].T
    w0 = np.random.randn(v.shape[0],1)
    w0 = normed( P.T.dot(P.dot(w0)) )

    #maximise f
    w, converged, curob = maximise(f,None,w,w0)

    return w, converged, curob


'''
MERLiNbpicoh (cf. Algorithm 5)
Input
    S: (m x 1) vector of samples of S
    Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
    v: (d x 1) vector corresponding to C1 in S->C1
    fs: sampling rate
    omega1, omega2: low/high limit of desired frequency band
Optional input
    preprocessed: list of already preprocessed data Vi, Vr, Fi, Fr, n
Output
    w: found (d x 1) vector
    converged: whether the stopping criterion was met
    curob: value of f at w
'''
def MERLiNbpicoh(S,Ftw,v,fs,omega1,omega2,preprocessed = False):
    if preprocessed:
        Vi, Vr, Fi, Fr, n = preprocessed
    else:
        Vi, Vr, Fi, Fr = bpPreprocessing(Ftw,v,fs,omega1,omega2)
        n = Ftw.shape[2]

    #rearrange (d x m x n') tensors Fi/Fr into (d x m*n') matrices
    d = Fi.shape[0]
    Fi = Fi.reshape(d,-1)
    Fr = Fr.reshape(d,-1)

    #get objective function and its argument as theano variables
    f, w = getMERLiNbpicohObjective(S,Vi,Vr,Fi,Fr,n)

    #random initial vector in orthogonal complement
    P = complementbasis(v)[:,1:].T
    w0 = np.random.randn(v.shape[0],1)
    w0 = normed( P.T.dot(P.dot(w0)) )

    #maximise f
    w, converged, curob = maximise(f,None,w,w0)

    return w, converged, curob


'''
Generate synthetic dataset (cf. Algorithm 6)
Input
    T: gaussian or binary dataset
    d: dimension
    m: number of samples
    a: noise parameter
    b: hidden confounding parameter
Optional input
    eye: random orthonormal mixing (False) or no/identity mixing (True)
Output
    S: (m x 1) vector of samples of S
    F: (d x m) matrix of linear mixture samples
    v: (d x 1) vector corresponding to C1 in S->C1
    wG0: ground truth (d x 1) vector to recover C2
'''
def genDataset(T,d,m,a,b,eye=False):
    if eye:
        A = np.eye(d)
    else:
        #generate random orthogonal d x d matrix
        A = gsortho( np.random.randn(d,d) )

    #set v and wG0
    v = A[:,0:1]
    wG0 = A[:,1:2]

    #generate S vector
    if T is 'binary':
        S = np.random.randint(low=1, high=3, size=(m,1))*2-3
    elif T is 'gaussian':
        S = np.random.randn(m,1)

    #generate hidden confounder
    h = np.random.randn(m,1) + np.random.randn(1,1)

    #generate mean for each Ci
    mu = np.random.randn(d,1)

    #SEM
    C = np.random.randn(d,m)
    C = C + mu
    C[0:1,:] =   C[0:1,:] + S.T + b*h.T
    C[1:2,:] = a*(C[1:2,:]-mu[1]) + mu[1] + C[0:1,:]
    C[2:3,:] =   C[2:3,:] + S.T
    C[3:4,:] =   C[3:4,:] + b*h.T

    #orthogonal mixing
    F = A.dot(C)

    return [S,F,v,wG0]
