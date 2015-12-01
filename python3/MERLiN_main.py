from MERLiN_helper import *
import numpy as np


#Stiefel gradient ascent (cf. Algorithm 1)
'''
Input
    f, fprime: objective function and its gradient as theano functions
    w: initial point
Optional input
    tol: tolerance for stopping criterion
    maxsteps: maximum number of Stiefel gradient ascent steps
    lbd: initial step size
Output
    w: found solution after maxsteps steps or when the stopping criterion was met
    converged: whether the stopping criterion was met
    curob: value of f at w
'''
def stiefasc(f,fprime,w,tol=1e-16,maxsteps=500,lbd=1):
    converged = 0
    curob = f(w)
    for k in range(0,maxsteps):
        #while there is no increase, i.e. step too large
        while f(stiefel_update(w, fprime(w), lbd)) < curob:
            lbd = lbd*.5
        w = stiefel_update(w, fprime(w), lbd)

        newob = f(w)

        #'converged'?
        if np.abs(curob-newob) < tol:
            converged = 1
            break

        curob = newob
    return w, converged, curob


#MERLiN (cf. Algorithm 2)
'''
Input
    S: (m x 1) vector of samples of S
    F: (d x m) matrix of linear mixture samples
    v: (d x 1) vector corresponding to C1 in S->C1
Output
    w: found (d x 1) vector
    converged: whether the stopping criterion of Stiefel gradient ascent was met
    curob: value of f at w
'''
def MERLiN(S,F,v):
    #set C
    C = F.T.dot(v)
    #replace F, i.e. remove v signal
    P = complementbasis(v)[:,1:].T
    F = P.dot(F)

    #set function and derivative
    func = getMERLiNObjective()
    f = lambda w: func(S,C,F,w)[0]
    fprime = lambda w: func(S,C,F,w)[1]

    #maximise f, fprime
    w0 = normed(np.random.randn(v.shape[0]-1,1))
    w, converged, curob = stiefasc(f,fprime,w0)

    return P.T.dot(w), converged, curob


#Generate synthetic dataset (cf. Algorithm 6)
'''
Input
    T: gaussian or binary dataset
    d: dimension
    m: number of samples
    a: noise parameter
    b: hidden confounding parameter
Output
    S: (m x 1) vector of samples of S
    F: (d x m) matrix of linear mixture samples
    v: (d x 1) vector corresponding to C1 in S->C1
    wG0: ground truth (d x 1) vector to recover C2
'''
def genDataset(T,d,m,a,b):
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
    C[0,:] =   C[0,:] + S[0] + b*h[0]
    C[1,:] = a*(C[1,:]-mu[1]) + mu[1] + C[0,:]
    C[2,:] =   C[2,:] + S[0]
    C[3,:] =   C[3,:] + b*h[0]

    #orthogonal mixing
    F = A.dot(C)

    return [S,F,v,wG0]