import numpy as np
import scipy as sp
import theano.tensor as T
import theano.sandbox.linalg as Tlina
from theano import function
from scipy.special import betainc


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
    return gsortho(V) #assumes provided vectors w were already orthonormal, otherwise this would change these


#angle between u and v in radians
def angle(u, v):
    return np.asscalar(np.arccos(np.clip(np.dot(normed(u).T, normed(v)), -1 , 1)))


#from V take Stiefel ascent step defined by the gradient G and step size lbd
def stiefel_update(V, G, lbd):
    n = V.shape[0]
    p = V.shape[1]
    Vp = null(V)
    Z = np.bmat([ [ V.T.dot(G)-G.T.dot(V) , -G.T.dot(Vp) ] , [ Vp.T.dot(G) ,  np.zeros((n-p,n-p)) ] ])
    return np.bmat([[V, Vp]]).dot( sp.linalg.expm(lbd*Z).dot( np.eye(n,p) ) )


#set up MERLiN objective function and its gradient as theano function
#(cf. Algorithm 2)
def getMERLiNObjective():
    s = T.matrix('s') #m x 1
    c = T.matrix('c') #m x 1
    F = T.matrix('F') #(d-1) x m
    w = T.matrix('w') #(d-1) x 1
    m = s.shape[0]

    #linear combination
    wF = F.T.dot(w) #m x 1

    #centering matrix
    I = T.eye(m,m)
    H = I - T.mean(I)
    #column-centered data
    X = H.dot( T.concatenate([s,c,wF], axis=1) ) #m x 3

    #covariance matrix
    S = X.T.dot(X) / (m-1)
    #precision matrix
    prec = Tlina.matrix_inverse(S)

    #objective and gradient
    objective = T.abs_(prec[1,2])-T.abs_(prec[0,2])
    gradient = T.grad(objective, w)

    #return compiled function
    return function([s,c,F,w], [objective, gradient])