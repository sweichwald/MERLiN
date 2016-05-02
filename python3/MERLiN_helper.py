import numpy as np
from scipy.special import betainc


def normed(x):
    # normalise vector x
    return x / np.linalg.norm(x)


def project(u, v):
    # project v onto u
    return v.T.dot(u) / u.T.dot(u) * u


def gsortho(V):
    # Gram-Schmidt orthonormalise n x d matrix V
    d = V.shape[1]
    for i in range(0, d):
        # normalise
        V[:, i] = normed(V[:, i])
        # remove components in i-th direction
        for j in range(i+1, d):
            V[:, j] = V[:, j] - project(V[:, i], V[:, j])
    return V


def null(A, eps=1e-15):
    # returns basis of A's null space
    A = A.T
    # svd
    u, s, v = np.linalg.svd(A)
    # dimension of null space
    padding = max(0, np.shape(A)[1]-np.shape(s)[0])
    # select columns/rows corresponding to v
    null_mask = np.concatenate(((s <= eps), np.ones((padding, ), dtype=bool)),
                               axis=0)
    null_space = np.compress(null_mask, v, axis=0)
    return null_space.T


def complementbasis(w):
    # complements w to a full basis
    V = np.concatenate((w, null(w)), axis=1)
    return gsortho(V)  # assumes provided vectors w were already orthonormal


def angle(u, v):
    # angle between u and v in radians
    return np.asscalar(np.arccos(np.clip(
        np.dot(normed(u).T, normed(v)), -1, 1)))


def genDataset(T, d, m, a, b, eye=False):
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
    if eye:
        A = np.eye(d)
    else:
        # generate random orthogonal d x d matrix
        A = gsortho(np.random.randn(d, d))

    # set v and wG0
    v = A[:, 0:1]
    wG0 = A[:, 1:2]

    # generate S vector
    if T is 'binary':
        S = np.random.randint(low=1, high=3, size=(m, 1))*2-3
    elif T is 'gaussian':
        S = np.random.randn(m, 1)

    # generate hidden confounder
    h = np.random.randn(m, 1) + np.random.randn(1, 1)

    # generate mean for each Ci
    mu = np.random.randn(d, 1)

    # SEM
    C = np.random.randn(d, m)
    C = C + mu
    C[0:1, :] =    C[0:1, :] + S.T + b*h.T
    C[1:2, :] = a*(C[1:2, :]-mu[1]) + mu[1] + C[0:1, :]
    C[2:3, :] =    C[2:3, :] + S.T
    C[3:4, :] =    C[3:4, :] + b*h.T

    # orthogonal mixing
    F = A.dot(C)

    return [S, F, v, wG0]


def genToyTimeseriesTensor(F, fs, n, omega1, omega2):
    # generate timeseries tensor Ftw from given F (cf. Section III.A.)
    # random (time)series
    d, m = F.shape
    Ftw = np.random.randn(d, m, n)

    # frequency range
    a = min([k for k in range(1, int(np.ceil(n/2))+1) if k*fs/n > omega1])-1
    b = max([k for k in range(1, int(np.ceil(n/2))+1) if k*fs/n <= omega2])

    # loop through trials x channels and match log-bandpower
    for trial in range(0, m):
        for channel in range(0, d):
            x = Ftw[channel, trial, :]
            # center, Hanning window, fft
            hanning = 0.5*(1-np.cos((2*np.pi*np.arange(0, n)) / (n-1)))
            prevmean = np.mean(x)
            x = (x - prevmean) * hanning
            fft = np.fft.rfft(x)

            curbp = np.mean(np.log(np.abs(fft[a:b])) - np.log(n))
            aimbp = F[channel, trial]

            lbd = np.exp(aimbp-curbp)

            # bump each coefficient by factor lbd
            for f in range(0, len(fft)):
                fft[f] = lbd*fft[f]

            hanning[hanning == 0] = 1
            Ftw[channel, trial, :] = np.fft.irfft(fft) / hanning + prevmean

    return Ftw


# performance measures (cf. Section III.B.)
'''
Input to the following two functions
    wG0: ground truth vector to compare against
    w: vector to assess
Output
    probability of a better vector than w
        or
    angular distance between wG0 and w
'''
def pobv(wG0, w):
    # probability of a better vector
    a = (wG0.shape[0]-1)/2
    b = 0.5
    # ensure both vectors are normed -> r=1
    wG0 = normed(wG0)
    w = normed(w)
    h = 1 - np.abs(wG0.T.dot(w))
    x = h*(2-h)
    # betainc(a,b,x) computes the regularized incomplete beta function I_x(a,b)
    # http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.special.betainc.html#scipy.special.betainc
    return betainc(a, b, x)[0, 0]

def andi(wG0, w):
    # angular distance
    return min(angle(wG0, w), angle(-wG0, w))
