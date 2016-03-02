from MERLiN_helper import complementbasis
import numpy as np
from theano import function # TODO
import theano.tensor as T
import theano.compile.sharedvalue as TS
import theano.sandbox.linalg as Tlina
from pymanopt import Problem
from pymanopt.manifolds import Sphere, Euclidean, Product
from pymanopt.solvers import SteepestDescent


class MERLiN:
    '''
    MERLiN: Mixture Effect Recovery in Linear Networks

    References
        [1] MERLiN: Mixture Effect Recovery in Linear Networks, S Weichwald,
        M Grosse-Wentrup, A Gretton, arxiv.org/abs/1512.01255
    '''

    def __init__(self, verbosity=0):
        '''
        Instantiate MERLiN class.

        Input (default)
            - verbosity (0)
                level of detail of printed information (0 = silent,
                1 = verbose)
        '''
        self._problem_MERLiNbp_val = None
        self._problem_MERLiNbpicoh_val = None
        self._problem_MERLiNnlbp_val = None
        self._verbosity = verbosity

    def run(self, S, F, v=None, C=None, fs=None, omega=None,
            maxiter=500, tol=1e-10, variant='bp'):
        '''
        Run MERLiN algorithm.
        Whether to run a scalar variant, i.e. S -> C -> w'F, or a
        timeseries variant, i.e. S -> C -> bp(w'F) is determined by the
        dimensionality of the input F.

        Input (default)
            - S
                (m x 1) np.array that contains the samples of S
            - F
                either a (d x m) np.array that contains the linear mixture
                samples or a (d x m x n) np.array that contains the linearly
                mixed timeseries of length n (d channels, m trials)
            - v
                (d x 1) np.array holding the linear combination that
                extracts middle node C from F
            - C
                (m x 1) np.array that contains the samples of the middle
                node C
            - fs
                sampling rate in Hz
            - omega
                tuple of (low, high) cut-off of desired frequency band
            - maxiter (500)
                maximum iterations to run the optimisation algorithm for
            - tol (1e-16)
                terminate optimisation if step size < tol or grad norm < tol
            - variant ('bp')
                determines which MERLiN variant to use on timeseries data
                ('bp' = MERLiNbp algorithm ([1], Algorithm 4),
                 'bpicoh' = MERLiNbpicoh algorithm ([1], Algorithm 5),
                 'nlbp' = MERLiNnlbp)

        Output
            - w
                linear combination that was found and should extract the
                effect of C from F
            - converged
                boolean that indicates whether the stopping criterion was
                met before the maximum number of iterations was performed
            - curob
                objecive functions value at w
        '''
        self._S = S
        self._Forig = F
        self._fs = fs
        self._omega = omega
        self._d = F.shape[0]
        self._m = F.shape[1]

        # scalar or timeseries mode
        if F.ndim == 3:
            self._mode = 'timeseries'
            self._n = F.shape[2]
            if not (fs and omega):
                raise ValueError('Both the optional arguments fs and omega '
                                 'need to be provided.')
            if self._verbosity:
                print('Launching MERLiN' + variant + ' for iid sampled '
                      'timeseries chunks.')
        elif F.ndim == 2:
            self._mode = 'scalar'
            if self._verbosity:
                print('Launching MERLiN for iid sampled scalars.')
        else:
            raise ValueError('F needs to be a 2-dimensional numpy array '
                             '(iid sampled scalars) or a 3-dimensional '
                             'numpy array (iid sampled timeseries chunks).')

        self._prepare(v, C)

        if self._mode is 'scalar':
            problem = self._problem_MERLiN()
        elif variant is 'bp':
            problem = self._problem_MERLiNbp()
        elif variant is 'bpicoh':
            problem = self._problem_MERLiNbpicoh()
        elif variant is 'nlbp':
            problem = self._problem_MERLiNnlbp()
        else:
            raise NotImplementedError

        if variant is not 'nlbp':
            problem.manifold = Sphere(self._d)
        elif variant is 'nlbp':
            problem.manifold = Product([Sphere(self._d), Euclidean(1, 1), Euclidean(1, 1)])

        # choose best out of ten 10-step runs as initialisation
        solver = SteepestDescent(maxiter=10, logverbosity=1)
        res = [solver.solve(problem) for k in range(0, 10)]
        obs = [-r[1]['final_values']['f(x)'] for r in res]
        w0 = res[obs.index(max(obs))][0]

        solver = SteepestDescent(maxtime=float('inf'), maxiter=maxiter,
                                 mingradnorm=tol, minstepsize=tol,
                                 logverbosity=1)
        if self._verbosity:
            print('Running optimisation algorithm.')
        w, info = solver.solve(problem, x=w0)
        if variant is 'nlbp':
            w = w[0]
        converged = maxiter != info['final_values']['iterations']
        curob = -float(info['final_values']['f(x)'])
        if self._verbosity:
            print('DONE.')
        return self._P.T.dot(w), converged, curob

    def _prepare(self, v, C):
        '''
        Prepare the data for optimisation.
        Extract the middle node's samples or, if working with timeseries
        chunks, center, apply Hanning window, compute Fourier coefficients
        of desired frequency band, extract the middle node's coefficients.

        Sets/updates self._F, self._C, self._P

        In timeseries mode self._F and self._C are tuples of (d x m x n')
        or (m x n') np.arrays that contain the imaginary and real parts
        of the Fourier coefficients.

        Input
            - v
                Passed over from self.run(), either None or a (d x 1)
                np.array that corresponds to the linear combination
                that extracts middle node C from F
            - C
                Passed over from self.run(), either None or a (m x 1)
                np.array that contains the samples of the middle node C
        '''
        # conflicting input
        if (v is None and C is None) or (v is not None and C is not None):
            raise ValueError('Either C or v needs to be provided '
                             'but not both.')
        # C contains the samples
        elif C is not None:
            if self._verbosity:
                print('The scalar samples of the middle node were '
                      'provided.')
            self._P = np.eye(self._d)
            self._C = C
            self._F = self._Forig
        # extract the samples
        elif v is not None:
            self._d = self._d-1
            self._P = complementbasis(v)[:, 1:].T
            # scalar samples
            if self._mode is 'scalar':
                if self._verbosity:
                    print('Computing samples of the middle node '
                          'corresponding to the linear combination v and '
                          'removing its signal from F.')
                self._C = self._Forig.T.dot(v)
                self._F = self._P.dot(self._Forig)
            # timeseries chunks
            else:
                if self._verbosity:
                    print('Centering, applying Hanning window, computing '
                          'Fourier coefficients of desired frequency band, '
                          'extracting the coefficients corresponding to '
                          'the linear combination v and removing its signal '
                          'from F.')
                d, m, n = self._Forig.shape
                omega = self._omega
                fs = self._fs
                # frequency range
                a = min([k for k in range(1, int(np.ceil(n/2))+1)
                        if k*fs/n > omega[0]])-1
                b = max([k for k in range(1, int(np.ceil(n/2))+1)
                        if k*fs/n <= omega[1]])
                Vi = np.zeros([m, b-a])
                Vr = np.zeros([m, b-a])
                Fi = np.zeros([d-1, m, b-a])
                Fr = np.zeros([d-1, m, b-a])
                # hanning window
                hanning = 0.5*(1-np.cos((2*np.pi*np.arange(0, n)) / (n-1)))
                # loop through trials
                for trial in range(0, m):
                    F = self._Forig[:, trial, :]  # d x n
                    # extract v signal
                    V = v.T.dot(F)
                    # center, hanning window, fft
                    V = (V - np.mean(V)) * hanning
                    V = np.fft.rfft(V)[0, a:b]
                    Vi[trial, :] = np.imag(V)
                    Vr[trial, :] = np.real(V)
                    # remove v signal
                    F = self._P.dot(F)
                    # loop through channels
                    for channel in range(0, d-1):
                        x = F[channel, :]
                        x = (x - np.mean(x)) * hanning
                        x = np.fft.rfft(x)[a:b]
                        Fi[channel, trial, :] = np.imag(x)
                        Fr[channel, trial, :] = np.real(x)
                self._F = (Fi, Fr)
                self._C = (Vi, Vr)

    def _problem_MERLiN(self):
        '''
        Set up cost function and return the pymanopt problem of the MERLiN
        algorithm ([1], Algorithm 2).
        '''
        S = self._S
        C = self._C
        F = self._F
        m = self._m
        # set O,Q,R
        H = np.eye(m) - np.ones((m, m))/m
        O = ((S.T.dot(H.dot(C.dot(C.T))) -
              C.T.dot(H.dot(C.dot(S.T)))).dot(H)).dot(F.T)
        Q = ((S.T.dot(H.dot(C.dot(S.T))) -
              S.T.dot(H.dot(S.dot(C.T)))).dot(H)).dot(F.T)
        r1 = (S.T.dot(H.dot(S.dot(C.T.dot(H.dot(C))))))*np.eye(m)
        r2 = S.T.dot(H.dot(C))*C.dot(S.T)
        r3 = S.T.dot(H.dot(C))*S.dot(C.T)
        r4 = C.T.dot(H.dot(C))*S.dot(S.T)
        r5 = (S.T.dot(H.dot(C))**2)*np.eye(m)
        r6 = S.T.dot(H.dot(S))*C.dot(C.T)
        R = F.dot(((H.dot(r1 + r2 + r3 - r4 - r5 - r6)).dot(H)).dot(F.T))

        # set objective function and derivative
        def f(w):
            return -(np.abs(Q.dot(w)) - np.abs(O.dot(w))) / (
                np.abs(w.T.dot(R.dot(w))))

        def fprime(w):
            return -np.asarray((
                np.abs(w.T.dot(R.dot(w)))*(np.sign(Q.dot(w))*Q.T -
                                           np.sign(O.dot(w))*O.T) -
                np.sign(w.T.dot(R.dot(w)))*(np.abs(Q.dot(w)) -
                                            np.abs(O.dot(w)))*(R+R.T).dot(w)) /
                (np.abs(w.T.dot(R.dot(w)))**2))

        return Problem(manifold=None, cost=f, egrad=fprime, verbosity=0)

    def _problem_MERLiNbp(self, icoh=False):
        '''
        Set up cost function and return the pymanopt problem of the
        MERLiNbp algorithm ([1], Algorithm 4) or the MERLiNbpicoh
        algorithm ([1], Algorithm 5)

        Input (default)
            - icoh (False)
                False = set up MERLiNbp, True = set up MERLiNbpicoh

        Sets/updates
        self._problem_MERLiNbp_val or self._problem_MERLiNbpicoh_val
        and the shared theano variables
        self._T_S, self._T_Vi, self._T_Vr, self._T_Fi, self._T_Fr, self._T_n
        '''
        if (not icoh and self._problem_MERLiNbp_val is None) or (
                icoh and self._problem_MERLiNbpicoh_val is None):
            S = self._T_S = TS.shared(self._S)
            Vi = self._T_Vi = TS.shared(self._C[0])
            Vr = self._T_Vr = TS.shared(self._C[1])
            Fi = self._T_Fi = TS.shared(self._F[0].reshape(self._d, -1))
            Fr = self._T_Fr = TS.shared(self._F[1].reshape(self._d, -1))
            n = self._T_n = TS.shared(self._n)
            w = T.matrix()
            m = self._m
            # linear combination
            wFr = T.reshape(w.T.dot(Fr), (m, -1))  # m x n'
            wFi = T.reshape(w.T.dot(Fi), (m, -1))  # m x n'

            # replace zeros, since we're taking logs
            def unzero(x):
                return T.switch(T.eq(x, 0), 1, x)

            # bandpower
            def bp(re, im):
                return T.reshape(T.mean(
                    T.log(unzero(T.sqrt(re*re + im*im))) - T.log(n),
                    axis=1), (m, 1))

            wFbp = bp(wFr, wFi)  # m x 1
            vFbp = bp(Vr, Vi)  # m x 1
            # centering matrix
            I = T.eye(m, m)
            H = I - T.mean(I)
            # column-centered data
            X = H.dot(T.concatenate([S, vFbp, wFbp], axis=1))  # m x 3
            # covariance matrix
            S = X.T.dot(X) / (m-1)
            # precision matrix
            prec = Tlina.matrix_inverse(S)
            # MERLiNbpicoh
            if icoh:
                # complex row-wise vdot
                # (x+yi)(u+vi) = (xu-yv)+(xv+yu)i
                # vdot i.e. -v instead of +v
                def vdot(x, y, u, v):
                    return x*u+y*v

                def vdoti(x, y, u, v):
                    return -x*v+y*u

                def cross(x, y, u, v):
                    return T.sum(vdot(x, y, u, v), axis=0) / m

                def crossi(x, y, u, v):
                    return T.sum(vdoti(x, y, u, v), axis=0) / m

                def sqrtcross(x, y):
                    return T.sqrt(cross(x, y, x, y) + crossi(x, y, x, y))

                icoherency = crossi(Vr, Vi, wFr, wFi) / (
                    sqrtcross(Vr, Vi)*sqrtcross(wFr, wFi))  # n'
                cost = -(T.abs_(T.sum(icoherency))*T.abs_(prec[1, 2]) -
                         T.abs_(prec[0, 2]))
                self._problem_MERLiNbpicoh_val = Problem(manifold=None,
                                                         cost=cost, arg=w,
                                                         verbosity=0)
            # MERLiNbp
            else:
                cost = -(T.abs_(prec[1, 2])-T.abs_(prec[0, 2]))
                self._problem_MERLiNbp_val = Problem(manifold=None,
                                                     cost=cost, arg=w,
                                                     verbosity=0)
        else:
            self._T_S.set_value(self._S)
            self._T_Vi.set_value(self._C[0])
            self._T_Vr.set_value(self._C[1])
            self._T_Fi.set_value(self._F[0].reshape(self._d, -1))
            self._T_Fr.set_value(self._F[1].reshape(self._d, -1))
            self._T_n.set_value(self._n)
        if not icoh:
            return self._problem_MERLiNbp_val
        else:
            return self._problem_MERLiNbpicoh_val

    def _problem_MERLiNbpicoh(self):
        '''
        Set up cost function and return the pymanopt problem of the MERLiN
        algorithm ([1], Algorithm 5).
        '''
        return self._problem_MERLiNbp(icoh=True)

    def _problem_MERLiNnlbp(self):
        '''
        Set up cost function and return pymanopt problem of the non-linear
        MERLiNbp algorithm MERLiNnlbp.

        Sets/updates self._problem_MERLiNnlbp_val
        and the shared theano variables
        self._T_S, self._T_n, self._T_Vi123, self._T_Vr123, self._T_Fi123,
        self._T_Fr123
        '''
        if self._problem_MERLiNnlbp_val is None:
            S = self._T_S = TS.shared(self._S)
            n = self._T_n = TS.shared(self._n)
            # random three-split
            indices = [list(range(self._m))[i::3] for i in range(3)]
            Vi1 = self._T_Vi1 = TS.shared(self._C[0][indices[0], :])
            Vi2 = self._T_Vi2 = TS.shared(self._C[0][indices[1], :])
            Vi3 = self._T_Vi3 = TS.shared(self._C[0][indices[2], :])
            Vr1 = self._T_Vr1 = TS.shared(self._C[1][indices[0], :])
            Vr2 = self._T_Vr2 = TS.shared(self._C[1][indices[1], :])
            Vr3 = self._T_Vr3 = TS.shared(self._C[1][indices[2], :])
            Fi1 = self._T_Fi1 = TS.shared(self._F[0][:, indices[0], :].reshape(self._d, -1))
            Fi2 = self._T_Fi2 = TS.shared(self._F[0][:, indices[1], :].reshape(self._d, -1))
            Fi3 = self._T_Fi3 = TS.shared(self._F[0][:, indices[2], :].reshape(self._d, -1))
            Fr1 = self._T_Fr1 = TS.shared(self._F[1][:, indices[0], :].reshape(self._d, -1))
            Fr2 = self._T_Fr2 = TS.shared(self._F[1][:, indices[1], :].reshape(self._d, -1))
            Fr3 = self._T_Fr3 = TS.shared(self._F[1][:, indices[2], :].reshape(self._d, -1))
            w = T.matrix()
            sigma_signed = T.matrix()
            sigma = T.abs_(sigma_signed[0, 0])
            theta_signed = T.matrix()
            theta = T.abs_(theta_signed[0, 0])
            m = self._m
            # linear combination
            wFr1 = T.reshape(w.T.dot(Fr1), (Vr1.shape[0], -1))  # m1 x n'
            wFi1 = T.reshape(w.T.dot(Fi1), (Vi1.shape[0], -1))  # m1 x n'
            wFr2 = T.reshape(w.T.dot(Fr2), (Vr2.shape[0], -1))  # m2 x n'
            wFi2 = T.reshape(w.T.dot(Fi2), (Vi2.shape[0], -1))  # m2 x n'
            wFr3 = T.reshape(w.T.dot(Fr3), (Vr3.shape[0], -1))  # m3 x n'
            wFi3 = T.reshape(w.T.dot(Fi3), (Vi3.shape[0], -1))  # m3 x n'

            # replace zeros, since we're taking logs
            def unzero(x):
                return T.switch(T.eq(x, 0), 1, x)

            # bandpower
            def bp(re, im):
                return T.reshape(T.mean(
                    T.log(unzero(T.sqrt(re*re + im*im))) - T.log(n),
                    axis=1), (re.shape[0], 1))

            wFbp1 = bp(wFr1, wFi1)  # m1 x 1
            wFbp2 = bp(wFr2, wFi2)  # m2 x 1
            wFbp3 = bp(wFr3, wFi3)  # m3 x 1
            wFbp = T.concatenate([wFbp1, wFbp2, wFbp3])  # (m1+m2+m3 x 1)
            vFbp1 = bp(Vr1, Vi1)    # m1 x 1
            vFbp2 = bp(Vr2, Vi2)    # m2 x 1
            vFbp3 = bp(Vr3, Vi3)    # m3 x 1
            vFbp = T.concatenate([vFbp1, vFbp2, vFbp3])  # (m1+m2+m3 x 1)

            def median(x):
                return T.switch(T.eq((x.shape[0] % 2), 0),
                                T.mean(T.sort(x)[ ((x.shape[0]/2)-1) : ((x.shape[0]/2)+1) ]),
                                T.sort(x)[x.shape[0]//2])

            def kernmat(x, y, sig=None):
                xx = T.sum(x**2, axis=1, keepdims=True)
                yy = T.sum(y**2, axis=1, keepdims=True).T
                diffsq = xx + yy -2*x.dot(y.T)
                if sig is None:
                    # only use 100 samples or less
                    subdiffsq = diffsq[:T.min([diffsq.shape[0], 100]),
                                       :T.min([diffsq.shape[1], 100])]
                    # correct for the diagonal of zeros
                    sig = median(T.concatenate([subdiffsq.flatten(), subdiffsq.diagonal() + 1e10]))
                return T.exp(-diffsq/sig)

            def kreg(xtrain, ytrain, xtest):
                return kernmat(xtest, xtrain, sigma).dot(
                    Tlina.matrix_inverse(kernmat(xtrain, xtrain, sigma) +
                                         theta*T.eye(xtrain.shape[0])
                                         ).dot(ytrain))

            # regression residuals
            resi1 = (kreg(
                T.concatenate([vFbp2, vFbp3]),
                T.concatenate([wFbp2, wFbp3]),
                vFbp1) - wFbp1)
            resi2 = (kreg(
                T.concatenate([vFbp1, vFbp3]),
                T.concatenate([wFbp1, wFbp3]),
                vFbp2) - wFbp2)
            resi3 = (kreg(
                T.concatenate([vFbp1, vFbp2]),
                T.concatenate([wFbp1, wFbp2]),
                vFbp3) - wFbp3)
            resi = T.concatenate([resi1, resi2, resi3])

            def hsic(x, y):
                K = kernmat(x, x)
                L = kernmat(y, y)
                H = T.eye(m)-T.mean(T.eye(m))
                # omit (m-1)^-2 factor
                return (K.dot(H.dot(L.dot(H)))).trace()

            cost = -(hsic(vFbp, wFbp) - hsic(T.concatenate([S, vFbp], axis=1), resi))

            self._problem_MERLiNnlbp_val = Problem(manifold=None,
                                                   cost=cost, verbosity=2,
                                                   arg=[w, sigma_signed, theta_signed])
        else:
            raise NotImplementedError
        return self._problem_MERLiNnlbp_val
