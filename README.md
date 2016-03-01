# MERLiN: Mixture Effect Recovery in Linear Networks

*This is code accompanying the manuscript of the same title by Sebastian Weichwald, Moritz Grosse-Wentrup, Arthur Gretton; cf. http://arxiv.org/abs/1512.01255.*

MERLiN is a causal inference algorithm that can recover from an observed linear mixture a causal variable that is an effect of another given variable.
MERLiN implements a novel idea on how to (re-)construct causal variables and is robust against hidden confounding.

As a motivational example consider the following causal structure

![examplegraph](http://sweichwald.bplaced.net/MERLiN/examplegraph.png)

where S is a randomised variable, C=[C1,...,Cd]' are causal variables, and h is a hidden confounder.
Often the causal variables C1,...,Cd cannot be measured directly but only a linear mixture F=[F1,...,Fd]'=AC thereof can be observed.
In such scenarios, MERLiN is still able to establish the cause-effect relationship C1→C2.
Given

* samples of S,
* samples of F, and
* a vector v such that C1=v'F

the algorithm searches for a vector w such that w'F is an effect of C1, e.g. recovering C2=w'F as an effect of C1.

One practical example is the application to electroencephalographic (EEG) data recorded during a neurofeedback experiment.
Here, F1,...,Fd denote the EEG channel recordings that are a linear mixture F=AC of the underlying cortical sources C1,...,Cd.
S denotes the randomised instruction to up-/downregulate the neurofeedback signal C1=v'F.
In this setup, MERLiNbp and MERLiNbpicoh are able to recover from F a causal effect C2=w'F of C1, i.e. establish a cause-effect relationship between two cortical sources C1 and C2.

This repository provides python and matlab implementations of the following algorithms:

* **MERLiN**: precision matrix based algorithm that works on iid samples
* **MERLiNbp**: precision matrix based algorithm that works on iid sampled timeseries chunks and searches for a certain cause-effect relationship between the resulting log-bandpower features
* **MERLiNbpicoh**: extends MERLiNbp by an imaginary coherency regularisation

Note that the python implementation should be preferred over the matlab implementation. While the provided matlab implementation implements the algorithms MERLiN/-bp/-bpicoh as first described in the original manuscript, only the python implementation includes all newer modifications/extensions/improvements.

The latter two algorithms may be applied to any type of timeseries data.
In [this manuscript](http://arxiv.org/abs/1512.01255), for example, they have been employed in the analysis of EEG data.


---


## python3

The file [MERLiN_example.py](python3/MERLiN_example.py) provides a simple example of use --- call `python3 MERLiN_example.py` or `%run MERLiN_example.py` in ipython3.

### get going

The file [MERLiN.py](python3/MERLiN.py) provides the main functionality --- in a nutshell:

```python
from MERLiN import MERLiN

merlin = MERLiN()

# the basic algorithm
# S: (m x 1) np.array that contains the samples of S
# F: (d x m) np.array that contains the linear mixture samples
# v: (d x 1) np.array, the linear combination corresponding to C1 in S->C1,
#     instead the middle node's samples arranged as (m x 1) np.array can
#     also be handed over as optional argument C
res = merlin.run(S, F, v=v)

# the bp algorithms for iid sampled timeseries chunks
# S: (m x 1) np.array that contains the samples of S
# Ftw: (d x m x n) np.array that contains the linearly mixed timeseries of
#       length n (d channels, m trials)
# v: (d x 1) np.array, the linear combination corresponding to C1 in S->C1
# fs: sampling rate
# omega: tuple of (low, high) cut-off of desired frequency band
res = merlin.run(S, Ftw, v=v, fs=fs, omega=omega)
res = merlin.run(S, Ftw, v=v, fs=fs, omega=omega, variant='bpicoh')

# the solution vector
w = res[0]
```

### notes

* Requires numpy, scipy, theano, and [pymanopt](https://pymanopt.github.io/)
* Tested with python3.4.3.
* No validation of user input to functions.


## python2

Currently the python3 implementation is also compatible with python2. Tested with python2.7.6.

The file [MERLiN_example.py](python3/MERLiN_example.py) provides a simple example of use --- call `python2 MERLiN_example.py` or `%run MERLiN_example.py` in ipython2.


## matlab

Note that the python implementation should be preferred over the matlab implementation. While the provided matlab implementation implements the algorithms MERLiN/-bp/-bpicoh as first described in the original manuscript, only the python implementation includes all newer modifications/extensions/improvements.

The file [MERLiN_example.m](matlab/MERLiN_example.m) provides a simple example of use --- call `MERLiN_example` in matlab.

### get going

The file [MERLiN.m](matlab/MERLiN.m)/[MERLiNbp.m](matlab/MERLiNbp.m)/[MERLiNbpicoh.m](matlab/MERLiNbpicoh.m) provides the MERLiN/-bp/-bpicoh function --- in a nutshell:

```matlab
% the basic algorithm
% S: (m x 1) vector of samples of S
% F: (d x m) matrix of linear mixture samples
% v: (d x 1) vector corresponding to C1 in S->C1
[w, converged, curob] = MERLiN(S,F,v)

% the bp algorithms for iid sampled timeseries chunks
% S: (m x 1) vector of samples of S
% Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
% v: (d x 1) vector corresponding to C1 in S->C1
% fs: sampling rate
% omega1, omega2: low/high limit of desired frequency band
[w, converged, curob] = MERLiNbp(S,Ftw,v,fs,omega1,omega2)
[w, converged, curob] = MERLiNbpicoh(S,Ftw,v,fs,omega1,omega2)

% the solution vector
w
```

### notes

* Requires [ADiGator](http://adigator.sourceforge.net/) (tested with V1.1.1). Download and add to matlab search path via `addpath(genpath('/path/to/adigator'))`.
* Requires [Manopt](http://manopt.org/) (tested with V2.0). Download and add to matlab search path via `addpath(genpath('/path/to/manopt'))`.
* Tested with Matlab R2014a.
* No validation of user input to functions.