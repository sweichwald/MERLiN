# MERLiN: Mixture Effect Recovery in Linear Networks

MERLiN is a causal inference algorithm that can recover from an observed linear mixture a causal variable that is a causal effect of another given variable.
MERLiN implements a novel idea on how to (re-)construct causal variables and is robust against hidden confounding.

As a motivational example consider the following causal structure

![examplegraph](https://drive.google.com/uc?id=0B8zEovCWLE22UkJpeV9QWUY4Wlk)

where S is a randomised variable, C=[C1,...,Cd]' are causal variables, and h is a hidden confounder.
Often the causal variables C1,...,Cd cannot be measured directly but only a linear mixture F=[F1,...,Fd]'=AC thereof can be observed.
In such scenarios, MERLiN is still able to establish the cause-effect relationship C1→C2.
Given

* samples of S,
* samples of F, and
* a vector v such that C1=v'F

the algorithm searches for a vector w such that w'F is an effect of C1, e.g. recovering C2=w'F as an effect of C1.

One practical example is the application to electroencephalographic (EEG) data recorded during a neurofeedback experiment.
Here, F1,...,Fd denote the electroencephalographic (EEG) channel recordings that are a linear mixture F=AC of the underlying cortical sources C1,...,Cd.
S denotes the randomised instruction to up-/downregulate the neurofeedback signal v'F.
In this setup, MERLiNbp and MERLiNbpicoh are able to recover a causal effect w'F of C1=v'F, i.e. establish a cause-effect relationship between two cortical signals v'F and w'F.

This repository will contain implementations of the following algorithms:

* **MERLiN**: precision matrix based algorithm that works on iid samples
* **MERLiNbp**: precision matrix based algorithm that works on iid sampled timeseries chunks and searches for a certain cause-effect relationship between the resulting log-bandpower features
* **MERLiNbpicoh**: extends MERLiNbp by an imaginary coherency regularisation

The latter two algorithms may be applied to any type of timeseries data while they are tailored to analysis of EEG data.

---

## python3

`python3 MERLiN_example` or `%run MERLiN_example.py` in ipython3 for a simple example of use.

* Requires numpy, scipy, theano (install via pip3).
* Tested with python3.4.3.
* No validation of user input to functions.


## python2

Currently the python3 implementation is also compatible with python2. Tested with python2.7.6.

`python2 MERLiN_example` or `%run MERLiN_example.py` in ipython2 for a simple example of use.


## matlab

Following soon.
Send an email to *merlin-matlab* (at) *sweichwald* (dot) *de* and you'll get notified once the matlab implementation is available.