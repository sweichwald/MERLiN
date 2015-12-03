# MERLiN: Mixture Effect Recovery in Linear Networks

MERLiN aims at recovering/constructing from an observed linear mixture a variable that is a causal effect of another given variable.

This repository will contain implementations of the following algorithms:

* **MERLiN**: precision matrix based algorithm that works on iid samples
* **MERLiNbp**: precision matrix based algorithm that works on iid sampled timeseries chunks and searches for a certain cause-effect relationship between the resulting log-bandpower features
* **MERLiNbpicoh**: extends MERLiNbp by an imaginary coherency regularisation

The latter two algorithms may be applied to any type of timeseries data while they are tailored to analysis of electroencephalographic (EEG) data.

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