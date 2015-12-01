# MERLiN: Mixture Effect Recovery in Linear Networks

**Currently setting-up this repo.** Come back later for python3/python2/matlab implementations or send an email to *merlin-{python3,python2,matlab}* (at) *sweichwald* (dot) *de* and you'll get notified once the respective code is available.

MERLiN aims at recovering/constructing from an observed linear mixture a variable that is a causal effect of another given variable.

This repository will contain implementations of the following algorithms:

* **MERLiN**: precision matrix based algorithm that works on iid samples
* **MERLiNbp**: precision matrix based algorithm that works on iid sampled timeseries chunks and searches for a certain cause-effect relationship between the resulting log-bandpower features
* **MERLiNbpicoh**: extends MERLiNbp by an imaginary coherency regularisation

The latter two algorithms may be applied to any type of timeseries data while they are tailored to analysis of electroencephalographic (EEG) data.

---

## python3

`%run MERLiN_example.py` in IPython for a simple example of use.

* MERLiNbp and MERLiNbpicoh not yet implemented.
* Requires numpy, scipy, theano (install via pip3).
* No validation of user input to functions.


## python2

Not yet implemented.


## matlab

Not yet implemented.