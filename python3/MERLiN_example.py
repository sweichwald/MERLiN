from MERLiN import MERLiN
from MERLiN_helper import pobv, andi, genDataset, genToyTimeseriesTensor


# function for printing results
def printres(algo, T, d, m, a, b, w, wG0):
    print('Algorithm: ' + algo + '.')
    timeseries = ''
    if algo is not 'MERLiN':
        timeseries = 'timeseries '
    print('On a ' + timeseries + 'toy dataset with parameters T=' + T +
          ', d=' + str(d) + ', m=' + str(m) + ', a=' + str(a) + ', b=' +
          str(b))
    print('and ground truth vector wG0=' + str(wG0.T))
    print('MERLiN yielded the vector w=' + str(w.T) + '.')
    print('The angular distance is ' + str(andi(wG0, w)) + 'rad.')
    print('The probability of a better vector is ' + str(pobv(wG0, w)) + '.\n')


# generate dataset with the following parameters
T = 'gaussian'
d = 5
m = 300
a = 0.5
b = 0.5
[S, F, v, wG0] = genDataset(T, d, m, a, b)


# run MERLiN
merlin = MERLiN()
w = merlin.run(S, F, v=v)[0]
printres('MERLiN', T, d, m, a, b, w, wG0)


# generate toy timeseries dataset
[S, F, v, wG0] = genDataset(T, d, m, a, b, eye=True)
fs = 100
n = 100
omega1 = 8
omega2 = 12
Ftw = genToyTimeseriesTensor(F, fs, n, omega1, omega2)


# run MERLiNbp
res = merlin.run(S, Ftw, v=v, fs=fs, omega=(omega1, omega2), variant='nlbp')
w = res[0]
printres('MERLiNbp', T, d, m, a, b, w, wG0)
