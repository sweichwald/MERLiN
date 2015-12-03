from MERLiN_main import pobv, andi, genDataset, MERLiN, bpPreprocessing
from MERLiN_helper import genToyTimeseriesTensor


#generate dataset with the following parameters
T = 'gaussian'
d = 5
m = 200
a = 0.5
b = 0.5

[S,F,v,wG0] = genDataset(T,d,m,a,b)


#run MERLiN
w = MERLiN(S,F,v)[0]


#print results
print('On a dataset with parameters T=G, d=5, m=200, a=.5, b=.5')
print('and ground truth vector wG0=' + str(wG0.T))
print('MERLiN yielded the vector w=' + str(w.T) + '.')
print('The angular distance is ' + str(andi(wG0,w)) + 'rad.')
print('The probability of a better vector is ' + str(pobv(wG0,w)) + '.')


#generate toy timeseries dataset
[S,F,v,wG0] = genDataset(T,d,m,a,b,eye=True)
fs = 250
n = 500
omega1 = 55
omega2 = 85
Ftw = genToyTimeseriesTensor(F,v,fs,n,omega1,omega2)


#preprocessing
Vi, Vr, Fi, Fr = bpPreprocessing(Ftw,v,fs,omega1,omega2)