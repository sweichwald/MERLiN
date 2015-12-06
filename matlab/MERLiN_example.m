%  generate dataset with the following parameters
T = 'gaussian';
d = 5;
m = 300;
a = 0.5;
b = 0.5;
[S,F,v,wG0] = genDataset(T,d,m,a,b);

%  run MERLiN
w = MERLiN(S,F,v);

disp('Algorithm: MERLiN')
disp(['On a dataset with parameters T=' T ' d=' num2str(d) ', m=' num2str(m) ', a=' num2str(a) ', b=' num2str(b)])
disp('and ground truth vector wG0=')
disp(wG0')
disp('MERLiN yielded the vector w=')
disp(w')
disp(['The angular distance is ' num2str(andi(wG0,w)) 'rad.'])
disp(['The probability of a better vector is ' num2str(pobv(wG0,w)) '.'])