%  MERLiNbp (cf. Algorithm 4)
%  Input
%      S: (m x 1) vector of samples of S
%      Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
%      v: (d x 1) vector corresponding to C1 in S->C1
%      fs: sampling rate
%      omega1, omega2: low/high limit of desired frequency band
%  Optional input
%      'C', C: precomputed samples of middle node, v will be ignored
%  Optional input (not yet implemented)
%      preprocessed: dict of already preprocessed data Vi, Vr, Fi, Fr, n
%  Output
%      w: found solution after maxsteps steps or when the stopping criterion was met
%      converged: whether the stopping criterion was met
%      curob: value of f at w
function [w, converged, curob] = MERLiNbp(S,Ftw,v,fs,omega1,omega2,varargin)

%  check adigator availability
checkADiGator();

[d,m,n] = size(Ftw);

%  preprocess
%  optional C given?
if ~isempty(find(strcmp(varargin,'C')))
    C = varargin{find(strcmp(varargin,'C'))+1};
    [Fi, Fr, dummy] = preprocess(Ftw,zeros(size(Ftw,1),1),fs,omega1,omega2);
else
    [Fi, Fr, C] = preprocess(Ftw,v,fs,omega1,omega2);
end

nprime = size(Fi,2)/m;

%  set O,Q,R
H = eye(m) - ones(m)/m;
O = ((S'*H*C)*C' - (C'*H*C)*S')*H;
Q = ((S'*H*C)*S' - (S'*H*S)*C')*H;
R = H*((S'*H*S)*(C'*H*C)*eye(m) + (S'*H*C)*C*S' + (S'*H*C)*S*C' - (C'*H*C)*S*S' - (S'*H*C)^2*eye(m) - (S'*H*S)*C*C')*H;

%  compile objective's gradient
options = adigatorOptions('OVERWRITE',1);
w = adigatorCreateDerivInput([d 1],'w');
evalc('adigator(''objective_MERLiNbp'',{w,n,Fi,Fr,O,Q,R},''gradient_MERLiNbp'',options);');

%  random initial vector in orthogonal complement
w0 = randn(d,1);
if isempty(find(strcmp(varargin,'C')))
    w0 = null(v')*null(v')'*w0;
end
w0 = w0/norm(w0);
w = struct('f',w0,'dw',ones(d,1));

f = @(w) objective_MERLiNbp(w.f,n,Fi,Fr,O,Q,R);
fprime = @(w) gradient_MERLiNbp(w,n,Fi,Fr,O,Q,R);

[w, converged, curob] = stiefasc(f,fprime,w);

w = w.f;

end