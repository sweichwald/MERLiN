%  MERLiN (cf. Algorithm 2)
%  Input
%      S: (m x 1) vector of samples of S
%      F: (d x m) matrix of linear mixture samples
%      v: (d x 1) vector corresponding to C1 in S->C1
%  Output
%      w: found (d x 1) vector
%      converged: whether the stopping criterion of Stiefel gradient ascent was met
%      curob: value of f at w
function [w, converged, curob] = MERLiN(S,F,v)

%  check adigator availability
checkADiGator();

[d, m] = size(F);

%  set C
C = F'*v;

%  remove v from F
F = null(v')'*F;

%  set O,Q,R
H = eye(m) - ones(m)/m;
O = ((S'*H*C)*C' - (C'*H*C)*S')*H*F';
Q = ((S'*H*C)*S' - (S'*H*S)*C')*H*F';
R = F*H*((S'*H*S)*(C'*H*C)*eye(m) + (S'*H*C)*C*S' + (S'*H*C)*S*C' - (C'*H*C)*S*S' - (S'*H*C)^2*eye(m) - (S'*H*S)*C*C')*H*F';

%  compile objective's gradient
options = adigatorOptions('OVERWRITE',1);
w = adigatorCreateDerivInput([d-1 1],'w');
evalc('adigator(''objective_MERLiN'',{w,F,O,Q,R},''gradient_MERLiN'',options);');

w0 = randn(d-1,1);
w0 = w0/norm(w0);
w = struct('f',w0,'dw',ones(d-1,1));

f = @(w) objective_MERLiN(w.f,F,O,Q,R);
fprime = @(w) gradient_MERLiN(w,F,O,Q,R);

[w, converged, curob] = stiefasc(f,fprime,w);

w = null(v')*w.f;

end