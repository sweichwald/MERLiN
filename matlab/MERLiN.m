%  MERLiN (cf. Algorithm 2)
%  Input
%      S: (m x 1) vector of samples of S
%      F: (d x m) matrix of linear mixture samples
%      v: (d x 1) vector corresponding to C1 in S->C1
%  Optional input
%      'C', C: precomputed samples of middle node, v will be ignored
%  Output
%      w: found (d x 1) vector
%      converged: whether the stopping criterion of Stiefel gradient ascent was met
%      curob: value of f at w
function [w, converged, curob] = MERLiN(S,F,v,varargin)

[d, m] = size(F);

%  set C
%  optional C given?
if ~isempty(find(strcmp(varargin,'C')))
    C = varargin{find(strcmp(varargin,'C'))+1};
else
    C = F'*v;
end

%  remove v from F
F = null(v')'*F;

%  set O,Q,R
H = eye(m) - ones(m)/m;
O = ((S'*H*C)*C' - (C'*H*C)*S')*H*F';
Q = ((S'*H*C)*S' - (S'*H*S)*C')*H*F';
R = F*H*((S'*H*S)*(C'*H*C)*eye(m) + (S'*H*C)*C*S' + (S'*H*C)*S*C' - (C'*H*C)*S*S' - (S'*H*C)^2*eye(m) - (S'*H*S)*C*C')*H*F';

%  objective and gradient as inline functions
f = @(w) ( abs(Q*w) - abs(O*w) ) / abs( w'*R*w );
fprime = @(w) ( abs(w'*R*w)*( sign(Q*w)*Q'-sign(O*w)*O' ) - sign(w'*R*w)*( abs(Q*w) - abs(O*w) )*(R+R')*w ) / (abs(w'*R*w)^2);

w0 = randn(d-1,1);
w0 = w0/norm(w0);

[w, converged, curob] = maximise(f,fprime,w0);

w = null(v')*w;

end