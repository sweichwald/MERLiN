%  Generate synthetic dataset (cf. Algorithm 6)
%  Input
%      T: gaussian or binary dataset
%      d: dimension
%      m: number of samples
%      a: noise parameter
%      b: hidden confounding parameter
%  Optional input (not yet implemented)
%      eye: random orthonormal mixing (False) or no/identity mixing (True)
%  Output
%      S: (m x 1) vector of samples of S
%      F: (d x m) matrix of linear mixture samples
%      v: (d x 1) vector corresponding to C1 in S->C1
%      wG0: ground truth (d x 1) vector to recover C2
function [S,F,v,wG0] = genDataset(T,d,m,a,b)

eyemix = false;

if eyemix
    A = eye(d);
else
    % generate random orthogonal d x d matrix
    [A,dummy] = qr(randn(d));
end

%  set v and wG0
v = A(:,1);
wG0 = A(:,2);

%  generate S vector
if strcmp(T,'binary')
    S = randi(2,m,1)*2-3;
elseif strcmp(T,'gaussian')
    S = randn(m,1);
end

%  generate hidden confounder
h = randn(m,1) + randn(1,1);

%  generate mean for each Ci
mu = randn(d,1);

%  SEM
C = randn(d,m);
C = C + repmat(mu,1,m);
C(1,:) =   C(1,:) + S' + b*h';
C(2,:) = a*(C(2,:)-mu(2)) + mu(2) + C(1,:);
C(3,:) =   C(3,:) + S';
C(4,:) =   C(4,:) + b*h';

%  orthogonal mixing
F = A*C;

end