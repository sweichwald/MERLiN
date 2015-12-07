%  MERLiNbp objective function
function obj = objective_MERLiNbp(w,n,Fi,Fr,P,Q,R)
    m = size(R,1);
    [d,nm] = size(Fi); % trial 1:coefficient 1 , trial 1:coefficient 2, ...
    nprime = nm/m;
    % trial-wise mean matrix
    H = zeros(m,m*nprime);
    for trial=1:m
        H(trial, ((trial-1)*nprime+1) : trial*nprime ) = ones(1,nprime)/nprime;
    end
    % m x 1 vector of log-bandpowers
    Fw = H*log( unzero(sqrt( (Fi'*w).^2 + (Fr'*w).^2 )) )-log(n);
    obj = ( abs(Q*Fw) - abs(P*Fw) ) / abs( Fw'*R*Fw );
end