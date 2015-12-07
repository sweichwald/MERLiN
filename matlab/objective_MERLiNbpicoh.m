%  MERLiNbpicoh objective function
function obj = objective_MERLiNbpicoh(w,n,Fi,Fr,Vi,Vr,P,Q,R)
    m = size(R,1);
    [d,nm] = size(Fi); % trial 1:coefficient 1 , trial 1:coefficient 2, ...
    nprime = nm/m;

    Fwi = Fi'*w;
    Fwr = Fr'*w;

    % trial-wise mean matrix
    H = zeros(m,m*nprime);
    for trial=1:m
        H(trial, ((trial-1)*nprime+1) : trial*nprime ) = ones(1,nprime)/nprime;
    end

    % m x 1 vector of log-bandpowers
    Fw = H*log( unzero(sqrt( (Fwi).^2 + (Fwr).^2 )) )-log(n);

    % coefficient-wise mean matrix
    H = repmat(diag(ones(1,nprime))/m,1,m);
    % imaginary coherency
    icoh = abs(sum( H*(Vi.*Fwr-Vr.*Fwi) ./ sqrt( (H*(Vi.^2+Vr.^2)) .* (H*(Fwi.^2+Fwr.^2)) ) ));

    obj = (m-1)*( icoh*abs(Q*Fw) - abs(P*Fw) ) / abs( Fw'*R*Fw );
end