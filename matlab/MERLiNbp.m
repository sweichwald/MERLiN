%  MERLiNbp (cf. Algorithm 4)
%  Input
%      S: (m x 1) vector of samples of S
%      Ftw: (d x m x n) tensor containing timeseries of length n (d channels, m trials)
%      v: (d x 1) vector corresponding to C1 in S->C1
%      fs: sampling rate
%      omega1, omega2: low/high limit of desired frequency band
%  Optional input (not yet implemented)
%      preprocessed: dict of already preprocessed data Vi, Vr, Fi, Fr, n
%  Output
%      w: found solution after maxsteps steps or when the stopping criterion was met
%      converged: whether the stopping criterion was met
%      curob: value of f at w
function [w, converged, curob] = MERLiNbp(S,Ftw,v,fs,omega1,omega2)

%  check adigator availability
checkADiGator();

[d,m,n] = size(Ftw);

%  preprocess
[Fi, Fr, C] = preprocess(Ftw,v,fs,omega1,omega2);

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

w0 = randn(d,1);
w0 = w0/norm(w0);
w = struct('f',w0,'dw',ones(d,1));

f = @(w) objective_MERLiNbp(w.f,n,Fi,Fr,O,Q,R);
fprime = @(w) gradient_MERLiNbp(w,n,Fi,Fr,O,Q,R);

[w, converged, curob] = stiefasc(f,fprime,w);

w = w.f;

end


function [Fi, Fr, C] = preprocess(Ftw,v,fs,omega1,omega2)
    [d,m,n] = size(Ftw);

    % frequency range
    a = min(find( (1:ceil(n/2))*fs/n > omega1 ));
    b = max(find( (1:ceil(n/2))*fs/n <= omega2 ));
    nprime = (b-a+1);

    Fnm = zeros(d,m*nprime);
    C = zeros(m,1);

    % hanning window
    hanning = 0.5*(1-cos( 2*pi*(0:n-1) / (n-1) ));

    for trial=1:m
        F = squeeze(Ftw(:,trial,:));

        % extract v signal
        V = v'*F;

        % center, hanning window, fft
        V = (V-mean(V)) .* hanning;
        V = fft(V);
        V = V(a:b);
        C(trial) = mean(log(unzero(abs(V))))-log(n);

        % remove v signal
        P = null(v')';
        F = P'*P*F;

        % hanning and fft
        F = F .* repmat(hanning,d,1);
        F = fft(F')';
        Fnm(:, ((trial-1)*nprime+1):trial*nprime) = F(:,a:b);
    end

    Fi = imag(Fnm);
    Fr = real(Fnm);
end


function x = unzero(x)
    x(x == 0) = 1;
end