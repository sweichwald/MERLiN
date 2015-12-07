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