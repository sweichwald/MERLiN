%  performance measures (cf. Section III.B.)
%  Input to the following two functions
%      wG0: ground truth vector to compare against
%      w: vector to assess
%  Output
%      probability of a better vector than w
function p = pobv(wG0,w)
    a = (size(wG0,1)-1)/2;
    b = 0.5;
    % ensure both vectors are normed -> r=1
    wG0 = wG0 / norm(wG0);
    w = w / norm(w);
    h = 1 - abs( wG0'*w );
    x = h*(2-h);
    p = betainc(x,a,b);
end