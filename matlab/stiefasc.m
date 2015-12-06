%  Stiefel gradient ascent (cf. Algorithm 1)
%  Input
%      f, fprime: objective function and its gradient as theano functions
%      w: initial point
%  Optional input (not yet implemented)
%      tol: tolerance for stopping criterion
%      maxsteps: maximum number of Stiefel gradient ascent steps
%      lbd: initial step size
%  Output
%      w: found solution after maxsteps steps or when the stopping criterion was met
%      converged: whether the stopping criterion was met
%      curob: value of f at w
function [w, converged, curob] = stiefasc(f,fprime,w)

tol=1e-16;
maxsteps=500;
lbd=1;

converged = 0;
curob = f(w);

for k=1:maxsteps
    % while there is no increase, i.e. step too large
    dw = fprime(w);
    dw = dw.dw;
    wnew = w;
    wnew.f = stiefel_update(w.f, dw, lbd);
    curlbd = lbd;
    while f(wnew) < curob
        curlbd = curlbd*.5;
        wnew.f = stiefel_update(w.f, dw, curlbd);
    end
    w = wnew;

    newob = f(w);

    % 'converged'?
    if abs(curob-newob) < tol
        converged = 1;
        break
    end

    curob = newob;
end

end


%  from V take Stiefel ascent step defined by the gradient G and step size lbd
function Vnew = stiefel_update(V, G, lbd)
    [n, p] = size(V);
    Vp = null(V');
    Z = [ V'*G-G'*V , -G'*Vp ; Vp'*G ,  zeros(n-p) ];
    Vnew = [V,Vp] * expm(lbd*Z) * eye(n,p);
end