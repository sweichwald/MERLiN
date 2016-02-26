%  maximise using pymanopt
%  Input
%      f, fprime: objective function and its gradient as anonymous functions
%                 (either both taking adigator structs or vectors as input)
%      w: initial point (either an adigator struct or a vector)
%  Optional input (not yet implemented)
%      tol: tolerance for stopping criterion
%      maxsteps: maximum number of Stiefel gradient ascent steps
%      lbd: initial step size
%  Output
%      w: found solution after maxsteps steps or when the stopping criterion was met
%      converged: whether the stopping criterion was met
%      curob: value of f at w
function [w, converged, curob] = maximise(f,fprime,w)

%  check manopt availability
checkManopt();

%  adigatormode
if isstruct(w)
    toadi = @(x) struct('f',x,'dw',w.dw);
    w = w.f;
    problem.cost = @(x) -f(toadi(x));
    problem.egrad = @(x) -adigator2vec(fprime(toadi(x)));
else
    problem.cost = @(x) -f(x);
    problem.egrad = @(x) -fprime(x);
end
problem.M = spherefactory(length(w));

tol = 1e-16;
options.maxtime = Inf;
options.maxiter = 500;
options.mingradnorm = 0;
options.minstepsize = tol;
options.verbosity = 0;
options.linesearch = @merlinlinesearch;

[w, curob, info, options] = steepestdescent(problem, w, options);

converged = info(end).iter ~= options.maxiter;
curob = -curob;

end


function [stepsize, newx, newkey, lsstats] = ...
           merlinlinesearch(problem, x, d, f0, ~, options, storedb, key)

    contraction_factor = .7;
    initial_stepsize = 1;
    minstepsize = options.minstepsize;

    norm_d = problem.M.norm(x, d);

    alpha = initial_stepsize / norm_d;

    newx = problem.M.retr(x, d, alpha);
    newkey = storedb.getNewKey();
    newf = getCost(problem, newx, storedb, newkey);

    % while there is no decrease, i.e. step too large
    while newf > f0 && alpha * norm_d > minstepsize
        alpha = contraction_factor * alpha;
        newx = problem.M.retr(x, d, alpha);
        newkey = storedb.getNewKey();
        newf = getCost(problem, newx, storedb, newkey);
    end
    if newf > f0
        alpha = 0;
        newx = x;
    end
    stepsize = alpha * norm_d;

    lsstats = NaN;
end


function v = adigator2vec(dw)
    v = zeros(dw.dw_size,1);
    v(dw.dw_location) = dw.dw;
end