%  MERLiN objective function
function obj = objective_MERLiN(w,F,O,Q,R)
    obj = ( abs(Q*w) - abs(O*w) ) / abs( w'*R*w );
end