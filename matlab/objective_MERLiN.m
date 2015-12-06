%  MERLiN objective function
function obj = objective_MERLiN(w,F,P,Q,R)
    obj = ( abs(Q*F'*w) - abs(P*F'*w) ) / abs( w'*F*R*F'*w );
end