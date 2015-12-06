%  performance measures (cf. Section III.B.)
%  Input to the following two functions
%      wG0: ground truth vector to compare against
%      w: vector to assess
%  Output
%      angular distance between wG0 and w
function a = andi(wG0, w)
    a = min(angle(wG0,w), angle(-wG0,w));
end


function rad = angle(u,v)
    u = u / norm(u);
    v = v / norm(v);
    rad = acos(u'*v);
end