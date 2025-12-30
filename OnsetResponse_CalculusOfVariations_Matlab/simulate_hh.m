function [t,y] = simulate_hh(state, tspan, Istim_func, params)

opts = odeset('RelTol',1e-6,'AbsTol',1e-9);
[t,y] = ode15s(@(t,y) hh_rhs(t,y,params, Istim_func), tspan, state, opts);

end

%% --------------------------------------------------------
function dydt = hh_rhs(t,y,p, Istim_func)
V = y(1); m = y(2); n = y(3); h = y(4);

% rates
[am, bm] = am_bm(V,p.Q);
[ah, bh] = ah_bh(V,p.Q);
[an, bn] = an_bn(V,p.Q);

% no external current
Istim = Istim_func(t);

dVdt = ( - p.gNa*m.^3.*h.*(V - p.ENa) ...
         - p.gK*n.^4.*(V - p.EK) ...
         - p.gL*(V - p.EL) ...
         + Istim - p.Ioff ) / p.C;
dmdt = - (am + bm).*m + am;
dhdt = - (ah + bh).*h + ah;
dndt = - (an + bn).*n + an;

dydt = [dVdt; dmdt; dndt; dhdt];
end

%% --------------------------------------------------------
% Rate functions (without derivatives needed here)
function [am,bm] = am_bm(V,Q)
x = 0.1*(25 - V);
am = 0.1*Q*(25 - V) ./ (exp(x) - 1);
bm = 4*Q*exp(-V/18);

% am(abs(25-V)<=0.01) = Q; % correct divisions by zero
end

function [ah,bh] = ah_bh(V,Q)
ah = 0.07*Q*exp(-V/20);
bh = Q ./ (exp(0.1*(30 - V)) + 1);
end

function [an,bn] = an_bn(V,Q)
x = 0.1*(10 - V);
an = 0.01*Q*(10 - V) ./ (exp(x) - 1);
bn = 0.125*Q*exp(-V/80);

% an(abs(10-V)<=0.01) = 0.1*Q; % correct divisions by zero
end
