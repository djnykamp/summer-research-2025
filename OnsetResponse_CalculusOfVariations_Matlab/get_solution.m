function [t, y, ly] = get_solution(state0, state1, params)
% Solve calculus-of-variations Euler equations for HH using bvp4c.
%   This implements the 8 ODEs (V,m,n,h,lV,lm,ln,lh) from the Methods section.
%
%   Usage:
%     - Edit state0/state1 and params as needed (units and offsets follow the paper).
%
%   Outputs:
%     - Plots of V(t), m,n,h and Lagrange multipliers, and Istim(t).

%% Time horizon
T = params.T;                % duration in ms (paper used 20 ms)
dt = 0.02;
if isfield(params, 'dt')
    dt = params.dt;
end
tmesh = 0:dt:T;

%% Initial guess for full 8-component solution across mesh
% state guess: linear interpolation V,m,n,h from state0 to state1
solinit = bvpinit(tmesh, @(t) init_guess(t, state0, state1, params));

% Solve with bvp4c
opts = bvpset('RelTol',1e-5,'AbsTol',1e-7,'Stats','on','Vectorized','on');
sol = bvp4c(@(t,y) odefun(t,y,params), @(ya,yb) bcfun(ya,yb,state0,state1), solinit, opts);

% Evaluate solution on fine grid
tt = linspace(0,T,500);
yy = deval(sol, tt);

t = tt;
V = yy(1,:)';  m = yy(2,:)';  n = yy(3,:)';  h = yy(4,:)';
lV= yy(5,:)'; lm = yy(6,:)'; ln = yy(7,:)'; lh = yy(8,:)'; 
y = [V,m,n,h];
ly = [lV,lm,ln,lh];

end


%% ---------------------------
%% ODE system
function dydt = odefun(t,y,p)
% y = [V; m; n; h; lV; lm; ln; lh]

V = y(1,:); m = y(2,:); n = y(3,:); h = y(4,:);
lV = y(5,:); lm = y(6,:); ln = y(7,:); lh = y(8,:);

% rates and derivatives
[am, bm, dam_dV, dbm_dV] = am_bm(V,p.Q);
[ah, bh, dah_dV, dbh_dV] = ah_bh(V,p.Q);
[an, bn, dan_dV, dbn_dV] = an_bn(V,p.Q);

% State derivatives (Hodgkin-Huxley form)
gNa = p.gNa; gK = p.gK; gL = p.gL;
ENa = p.ENa; EK = p.EK; EL = p.EL;
C = p.C;


% optimal control (from calculus of variations: d/dI( I^2 + lV*I ) = 0 -> I = -lV/2)
Istim = - lV / 2;
dVdt = ( - gNa*m.^3.*h.*(V - ENa) - gK*n.^4.*(V - EK) - gL*(V - EL) + Istim - p.Ioff ) / C;

% trying to match paper?
% dVdt = ( - gNa*m^3*h*(V - ENa) - gK*n^4*(V - EK) - gL*(V - EL) - lV / 2 - p.Ioff ) / C;
dmdt = - (am + bm).*m + am;
dhdt = - (ah + bh).*h + ah;
dndt = - (an + bn).*n + an;

% Adjoint equations (from paper's eqn 4)
% dlV/dt
dlVdt =  lV .* ( gNa*m.^3.*h + gK*n.^4 + gL ) ...
        - lm .* ( dam_dV .* (1-m) - dbm_dV .* m ) ...
        - ln .* ( dan_dV .* (1-n) - dbn_dV .* n ) ...
        - lh .* ( dah_dV .* (1-h) - dbh_dV .* h );

% dlm/dt, dln/dt, dlh/dt
% note: paper uses terms like -lV*360*m^2*h*(ENa - V) etc (gNa=120 -> 360 for derivative w.r.t m)
dlmdt = - lV .* ( 3 * gNa * m.^2 .* h .* (ENa - V) ) + lm .* (am + bm);
dlndt = - lV .* ( 4 * gK  * n.^3 .*      (EK  - V) ) + ln .* (an + bn);   % paper uses 144 n^3 -> gK*4 = 36*4 = 144
dlhdt = - lV .* (     gNa * m.^3 .*      (ENa - V) ) + lh .* (ah + bh);

dydt = [dVdt; dmdt; dndt; dhdt; dlVdt; dlmdt; dlndt; dlhdt];
end

%% ---------------------------
%% Boundary conditions
function res = bcfun(ya,yb,state0,state1)
% ya = y(0), yb = y(T)
% Enforce the four state BCs: V(0..)=state0, V(T..)=state1
% Costates are free -> natural BCs => set no additional constraints on costates.
% We implement freedom by not specifying costate values (i.e., zero residual for costate eqs)
% If you want to force final costates, change these rows.

res = zeros(8,1);
% initial states
res(1:4) = ya(1:4) - state0(:);
% final states
res(5:8) = yb(1:4)  - state1(:);

% If you want to fix costates instead, set res(5:8) = [ya(5:8)-l0; yb(5:8)-lT] etc.
end

%% ---------------------------
%% Initial guess for bvp solver
function y0 = init_guess(t, a0, b1, p)
% linear interpolation for states; small zero guess for costates
tau = t / p.T;
Vguess = (1-tau)*a0(1) + tau*b1(1);
mguess = (1-tau)*a0(2) + tau*b1(2);
nguess = (1-tau)*a0(3) + tau*b1(3);
hguess = (1-tau)*a0(4) + tau*b1(4);

% costate initial guess: small ramp or zero (improves convergence sometimes)
lV0 = 0; lm0 = 0; ln0 = 0; lh0 = 0;

y0 = [Vguess; mguess; nguess; hguess; lV0; lm0; ln0; lh0];
end

%% ---------------------------
%% Rate functions and their V-derivatives (analytic)
function [am,bm,dam_dV,dbm_dV] = am_bm(V,Q)
% am(V) = 0.1*Q*(25-V)/(exp(0.1*(25-V)) - 1)
% bm(V) = 4*Q*exp(-V/18)
x = 0.1*(25 - V);
expx = exp(x);
den = expx - 1; % denominator
am = 0.1*Q*(25 - V) ./ den;
% derivative am'
dN = -0.1*Q;
dD = -0.1*expx;
dam_dV = (dN .* den - (0.1*Q*(25 - V)) .* dD) ./ (den.^2);

bm  = 4*Q*exp(-V/18);
dbm_dV = - (4*Q/18) * exp(-V/18);
end

function [ah,bh,dah_dV,dbh_dV] = ah_bh(V,Q)
% ah = 0.07*Q*exp(-V/20)
% bh = Q/(exp(0.1*(30-V)) + 1)
ah = 0.07*Q*exp(-V/20);
dah_dV = - (0.07*Q/20) * exp(-V/20);

E = exp(0.1*(30 - V));
bh = Q ./ (E + 1);
% derivative: dbh/dV = Q * (0.1 * E) / (E + 1)^2
dbh_dV = Q * (0.1 * E) ./ ((E + 1).^2);
end

function [an,bn,dan_dV,dbn_dV] = an_bn(V,Q)
% an = 0.01*Q*(10-V)/(exp(0.1*(10-V)) - 1)
% bn = 0.125*Q*exp(-V/80)   (paper used 80 ms in standard HH)
x = 0.1*(10 - V);
exp_x = exp(x);
den = exp_x - 1;
an = 0.01*Q*(10 - V) ./ den;
dN = -0.01*Q;
dD = -0.1*exp_x;
% careful: derivative d/dV of denom exp(0.1*(10-V))-1 is -0.1*exp_x
dan_dV = (dN .* den - (0.01*Q*(10 - V)) .* dD) ./ (den.^2);

bn = 0.125*Q*exp(-V/80);
dbn_dV = - (0.125*Q/80) * exp(-V/80);
end
