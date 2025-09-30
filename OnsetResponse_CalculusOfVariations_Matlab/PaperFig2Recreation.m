% simulate_beta1  Forward HH simulation for 20 ms starting at beta1, no stimulus.

%% Parameters (same as in cv_hh_bvp)
params.C   = 1.0;
params.ENa = 115;
params.EK  = -12;
params.EL  = 10.613;
params.gNa = 120;
params.gK  = 36;
params.gL  = 0.3;
params.Q   = 1.5;
params.Ioff = 0;   % no external stimulus
params.T = 20;

%% Initial state = beta1 (from paper example)
alpha0 = [0.0036, 0.0530, 0.3177, 0.5960];   % resting (paper)
beta1 = [7.91, 0.1173, 0.3548, 0.5954];   % [V, m, n, h]
beta2 = [20.0188, 0.3138, 15.4266, 11.3144];

[ct1,cty1,ctly1] = get_solution(alpha0, beta1, params);
% sol2 = get_solution(alpha0, beta2, params);

Istim1 = - ctly1(:,1)/2;

Istim1_square = @(t) (t>=16 & t<=20) * 2.5;
Istim2_square = @(t) (t>=10 & t<=20) * -2.5;


%% Solve ODE
[t1,y1] = simulate_hh(alpha0, [0 params.T], Istim1_square, params);
[t2,y2] = simulate_hh(alpha0, [0 params.T], Istim2_square, params);

%% Plot


figure('Name','HH Simulation from beta1','NumberTitle','off');
subplot(4,2, [1,3,5,7])
plot3(y1(:,1), y1(:,3), y1(:,4),'b-', cty1(:,1),cty1(:,3),cty1(:,4),'r-');
% legend('cv1','cv2','sq1','sq2');
xlabel('V (mV)'); ylabel('n'); zlabel('h'); grid on;


subplot(4,2,2);
plot(t1,y1(:,1),'b-',ct1,cty1(:,1),'r-','LineWidth',1.2);
legend('Simulation','Calculus of Variations');
xlabel('t (ms)'); ylabel('V (mV)'); title('Membrane potential V(t)');

subplot(4,2,4);
plot(t,Istim1_square(t),'b-',ct1,Istim1,'r-','LineWidth',1.2);
xlabel('t (ms)'); ylabel('Istim');

subplot(4,2,6);
plot(t2,y2(:,1),'b-','LineWidth',1.2);
xlabel('t (ms)'); ylabel('V (mV)'); title('Membrane potential V(t)');

subplot(4,2,8);
plot(t,Istim2_square(t),'b-','LineWidth',1.2);
xlabel('t (ms)'); ylabel('Istim');



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
         - Istim - p.Ioff ) / p.C;
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
end

function [ah,bh] = ah_bh(V,Q)
ah = 0.07*Q*exp(-V/20);
bh = Q ./ (exp(0.1*(30 - V)) + 1);
end

function [an,bn] = an_bn(V,Q)
x = 0.1*(10 - V);
an = 0.01*Q*(10 - V) ./ (exp(x) - 1);
bn = 0.125*Q*exp(-V/80);
end
