% Simulate AC stimulus at different amplitudes and the optimal stimulus

%% Simulation Parameters
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
params.dt = 0.001;

linewidth = 0.7;

% Resting state (V,m,n,h)
% alpha0 = [0.0036, 0.0530, 0.3177, 0.5960];   % values from the paper
alpha0 = [0.003620668648695   0.052955086827096   0.317732399760949   0.595994124739172];

% The period is the same for each simulation but we simulate multiple
% amplitudes
period = 0.5; % ms
amps = 0:25:200;

%% Find what the end states are by simulating 
end_states = zeros(length(amps), 4);

steadystate_t = cell(length(amps),1);
steadystate_y = cell(length(amps),1);
Istims_AC = {};

for i = 1:length(amps)
amp = amps(i);
Istims_AC{i} = @(t) amp*sin(2*pi * t/period); 
[t,y] = simulate_hh(alpha0, [0 params.T], Istims_AC{i}, params);

steadystate_t(i,:) = {t};
steadystate_y(i,:,:) = {y};

end_states(i,:)=y(end,:);
end

disp(end_states);

%% Get a solution to the BVP that optimizes the stimulus
ts=[]; % times
lys=[]; % lambda solutions
ys=[]; % solutions
Istims = cell(length(amps),1);
for i = 1:length(amps)
[t, y, ly] = get_solution(alpha0, end_states(i,:), params);
ts(i,:) = t;
ys(i,:,:) = y;
lys(i,:,:) = ly;
Istim = -ly(:,1)/2;
Istims{i} = @(t) interp1(ts(i,:), Istim, t, 'linear'); 
end

%% Verify BVP solution by simulating the same stimulus
forward_ys=cell(length(amps),1);
forward_ts=cell(length(amps),1);
forward_end_states = zeros(length(amps), 4);

for i=1:length(amps)
[t,y] = simulate_hh(alpha0, [0 params.T], Istims{i}, params);
forward_ts(i,:) = {t};
forward_ys(i,:,:) = {y};
forward_end_states(i,:)=y(end,:);
end

%% Plot
figure('Name','Plots')

customColors = jet(length(amps));

subplot(3,2,1);
for i=1:length(steadystate_t)
plot(steadystate_t{i},Istims_AC{i}(steadystate_t{i}),'LineWidth',linewidth)
hold on;
end
hold off;
ax = gca; % Get the current axes object
ax.ColorOrder = customColors;
ylabel 'current (nA)'; title 'Alternative Currents (AC)';

colormap(customColors)
cb = colorbar;               % show colorbar
clim([min(amps) max(amps)]); % scale colorbar to line indices
ylabel(cb, 'Amplitude');     % label colorbar

subplot(3,2,2);
for i=1:length(steadystate_t)
plot(steadystate_t{i},steadystate_y{i}(:,1),'LineWidth',linewidth)
hold on;
end
hold off;
ax = gca; % Get the current axes object
ax.ColorOrder = customColors;
ylabel V(mV); title 'Simulate AC currents until steady state';

subplot(3,2,3);
plot(ts(:,:)',-lys(:,:,1)'/2,'LineWidth',linewidth)
xlabel time(ms); ylabel 'current (nA)'; title 'Calculus of Variations: Optimal current to reach steady state';
ax = gca; % Get the current axes object
ax.ColorOrder = customColors;


subplot(3,2,4);
plot(ts(:,:)',ys(:,:,1)','LineWidth',linewidth)
ylabel V(mV); title 'Calculus of Variations Path';

ax = gca; % Get the current axes object
ax.ColorOrder = customColors;

subplot(3,2,6);
for i=1:length(forward_ts)
plot(forward_ts{i},forward_ys{i}(:,1),'LineWidth',linewidth)
hold on;
end
hold off;
ax = gca; % Get the current axes object
ax.ColorOrder = customColors;
xlabel time(ms); ylabel V(mV); title 'Simulate optimal current (should match the path)';


