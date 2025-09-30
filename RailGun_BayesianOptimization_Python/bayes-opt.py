import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.signal import find_peaks
from neuron import h, gui


df_name = 'results_defaults_df_v4.csv'
use_version = 4
load_df = False
do_grid = True
do_iters = True 

scatter_x = 'Amp'
scatter_y = 'StimDur'

# Parameter ranges
amp_range = [-1, 1]
stim_dur_range = [0.1, 4]
delta_e_range = [1, 100]

# Initial parameter grid
n_init_amp = np.array([-1, -0.5, -0.1, 0, 0.1, 0.5, 1])
n_init_stim = np.array([0.1, 0.5, 1, 2, 4])
n_init_delta_e = np.array([1, 2, 10, 20, 30])

# Optimization loop
max_iters = 200
pool_size = 4
save_every = 100

np.random.seed(123)


class SimParameters:
    def __init__(self):
        """Default parameters"""
        # Simulation
        self.tstop = 20         # [ms]
        self.dt = 0.025         # [ms]
        self.v_init = -63       # [mV] initialize this membrane voltage everywhere

        # Axon
        self.axon_length = 2000 # [microns]
        self.axon_diameter = 1  # [microns] (default = 1)
        self.axon_nseg = 2001   # 101 segments = 101 locations, must be odd for midpoint

        # Left to default
        self.axon_Ra = 170      # Axial/internal resistance [Ohms * cm] (default = 35.4)
        self.cm = 1             # Membrane capacitance [uF/cm^2]        (default = 1)
        # Ion channels
        self.gkbar = 25e-3    # Potassium conductance [S/cm^2]     (defualt = 0.036)
        self.gnabar = 120e-3  # Sodium conductance                 (defualt = 0.12)
        self.gl = 0.3e-3      # Leak conductance                   (defualt = 0.0003)
        self.ek = -80         # Potassium reversal potential [mV]  (default = -77)
        self.ena = 40         # Sodium reversal potential          (default = 50)
        self.el = -49         # Leak reversal potential            (default = -54.3)

def stim_range(axon, xmin, xmax, delay, dur, amp):
    density = 1 # stims / micron
    n_positions = int(max((xmax - xmin) * density, 1))
    positions = np.linspace(xmin, xmax, n_positions)

    stims = []
    for pos in positions:
        pos_factor = pos / axon.L
        stims.append(h.IClamp(axon(pos_factor)))
        stims[-1].delay = delay # [ms]
        stims[-1].dur = dur     # [ms]
        stims[-1].amp = amp / n_positions     # [nA]
    return stims

class Electrode:
    def __init__(self, pos, delay, amp=0.125, dur=1, width=0):
        self.pos = pos
        self.amp = amp
        self.dur = dur
        self.delay = delay
        self.width = width

    def stims(self, axon):
        """Symmetric biphasic square pulse"""
        pulse1 = stim_range(axon, self.pos-self.width/2, self.pos+self.width/2, self.delay,            self.dur/2,  self.amp)
        pulse2 = stim_range(axon, self.pos-self.width/2, self.pos+self.width/2, self.delay+self.dur/2, self.dur/2, -self.amp)
        return pulse1 + pulse2

class ElectrodeRailConfiguration:
    def __init__(self, sim_input, version=1):
        self.delay = 5 # [ms] simulate this long before any stimulations
        self.stim_amp = sim_input['Amp'] # [nA]       
        self.electrode_width = 0 # [microns]
        self.electrode_spacing = sim_input['deltaE'] # [microns]

        if version == 1:
            self.n_electrodes = 4
            self.electrode_interval = sim_input['StimDur']/2  # [ms]
            self.stim_dur = sim_input['StimDur'] # [ms]
        elif version == 2:
            self.n_electrodes = 4
            self.electrode_interval = sim_input['StimDur']  # [ms]
            self.stim_dur = 1 # [ms]
        elif version == 3:
            self.n_electrodes = 10
            self.electrode_interval = sim_input['StimDur']/2  # [ms]
            self.stim_dur = sim_input['StimDur'] # [ms]
        elif version == 4:
            self.n_electrodes = 4
            self.electrode_interval = sim_input['StimDur']/6  # [ms]
            self.stim_dur = sim_input['StimDur'] # [ms]
        else:
            raise "unknown version"

    def electrodes(self, axon_length):
        electrodes = []
        for i in range(self.n_electrodes):
            electrodes.append(
                Electrode(axon_length/2 + i*self.electrode_spacing, self.delay+i*self.electrode_interval, amp=self.stim_amp, dur=self.stim_dur, width=0)
            )
        return electrodes

class ElectrodeBounceConfiguration:
    def __init__(self, sim_input, version=1):
        self.delay = 8 # [ms] simulate this long before any stimulations
        self.cathode_width = 0
        self.dur = 1

        if version == 1:
            self.gap = sim_input['Gap']
            self.anode_amp = sim_input['AnodeAmp']
            self.amp = sim_input['Amp']
            self.anode_width = 100
        else:
            raise "unknown version"

    def electrodes(self, axon_length):
        big_anode = Electrode(axon_length/2 - self.gap - self.anode_width/2, 0, amp=self.anode_amp, dur=1e3, width=self.anode_width)
        cathode = Electrode(axon_length/2, self.delay, amp=self.amp, dur=self.dur, width=self.cathode_width)
        return [big_anode, cathode]
       
class SimResult:
    def __init__(self, params, electrodes):
        # Create axon
        axon = h.Section(name='axon')
        axon.insert('hh')

        ### Set simulation parameters ###
        h.tstop = params.tstop
        h.dt = params.dt
        h.v_init = params.v_init

        axon.L = params.axon_length
        axon.diam = params.axon_diameter
        axon.nseg = params.axon_nseg
        # axon.Ra = params.axon_Ra
        # for seg in axon:
        #     seg.cm = params.cm
        #     seg.hh.gkbar = params.gkbar
        #     seg.hh.gnabar = params.gnabar
        #     seg.hh.gl = params.gl
        #     seg.k_ion.ek = params.ek
        #     seg.na_ion.ena = params.ena
        #     seg.hh.el = params.el

        stims = []
        for electrode in electrodes:
            stims += electrode.stims(axon)

        # Set up time recording
        t_vec = h.Vector().record(h._ref_t)

        # Set up voltage recording at all segment centers
        normalized_positions = np.linspace(0, 1, axon.nseg)  # normalized positions

        # Record voltage at each segment
        voltage_vectors = []
        for x in normalized_positions:
            vec = h.Vector().record(axon(x)._ref_v)
            voltage_vectors.append(vec)

        # Run the simulation
        h.run()

        # Convert recordings to a 2D numpy array: time × position
        v_matrix = np.array(voltage_vectors)  # shape: (position, time)
        v_matrix = v_matrix.T                 # now: (time, position)

        self.t_vec = t_vec
        self.x_vec = np.linspace(0, axon.L, axon.nseg)
        self.v_matrix = v_matrix

    def count_peaks(self):
        spike_threshold = 10
        end1_over_time = self.v_matrix[:, 1]
        end_over_pos = self.v_matrix[-1,:]
        end2_over_time = self.v_matrix[:, -1]
        border = np.concatenate((end1_over_time, end_over_pos, np.flip(end2_over_time)))

        # with open('border.txt', 'w') as f:
        #     for item in border:
        #         f.write(f"{item}\n") 

        if np.sum(border > spike_threshold) == 0:
            return 0
        else:
            peaks, pos = find_peaks(border, height=spike_threshold, distance=10)
            return len(peaks)


def sim_input_to_string(sim_input):
    return f"Dur:{sim_input['StimDur']:.3f} deltaE:{sim_input['deltaE']:.1f} Amp:{sim_input['Amp']:.3f}"

def run_sim(sim_input):
    params = SimParameters()
    electrodes = ElectrodeRailConfiguration(sim_input, version=use_version).electrodes(params.axon_length)
    return SimResult(params, electrodes) 

def evaluate_initial_points(results_df):
    product = len(n_init_amp) * len(n_init_stim) * len(n_init_delta_e)
    print("Total combinations:", product)

    # Full-factorial grid
    Agrid, Sgrid, Egrid = np.meshgrid(n_init_amp, n_init_stim, n_init_delta_e, indexing='ij')
    init_params = np.column_stack((Agrid.ravel(), Sgrid.ravel(), Egrid.ravel()))

    # Evaluate initial points
    for i, (amp, stim_dur, delta_e) in enumerate(init_params):
        sim_input = {'deltaE': delta_e, 'StimDur': stim_dur, 'Amp': amp}
        n_spikes = run_sim(sim_input).count_peaks()
        print(f"Init {i+1}: {sim_input_to_string(sim_input)} => {n_spikes} spikes")
        results_df.loc[len(results_df)] = [amp, stim_dur, delta_e, n_spikes]

def iterate_optimization(results_df, max_iters, pool_size):
    for iteration in range(1, max_iters + 1):
        X = results_df[['Amp', 'StimDur', 'deltaE']].values
        Y = results_df['Nspike'].values

        # Gaussian Process Regression
        kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, Y)

        # Generate random candidates
        n_candidates = 10000
        cand_amp = np.random.uniform(*amp_range, n_candidates)
        cand_stim_dur = np.random.uniform(*stim_dur_range, n_candidates)
        cand_delta_e = np.random.randint(delta_e_range[0], delta_e_range[1] + 1, n_candidates)
        Xcand = np.column_stack((cand_amp, cand_stim_dur, cand_delta_e))

        # Predict posterior
        y_pred, y_std = gpr.predict(Xcand, return_std=True)
        y_sample = np.abs(y_pred + 0.05 * np.random.randn(n_candidates) * y_std - 1)

        # Thompson sampling: select candidates with lowest y_sample
        best_idxs = np.argpartition(y_sample, pool_size)[:pool_size]

        for idx in best_idxs:
            amp, stim_dur, delta_e = Xcand[idx]
            stim_dur = round(stim_dur, 2)
            sim_input = {'deltaE': delta_e, 'StimDur': stim_dur, 'Amp': amp}
            n_spikes = run_sim(sim_input).count_peaks()
            print(f"Iter {iteration}: {sim_input_to_string(sim_input)} => {n_spikes} spikes")
            results_df.loc[len(results_df)] = [amp, stim_dur, delta_e, n_spikes]

        if iteration % save_every == 0:
            results_df.to_csv(df_name, index=False)
            

if load_df:
    results_df = pd.read_csv(df_name)
else:
    results_df = pd.DataFrame([], columns=['Amp', 'StimDur', 'deltaE', 'Nspike'])

if do_grid:
    evaluate_initial_points(results_df)
    results_df.to_csv(df_name, index=False)
if do_iters:
    iterate_optimization(results_df, max_iters, pool_size)
    results_df.to_csv(df_name, index=False)

print("\n==== Optimization Finished ====\n")

# Plot 2D scatter
fig = plt.figure(figsize=(15, 5))
gs = fig.add_gridspec(1, 3)

scatter_ax = fig.add_subplot(gs[0, 0])
heatmap_ax = fig.add_subplot(gs[0, 1])
ax3d = fig.add_subplot(gs[0, 2], projection='3d')  # <- make it 3D
sc = scatter_ax.scatter(results_df[scatter_x], results_df[scatter_y], c=results_df['Nspike'], cmap='viridis')
fig.colorbar(sc, ax=scatter_ax, label="# of Spikes")
scatter_ax.set_xlabel(scatter_x)
scatter_ax.set_ylabel(scatter_y)
scatter_ax.grid(True)

heatmap_colorbar = [None] # list so it's mutable in closure
heatmap_sim_input = [None]

# -- Click callback
def on_click(event):
    if event.inaxes != scatter_ax or not event.dblclick:
        return

    x_click = event.xdata
    y_click = event.ydata

    # Find nearest point
    dists = np.sqrt((results_df[scatter_x] - x_click)**2 + (results_df[scatter_y] - y_click)**2)
    idx = dists.idxmin()

    # Run simulation
    heatmap_sim_input[0] = results_df.loc[idx]
    update_heatmap(heatmap_sim_input[0])

# -- Drag callback
def on_drag(event):
    if event.inaxes != scatter_ax or event.button != 1:
        return

    heatmap_sim_input[0][scatter_x] = event.xdata
    heatmap_sim_input[0][scatter_y] = event.ydata
    update_heatmap(heatmap_sim_input[0])

def update_heatmap(sim_input):
    sim = run_sim(sim_input)
    sim.count_peaks()

    vmin = min(-70, sim.v_matrix.min())
    vmax = max(20, sim.v_matrix.max())

    # Plot heatmap
    heatmap_ax.cla()
    hm = heatmap_ax.imshow(sim.v_matrix, aspect='auto', origin='lower',
               extent=[sim.x_vec[0], sim.x_vec[-1], sim.t_vec[0], sim.t_vec[-1]],
               cmap='jet', vmin=vmin,vmax=vmax)

    if heatmap_colorbar[0] is not None:
        heatmap_colorbar[0].remove()
    heatmap_colorbar[0] = fig.colorbar(hm, ax=heatmap_ax, label="Voltage (mV)")

    heatmap_ax.set_xlabel("Position (μm)")
    heatmap_ax.set_ylabel("Time (ms)")
    heatmap_ax.set_title(f"Heatmap: {sim_input_to_string(sim_input)}")
    fig.canvas.draw()

# -- Connect callback
cid = fig.canvas.mpl_connect('button_press_event', on_click)
# fig.canvas.mpl_connect('motion_notify_event', on_drag)

# Highlight Nspike == 1

# 3D plot
no_spike = results_df['Nspike'] == 0
ax3d.scatter(results_df['Amp'][no_spike], results_df['deltaE'][no_spike], results_df['StimDur'][no_spike], c='w')

one_spike = results_df['Nspike'] == 1
ax3d.scatter(results_df['Amp'][one_spike], results_df['deltaE'][one_spike], results_df['StimDur'][one_spike], c='r')
ax3d.set_xlabel("Amp")
ax3d.set_ylabel("ΔE")
ax3d.set_zlabel("PW")
ax3d.set_title("3D Scatter: Amp vs ΔE vs PW (1-spike only)")

plt.show()
