
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.signal import find_peaks
from neuron import h, gui


df_name = 'results_ext_v1.csv'
load_df = False
do_grid = True
do_iters = True

scatter_x = 'Amp'
scatter_y = 'StimDur'

space = ['deltaE', 'StimDur', 'Amp']
configuration = 'rail'
use_version = 1

# space = ['Gap', 'AnodeAmp', 'Amp']
# configuration = 'bounce'
# use_version = 1

# space = ['Delay', 'AnodeAmp', 'Amp']
# configuration = 'bounce'
# use_version = 2
# space = ['Delay', 'Gap', 'Amp']
# configuration = 'bounce'
# use_version = 3
# space = ['Width', 'Gap', 'Amp']
# configuration = 'bounce'
# use_version = 4
# space = ['Reversal', 'AnodeAmp', 'Amp']
# configuration = 'bounce'
# use_version = 6

# Parameter ranges
# range1 = [1, 100]
# range2 = [0.1, 4]
# range3 = [-1, 1]

# Initial parameter grid
n_init1 = np.array([1, 20, 40, 70, 100])
n_init2 = np.array([0.1, 0.5, 1, 2, 4])
n_init3 = np.array([-10, -4, -2, -1, 0, 1, 2, 4, 10])

range1 = [min(n_init1), max(n_init1)]
range2 = [min(n_init2), max(n_init2)]
range3 = [min(n_init3), max(n_init3)]

# Optimization loop
max_iters = 100
pool_size = 4
save_every = 100

np.random.seed(123)


class Electrode:
    def __init__(self, pos, delay, amp=0.125, dur=1, reversal_time = 0.5, freq = None, radius = 10):
        """
        Electrode that stimulates at x = pos +- width/2 and delay < t < delay + dur.
        First phase provides current I=amp and second phase is charge balances.
        Combined phases take dur long and the second phase takes reversal_time of dur.
        """
        self.pos = pos # [microns]
        self.amp = amp # [nA]
        self.dur = dur # [ms]
        self.delay = delay #[ms]
        self.reversal_time = reversal_time
        self.freq = freq # [Hz]
        self.radius = radius # [microns] distance to axon

    def compute_extracellular_potential(self, x_positions, tvec, resistivity=300):
        """
        Compute Ve(x, t) for each position x at all times t.
        Resitivity (Ohms*cm)
        """
        ve_matrix = np.zeros((len(tvec), len(x_positions)))

        # Compute radial distance from each segment to the electrode
        dx = x_positions - self.pos
        r = np.sqrt(dx**2 + self.radius**2)  # Euclidean distance

        # Generate the current waveform
        i_wave = np.zeros_like(tvec)
        start = np.searchsorted(tvec, self.delay)
        end = np.searchsorted(tvec, self.delay + self.dur)

        if self.freq is None:
            reversal_point = start + int((end - start) * (1 - self.reversal_time))
            i_wave[start:reversal_point] = self.amp
            i_wave[reversal_point:end] = -self.amp * (1 - self.reversal_time) / self.reversal_time
        else:
            i_wave[start:end] = self.amp * np.sin(2 * np.pi * self.freq * (tvec[start:end] - self.delay) / 1000.0)

        # Point source model
        # V is I/4 pi sigma r
        ve_matrix = 1e2 * (resistivity / (4 * np.pi)) * np.outer(i_wave, 1.0 / r)
        # for i, r_val in enumerate(r):
        #     # Units: (Ohms*cm nA) / (um) -> 1e-5 V = 1e-2 mV
        #     ve_matrix[:, i] += 1e2 * (resistivity * i_wave) / (4 * np.pi * r_val)  # Ve in [mV] if units are consistent

        return ve_matrix

    def to_string(self):
        return print(f'amp:{self.amp} dur:{self.dur} radius:{self.radius} freq:{self.freq}')

class SimParameters:
    def __init__(self):
        """
        Default parameters
        """
        # Simulation
        self.tstop = 30         # [ms]
        self.dt = 0.025         # [ms]
        self.v_init = -63       # [mV] initialize this membrane voltage everywhere

        # Extracellular
        self.ext_resitivity = 300 # [Ohm*microns]

        # Axon
        self.axon_length = 2000 # [microns]
        self.axon_diameter = 1  # [microns]
        self.axon_nseg = 2001   # 101 segments = 101 locations, must be odd for midpoint
        
class SimResult:
    def __init__(self, params, electrodes):
        # Create axon
        axon = h.Section(name='axon')
        axon.insert('hh')
        axon.insert('extracellular')

        h.secondorder = 0  # Required for proper Ve support

        ### Set simulation parameters ###
        h.tstop = params.tstop
        h.dt = params.dt
        h.v_init = params.v_init

        axon.L = params.axon_length
        axon.diam = params.axon_diameter
        axon.nseg = params.axon_nseg

        # Set up time recording
        t_vec = h.Vector().record(h._ref_t)

        # Set up voltage recording at all segment centers
        normalized_positions = np.linspace(0, 1, axon.nseg)  # normalized positions

        # Record voltage at each segment
        voltage_vectors = []
        for x in normalized_positions:
            vec = h.Vector().record(axon(x)._ref_v)
            voltage_vectors.append(vec)

        x_positions = np.linspace(0, axon.L, axon.nseg)
        tvec = np.arange(0, h.tstop + h.dt, h.dt)

        # Apply Ve(x, t)
        ve_matrix = np.zeros((len(tvec), len(x_positions)))
        for electrode in electrodes:
            ve_matrix += electrode.compute_extracellular_potential(x_positions, tvec)

        ve = []
        tvec_hvec = h.Vector(tvec)
        for i, x in enumerate(normalized_positions):
            ve.append(h.Vector(ve_matrix[:, i]))
            ve[-1].play(axon(x)._ref_e_extracellular, tvec_hvec, 1)

        # Run the simulation
        h.run()

        # print(f'dt: {t_vec[1]-t_vec[0]:.5f}')

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

        if np.sum(border > spike_threshold) == 0:
            return 0
        else:
            peaks, pos = find_peaks(border, height=spike_threshold, distance=10)
            return len(peaks)

    def get_peaks_loc(self):
        spike_threshold = 10
        end1_over_time = self.v_matrix[:, 1]
        end_over_pos = self.v_matrix[-1,:]
        end2_over_time = self.v_matrix[:, -1]
        border = np.concatenate((end1_over_time, end_over_pos, np.flip(end2_over_time)))

        if np.sum(border > spike_threshold) == 0:
            return []
        else:
            peaks, pos = find_peaks(border, height=spike_threshold, distance=10)
            coords = []
            for peak in peaks:
                if peak < len(end1_over_time):
                    t_i = peak
                    x_i = 1
                elif peak < len(end1_over_time) + len(end_over_pos):
                    t_i = -1
                    x_i = peak - len(end1_over_time)
                else:
                    t_i = len(end2_over_time) - (peak - len(end1_over_time) - len(end_over_pos))
                    x_i = -1
                coords.append([self.x_vec[x_i], self.t_vec[t_i]])
            return np.array(coords)
                    
class ElectrodeRailConfiguration:
    def __init__(self, sim_input, version=1):
        self.delay = 5 # [ms] simulate this long before any stimulations
        self.stim_amp = sim_input['Amp'] # [nA]       
        self.electrode_width = 0 # [microns]
        self.electrode_spacing = sim_input['deltaE'] # [microns]
        self.electrode_dist = 30

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
                Electrode(axon_length/2 + i*self.electrode_spacing, self.delay+i*self.electrode_interval, amp=self.stim_amp, dur=self.stim_dur, radius=self.electrode_dist)
            )
        return electrodes

class ElectrodeBounceConfiguration:
    def __init__(self, sim_input, version=1):
        self.cathode_width = 0
        self.dur = 2
        self.electrode_dist = 30

        if version == 1:
            self.anode_reversal = 0
            self.anode_dur = 1e3
            self.init_time = 0
            self.delay = 8
            self.gap = sim_input['Gap']
            self.anode_amp = sim_input['AnodeAmp']
            self.amp = sim_input['Amp']
            self.anode_dist = 100
        elif version == 2:
            self.anode_reversal = 0
            self.anode_dur = 1e3
            self.init_time = 0
            self.delay = sim_input['Delay']
            self.gap = 10
            self.anode_amp = sim_input['AnodeAmp']
            self.amp = sim_input['Amp']
            self.anode_dist = 500
        elif version == 3:
            self.anode_reversal = 0
            self.anode_dur = 1e3
            self.init_time = 0
            self.delay = sim_input['Delay']
            self.gap = sim_input['Gap']
            self.anode_amp = -3
            self.amp = sim_input['Amp']
            self.anode_dist = 500
        elif version == 4:
            self.anode_reversal = 0
            self.anode_dur = 1e3
            self.init_time = 0
            self.delay = 8
            self.gap = sim_input['Gap']
            self.anode_amp = -3
            self.amp = sim_input['Amp']
            self.anode_dist = sim_input['Width']
        elif version == 5:
            self.anode_reversal = 0.3
            self.anode_dur = 10
            self.init_time = 5
            self.delay = sim_input['Delay']
            self.gap = 10
            self.anode_amp = sim_input['AnodeAmp']
            self.amp = sim_input['Amp']
            self.anode_dist = 500
        elif version == 6:
            self.anode_reversal = sim_input['Reversal']
            self.anode_dur = 10
            self.init_time = 5
            self.delay = 0
            self.gap = 10
            self.anode_amp = sim_input['AnodeAmp']
            self.amp = sim_input['Amp']
            self.anode_dist = 500
        else:
            raise "unknown version"

    def electrodes(self, axon_length):
        big_anode = Electrode(axon_length/2 - self.gap - self.anode_dist/2, self.init_time, amp=self.anode_amp, dur=self.anode_dur, radius=self.anode_radius, reversal_time=self.anode_reversal)
        cathode = Electrode(axon_length/2, self.init_time+self.delay, amp=self.amp, dur=self.dur, radius=self.electrode_dist)
        return [big_anode, cathode]
    
def sim_input_to_string(sim_input):
    return f"{space[0]}:{sim_input[0]:.3f} {space[1]}:{sim_input[1]:.3f} {space[2]}:{sim_input[2]:.3f}"

def run_sim(sim_input):
    params = SimParameters()
    sim_input = {space[0]: sim_input[0], space[1]: sim_input[1], space[2]: sim_input[2]}
    if configuration == 'rail':
        electrodes = ElectrodeRailConfiguration(sim_input, version=use_version).electrodes(params.axon_length)
    elif configuration == 'bounce':
        electrodes = ElectrodeBounceConfiguration(sim_input, version=use_version).electrodes(params.axon_length)
    else:
        raise "not a configuration"

    return SimResult(params, electrodes) 

def evaluate_initial_points(results_df):
    product = len(n_init1) * len(n_init2) * len(n_init3)
    print("Total combinations:", product)

    # Full-factorial grid
    Agrid, Sgrid, Egrid = np.meshgrid(n_init1, n_init2, n_init3, indexing='ij')
    init_params = np.column_stack((Agrid.ravel(), Sgrid.ravel(), Egrid.ravel()))

    # Evaluate initial points
    for i, (c1,c2,c3) in enumerate(init_params):
        sim_input = [c1,c2,c3]
        n_spikes = run_sim(sim_input).count_peaks()
        print(f"Init {i+1}: {sim_input_to_string(sim_input)} => {n_spikes} spikes")
        results_df.loc[len(results_df)] = sim_input + [n_spikes]

def iterate_optimization(results_df, max_iters, pool_size):
    for iteration in range(1, max_iters + 1):
        X = results_df[space].values
        Y = results_df['Nspike'].values

        # Gaussian Process Regression
        kernel = RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X, Y)

        # Generate random candidates
        n_candidates = 10000
        cand1 = np.random.uniform(*range1, n_candidates)
        cand2 = np.random.uniform(*range2, n_candidates)
        cand3 = np.random.uniform(*range3, n_candidates)
        Xcand = np.column_stack((cand1, cand2, cand3))

        # Predict posterior
        y_pred, y_std = gpr.predict(Xcand, return_std=True)
        y_sample = np.abs(y_pred + 0.05 * np.random.randn(n_candidates) * y_std - 1)

        # Thompson sampling: select candidates with lowest y_sample
        best_idxs = np.argpartition(y_sample, pool_size)[:pool_size]

        for idx in best_idxs:
            sim_input = Xcand[idx].tolist()
            n_spikes = run_sim(sim_input).count_peaks()
            print(f"Iter {iteration}: {sim_input_to_string(sim_input)} => {n_spikes} spikes")
            row = sim_input + [n_spikes]
            results_df.loc[len(results_df)] = row

        if iteration % save_every == 0:
            results_df.to_csv(df_name, index=False)
            

if load_df:
    results_df = pd.read_csv(df_name)
else:
    results_df = pd.DataFrame([], columns=space + ['Nspike'])

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
ax3d = fig.add_subplot(gs[0, 2], projection='3d', computed_zorder=False)  # <- make it 3D
sc = scatter_ax.scatter(results_df[scatter_x], results_df[scatter_y], c=results_df['Nspike'], cmap='viridis')
fig.colorbar(sc, ax=scatter_ax, label="# of Spikes")
scatter_ax.set_xlabel(scatter_x)
scatter_ax.set_ylabel(scatter_y)
scatter_ax.grid(True)

heatmap_colorbar = [None] # list so it's mutable in closure
heatmap_sim_input = [None]

points = np.array([results_df[space[0]][:], results_df[space[1]][:], results_df[space[2]][:]]).T
# print(size)

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
    update_heatmap(heatmap_sim_input[0], peaks=heatmap_sim_input[0]['Nspike'])

def on_click_3d(event):
    if event.inaxes != ax3d or not event.dblclick:
        return

    # Project 3D points to 2D screen coordinates
    proj = np.array([proj3d.proj_transform(x, y, z, ax3d.get_proj())[:2] for x, y, z in points])
    x2, y2 = event.x, event.y  # pixel coords of click

    # Get pixel positions of projected data points
    trans = ax3d.transData.transform
    dists = np.linalg.norm(trans(proj) - np.array([x2, y2]), axis=1)
    idx = np.argmin(dists)

    if dists[idx] < 20:  # 20-pixel threshold (tweak as needed)
        # selected_point = points[idx]
        # print(f"Selected point: Amp={selected_point[0]}, ΔE={selected_point[1]}, PW={selected_point[2]}")

        heatmap_sim_input[0] = results_df.loc[idx]
        update_heatmap(heatmap_sim_input[0], peaks=heatmap_sim_input[0]['Nspike'])
    else:
        print("No point selected")

fig.canvas.mpl_connect('button_press_event', on_click_3d)# -- Drag callback

def on_drag(event):
    if event.inaxes != scatter_ax or event.button != 1:
        return

    heatmap_sim_input[0][scatter_x] = event.xdata
    heatmap_sim_input[0][scatter_y] = event.ydata
    update_heatmap(heatmap_sim_input[0])

number = ['','']
def on_press(event):
    key = event.key
    if key in '1234567890-.':
        if number[0] == '':
            number[0] = key
            number[1] = ''
        else:
            number[1] += key
    elif key=='enter':
        try:
            param = int(number[0])
            field = space[param-1]
            val = float(number[1])
            heatmap_sim_input[0][field] = val
            update_heatmap(heatmap_sim_input[0])
        except Exception:
            pass
        number[0] = ''
       

def update_heatmap(sim_input, peaks=None):
    sim = run_sim(sim_input)
    coords = sim.get_peaks_loc()

    vmin = min(-70, sim.v_matrix.min())
    vmax = max(20, sim.v_matrix.max())

    # Plot heatmap
    heatmap_ax.cla()
    hm = heatmap_ax.imshow(sim.v_matrix, aspect='auto', origin='lower',
               extent=[sim.x_vec[0], sim.x_vec[-1], sim.t_vec[0], sim.t_vec[-1]],
               cmap='jet', vmin=vmin,vmax=vmax)
    if len(coords) > 0:
        heatmap_ax.scatter(coords[:,0], coords[:,1], c='k')

    if heatmap_colorbar[0] is not None:
        heatmap_colorbar[0].remove()
    heatmap_colorbar[0] = fig.colorbar(hm, ax=heatmap_ax, label="Voltage (mV)")

    heatmap_ax.set_xlabel("Position (μm)")
    heatmap_ax.set_ylabel("Time (ms)")
    if peaks is None:
        heatmap_ax.set_title(f"Heatmap: {sim_input_to_string(sim_input)}")
    else:
        heatmap_ax.set_title(f"Heatmap: {sim_input_to_string(sim_input)}. {peaks:.1f} spikes")
    fig.canvas.draw()

# -- Connect callback
cid = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_press)# -- Click callback
# fig.canvas.mpl_connect('motion_notify_event', on_drag)

ax3d.scatter(results_df[space[0]], results_df[space[1]], results_df[space[2]], c=results_df['Nspike'], cmap='viridis')

one_spike = results_df['Nspike'] == 1
ax3d.scatter(results_df[space[0]][one_spike], results_df[space[1]][one_spike], results_df[space[2]][one_spike], c='r')
ax3d.set_xlabel(space[0])
ax3d.set_ylabel(space[1])
ax3d.set_zlabel(space[2])
ax3d.set_title("3D Scatter: Amp vs ΔE vs PW (1-spike only)")

plt.show()
