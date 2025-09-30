import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from neuron import h, gui

np.random.seed(123)

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
    def __init__(self, pos, delay, amp=0.125, dur=1, width=0, reversal_time = 0.5, freq = None):
        """Electrode that stimulates at x = pos +- width/2 and delay < t < delay + dur.
        First phase provides current I=amp and second phase is charge balances.
        Combined phases take dur long and the second phase takes reversal_time of dur."""
        self.pos = pos
        self.amp = amp
        self.dur = dur
        self.delay = delay
        self.width = width
        self.reversal_time = reversal_time
        self.freq = freq # [Hz]

        # these must be variables to stay in the scope so that Vector.play works
        self.vec_t = None
        self.vec_i = None

    def stims(self, axon):
        """Charge balanced biphasic square pulse or a sustained frequency"""
        if self.freq is None or self.freq == 0:
            pulses = stim_range(axon, self.pos-self.width/2, self.pos+self.width/2, self.delay,                                self.dur*(1-self.reversal_time),  self.amp)
            if self.reversal_time > 0:
                # charge balance
                a = -self.amp *(1-self.reversal_time)/self.reversal_time
                pulses += stim_range(axon, self.pos-self.width/2, self.pos+self.width/2, self.delay+self.dur*(1-self.reversal_time), self.dur * self.reversal_time, a)
            return pulses
        else:
            stims = stim_range(axon, self.pos-self.width/2, self.pos+self.width/2, self.delay, self.dur, 0)

            tvec = np.arange(0, h.tstop + h.dt, h.dt) # time vector [ms]
            stim_waveform = self.amp/len(stims) * np.sin(2 * np.pi * self.freq * (tvec - self.delay) / 1000.0)  # time in sec
            self.vec_t = h.Vector(tvec)
            self.vec_i = h.Vector(stim_waveform)

            for stim in stims:
                self.vec_i.play(stim._ref_amp, self.vec_t, 1)  # 1 = linear interpolation

            return stims

class SimParameters:
    def __init__(self):
        """Default parameters"""
        # Simulation
        self.tstop = 20         # [ms]
        self.dt = 0.01         # [ms]
        self.v_init = -63       # [mV] initialize this membrane voltage everywhere

        # Axon
        self.axon_length = 2000 # [microns]
        self.axon_diameter = 1  # [microns]
        self.axon_nseg = 2001   # 101 segments = 101 locations, must be odd for midpoint

        # # Disabled
        # self.axon_Ra = 170      # Axial/internal resistance [Ohms * cm] (default = 35.4)
        # self.cm = 1             # Membrane capacitance [uF/cm^2]        (default = 1)
        # # Ion channels
        # self.gkbar = 25e-3    # Potassium conductance [S/cm^2]     (defualt = 0.036)
        # self.gnabar = 120e-3  # Sodium conductance                 (defualt = 0.12)
        # self.gl = 0.3e-3      # Leak conductance                   (defualt = 0.0003)
        # self.ek = -80         # Potassium reversal potential [mV]  (default = -77)
        # self.ena = 40         # Sodium reversal potential          (default = 50)
        # self.el = -49         # Leak reversal potential            (default = -54.3)
        
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
        # normalized_positions = np.linspace(0, 1, axon.nseg)  # normalized positions

        normalized_positions = [seg.x for seg in axon]
        # for x in axon:
        #     print(dir(x))
        #     print(x.node_index())
        #     print(x.sec)
        #     print(x.x)
        #     break

        # Record voltage at each segment
        voltage_vectors = []
        m_vectors = []
        h_vectors = []
        n_vectors = []
        for x in normalized_positions:
            voltage_vectors.append(h.Vector().record(axon(x)._ref_v))
            m_vectors.append(h.Vector().record(axon(x).hh._ref_m))
            h_vectors.append(h.Vector().record(axon(x).hh._ref_h))
            n_vectors.append(h.Vector().record(axon(x).hh._ref_n))

        # Run the simulation
        h.run()

        # Convert recordings to a 2D numpy array: time × position
        v_matrix = np.array(voltage_vectors)  # shape: (position, time)
        m_matrix = np.array(m_vectors)
        h_matrix = np.array(h_vectors)
        n_matrix = np.array(n_vectors)
        v_matrix = v_matrix.T                 # now: (time, position)

        self.t_vec = t_vec
        self.x_vec = np.linspace(0, axon.L, axon.nseg)
        self.v_matrix = v_matrix
        self.m_matrix = m_matrix.T
        self.h_matrix = h_matrix.T
        self.n_matrix = n_matrix.T

    def count_peaks(self):
        spike_threshold = -20
        end1_over_time = self.v_matrix[:, 1]
        end_over_pos = self.v_matrix[-1,:]
        end2_over_time = self.v_matrix[:, -1]
        border = np.concatenate((end1_over_time, end_over_pos, np.flip(end2_over_time)))
        if np.sum(border > spike_threshold) == 0:
            return 0
        else:
            peaks, _ = find_peaks(border, height=spike_threshold, distance=10)
            return len(peaks)

params = SimParameters()
electrodes = []

fig, ax = plt.subplots()

sel = [-1]
number = ['']
heatmap_colorbar = [None] # list so it's mutable in closure
display_channels = [False]

def on_press(event):
    key = event.key

    if key == 'x':
        electrodes.clear()
    elif key == 'b':
        electrodes.pop()
    elif key == 'p':
        print(electrodes)
        return
    elif key == 'c':
        display_channels[0] = not display_channels[0]

    elif key in '1234567890-.':
        number[0] += key
        return
    elif number[0] == '':
        if key == 'a':
            electrodes[sel[0]].amp *= 1.1
        elif key == 'A':
            electrodes[sel[0]].amp /= 1.03
        elif key == 'n':
            electrodes[sel[0]].amp *= -1
        elif key == 'd':
            electrodes[sel[0]].dur *= 1.5
        elif key == 'D':
            electrodes[sel[0]].dur /= 1.5
        elif key == 'w':
            electrodes[sel[0]].width += 20
        elif key == 'W':
            electrodes[sel[0]].width -= 20
        elif key == 'e':
            sel[0] = -1
    else:
        if key == 'e':
            num = int(number[0])
            number[0] = ''
            sel[0] = num
            return
        elif key == 'a':
            num = float(number[0])
            number[0] = ''
            electrodes[sel[0]].amp = num
        elif key == 'd':
            num = float(number[0])
            number[0] = ''
            electrodes[sel[0]].dur = num
        elif key == 'w':
            num = int(number[0])
            number[0] = ''
            electrodes[sel[0]].width = num
        elif key == 'r':
            num = float(number[0])
            number[0] = ''
            electrodes[sel[0]].reversal_time = num
        elif key == 'y':
            num = float(number[0])
            number[0] = ''
            electrodes[sel[0]].freq = num
  
    if len(electrodes) > 0:
        print(f'amp:{electrodes[sel[0]].amp} dur:{electrodes[sel[0]].dur} width:{electrodes[sel[0]].width}')

    update_heatmap(params, electrodes)



def on_click(event):
    if event.inaxes != ax or not event.dblclick:
        return

    x_click = event.xdata
    y_click = event.ydata

    if sel[0] == -1:
        new_electrode = Electrode(y_click, x_click)
        electrodes.append(new_electrode)
        sel[0] = -1
    else:
        electrodes[sel[0]].delay = x_click
        electrodes[sel[0]].pos = y_click

    print(f'amp:{electrodes[sel[0]].amp} dur:{electrodes[sel[0]].dur}')

    # Find nearest point
    # dists = np.sqrt((results_df[scatter_x] - x_click)**2 + (results_df[scatter_y] - y_click)**2)
    # idx = dists.idxmin()

    # Run simulation
    update_heatmap(params, electrodes)

def update_heatmap(params, electrodes):
    sim = SimResult(params,electrodes)

    vmin = min(-70, sim.v_matrix.min())
    vmax = max(20, sim.v_matrix.max())

    # Plot heatmap
    ax.cla()
    if heatmap_colorbar[0] is not None:
        heatmap_colorbar[0].remove()
        heatmap_colorbar[0] = None

    if display_channels[0]:
        ax.imshow(np.stack([sim.m_matrix,sim.h_matrix,sim.n_matrix], axis=-1), aspect='auto', origin='lower',)
    else:
        hm = ax.imshow(sim.v_matrix.T, aspect='auto', origin='lower',
                   extent=[sim.t_vec[0], sim.t_vec[-1], sim.x_vec[0], sim.x_vec[-1]],
                   cmap='jet', vmin=vmin,vmax=vmax)

        heatmap_colorbar[0] = fig.colorbar(hm, ax=ax, label="Voltage (mV)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position (μm)")
    ax.set_title("Heatmap")
    fig.canvas.draw()
    
    # toolbar = fig.canvas.manager.toolbar
    # toolbar.pan()


# -- Connect callback
cid = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_press)# -- Click callback

update_heatmap(params, electrodes)
plt.show()
