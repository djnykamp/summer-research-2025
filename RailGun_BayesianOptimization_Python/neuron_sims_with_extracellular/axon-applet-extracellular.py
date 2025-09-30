import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from neuron import h, gui

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
            if self.reversal_time > 0:
                reversal_point = start + int((end - start) * (1 - self.reversal_time))
                i_wave[start:reversal_point] = self.amp
                i_wave[reversal_point:end] = -self.amp * (1 - self.reversal_time) / self.reversal_time
            else:
                i_wave[start:end] = self.amp
                
        else:
            i_wave[start:end] = self.amp * np.sin(2 * np.pi * self.freq * (tvec[start:end] - self.delay) / 1000.0)

        # Point source model
        # V is I/4 pi sigma r
        for i, r_val in enumerate(r):
            # Units: (Ohms*cm nA) / (um) -> 1e-5 V = 1e-2 mV
            ve_matrix[:, i] += 1e2 * (resistivity * i_wave) / (4 * np.pi * r_val)  # Ve in [mV] if units are consistent

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
        self.dt = 0.0125         # [ms]
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
electrodes = [Electrode(500,0,amp=10,dur=30)]

fig, ax = plt.subplots()

sel = [-1]
number = ['']
heatmap_colorbar = [None] # list so it's mutable in closure

def on_press(event):
    key = event.key

    if key == 'x':
        electrodes.clear()
    elif key == 'b':
        electrodes.pop()
    elif event.key == 'p':
        print(electrodes)
        return

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
        number_old = number[0]
        if key == 'e':
            number[0] = ''
            num = int(number_old)
            sel[0] = num
            print(electrodes[sel[0]].to_string())
            return
        elif key == 'a':
            number[0] = ''
            num = float(number_old)
            electrodes[sel[0]].amp = num
        elif key == 'd':
            number[0] = ''
            num = float(number_old)
            electrodes[sel[0]].dur = num
        elif key == 'w':
            number[0] = ''
            num = int(number_old)
            electrodes[sel[0]].radius = num
        elif key == 'r':
            number[0] = ''
            num = float(number_old)
            electrodes[sel[0]].reversal_time = num
        elif key == 'y':
            number[0] = ''
            num = float(number_old)
            electrodes[sel[0]].freq = num
  
    if len(electrodes) > 0:
        print(electrodes[sel[0]].to_string())

    update_heatmap(params, electrodes)



def on_click(event):
    if event.inaxes != ax or not event.dblclick:
        return

    x_click = event.xdata
    y_click = event.ydata

    if sel[0] == -1:
        new_electrode = Electrode(x_click, y_click)
        electrodes.append(new_electrode)
        sel[0] = -1
    else:
        electrodes[sel[0]].pos = x_click
        electrodes[sel[0]].delay = y_click

    print(electrodes[sel[0]].to_string())

    # Run simulation
    update_heatmap(params, electrodes)

def update_heatmap(params, electrodes):
    sim = SimResult(params,electrodes)

    vmin = min(0, sim.v_matrix.min())
    vmax = max(0, sim.v_matrix.max())

    # Plot heatmap
    ax.cla()
    hm = ax.imshow(sim.v_matrix, aspect='auto', origin='lower',
               extent=[sim.x_vec[0], sim.x_vec[-1], sim.t_vec[0], sim.t_vec[-1]],
               cmap='jet', vmin=vmin,vmax=vmax)

    if heatmap_colorbar[0] is not None:
        heatmap_colorbar[0].remove()
    heatmap_colorbar[0] = fig.colorbar(hm, ax=ax, label="Voltage (mV)")

    ax.set_xlabel("Position (μm)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Heatmap")
    fig.canvas.draw()


# -- Connect callback
cid = fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_press)# -- Click callback

update_heatmap(params, electrodes)
plt.show()
