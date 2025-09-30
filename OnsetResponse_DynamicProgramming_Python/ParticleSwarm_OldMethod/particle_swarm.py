import numpy as np
import itertools
import math
from neuron import h, gui
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from plotly.subplots import make_subplots

### Main parameters ###

# 1/n ms period => n kHz frequency
period = 1/2 # ms
dt = 0.01 # 20 dt per cycle as is
 
v_init = -64.975

n_cycles = 100
final_amplitude = 150
possible_amps = np.array([0.9,1,1.1])
# possible_amps = np.array([1])

max_iter = 2

# voxels in (v,m,h,n) space
voxel_dim = np.array([0.05, 0.05, 0.05, 0.05])
space_lower_bounds = np.array([-80, 0, 0, 0])
space_upper_bounds = np.array([ 0, 1, 1, 1])

n_voxels = np.array((space_upper_bounds-space_lower_bounds)/voxel_dim, dtype=int)

print(f'voxels {n_voxels}, max particles: {np.prod(n_voxels)}')

densify_size = 0.5 # units of voxels
densify_neighbor_radius = 3 # units how many of voxels away

perturbation_profile = np.array([-1,0,1])
perturbation_steps = [0.1, 0.0005, 0.0005, 0.0005]

### Other definitions ###

steps_per_period = int(period / dt)
print(f'{steps_per_period} steps per period')

axon = h.Section(name='axon')
axon.insert('hh')
axon_seg = axon(0.5)

h.secondorder=1

stim = h.IClamp(axon_seg)
stim.delay = 0
stim.dur = 1e9   # always on

counts = []

print_particles = False
def particle_msg(msg):
    if print_particles:
        print(msg)

# State of a simulation: time, voltage, and ion channels
class State:
    def __init__(self, v, m, h, n):
        self.v = v
        self.m = m
        self.h = h
        self.n = n

    def to_vec(self):
        """from_vec is State(*vec)"""
        return np.array([self.v, self.m, self.h, self.n])

    def state_string(self):
        """ This returns the full decimal places """
        return f'({self.v}, {self.m}, {self.h}, {self.n})'

    def perturb_state(self, steps=[0.01, 0.01, 0.01, 0.01], include_center=False):
        """
        Generate a grid of perturbed states around a given np.void `state`.

        Parameters:
            state: np.void of dtype state_dt
            steps: list of step sizes [v_step, m_step, h_step, n_step]
            include_center: bool, whether to include the original unperturbed state
        """
        delta_options = []
        for step in steps:
            if step == 0:
                delta_options.append([0])
            else:
                delta_options.append(perturbation_profile*step)

            if not include_center:
                delta_options[-1] = delta_options[-1][delta_options[-1] != 0]

        perturbations = list(itertools.product(*delta_options))

        perturbed_states = []
        for i, deltas in enumerate(perturbations):
            perturbed_states.append(State(*(self.to_vec() + deltas)))

        return perturbed_states

class SimulationState:
    def __init__(self, t, state):
        self.t = t
        self.state = state

class Particle:
    def __init__(self, amplitudes, state: SimulationState, period, cost, perturbation=[], interpolated=None, prev_state=None):
        self.amplitudes = list(amplitudes)
        self.perturbation = perturbation
        self.state = state
        self.period = period
        self.cost = cost
        self.interpolated = interpolated
        self.prev_state = prev_state

weight_v = 1
weight_current = 0.1
def cost_function(voltage, current):
    return weight_v*max(voltage-5,0)**2 + weight_current*(current/final_amplitude)**2

def cost_of_path(cycle_amplitudes, results):
    cost = 0
    for i in range(len(cycle_amplitudes)):
        cost += cost_function(results[1+i].state.v, cycle_amplitudes[i])
    return cost

### Simulation functions ###

def load_state(state):
    
    # Call finitialize() to make sure hh.gna and the like are not nan
    h.finitialize(state.state.v)
    h.t = state.t
    axon_seg.hh.m = state.state.m
    axon_seg.hh.h = state.state.h
    axon_seg.hh.n = state.state.n

def get_state():
    return SimulationState(h.t, State(axon_seg.v, axon_seg.hh.m, axon_seg.hh.h, axon_seg.hh.n))

def simulate_forward_one_cycle(state, amplitude):
    h.dt = dt
    load_state(state)

    for step in range(0, steps_per_period):
        stim.amp = amplitude * np.sin(2 * np.pi * step / steps_per_period)
        h.fadvance()

    return get_state()

def simulate_backward_one_cycle(state, amplitude):
    """
    Could return None if the simulation diverges
    """
    h.dt = -dt
    load_state(state)

    if math.isnan(axon_seg.hh.gna):
        raise "Already NaN"
 
    for step in range(steps_per_period, 0, -1):
        stim.amp = amplitude * np.sin(2 * np.pi * step / steps_per_period)
        h.fadvance()
    
    if math.isnan(axon_seg.v):
        return None

    return get_state()

def simulate_i_wave(cycle_amplitudes):
   
    h.finitialize(v_init)
    h.t = 0
    results = [get_state()]

    for (i, a) in enumerate(cycle_amplitudes):
        results.append(simulate_forward_one_cycle(results[i], a))

    return results


def stim_steady_state(amplitude):

    max_cycles = 500

    h.finitialize(v_init)
    h.t = 0

    # there's a point before the first cycle and after each
    results = [get_state()]

    steps_within = 20
    threshhold = 0.00001

    for i in range(max_cycles-1):
        results.append(simulate_forward_one_cycle(results[i], amplitude))

        if i < steps_within:
            continue

        recent = results[(i-steps_within):(1+i)]
        dvs = np.array([abs(prev.state.v - results[1+i].state.v) for prev in recent])
        dms = np.array([abs(prev.state.m - results[1+i].state.m) for prev in recent])
        dhs = np.array([abs(prev.state.h - results[1+i].state.h) for prev in recent])
        dns = np.array([abs(prev.state.n - results[1+i].state.n) for prev in recent])

        if sum(dvs > threshhold) + sum(dms > threshhold) + sum(dhs > threshhold) + sum(dns > threshhold) == 0:
            print(f'converged after {i} cycles, t={results[i].t}, dv={dvs[-1]}')
            return results, results[0], results[1+i]
    raise "Did not converge"

### Particle functions ###

def next_grid_iteration(particles):
    """
    Given the current particles, simulate backward one cycle with perturbations
    """
    counts.append({'nan':[], 'add':[], 'reject':[], 'out':[], 'notnan':[]})
    new_particles = []
    voxels = {}
    for p in particles:
        particle_msg(f'adding from state {p.state.state.state_string()}')
        for a in final_amplitude*possible_amps:

            if len(particles) == 1:
                perturbed_states = p.state.state.perturb_state(steps=perturbation_steps, include_center=True)
            else:
                perturbed_states = [p.state.state]
            for sim_state in perturbed_states:

                diff = sim_state.to_vec() - p.state.state.to_vec()
                
                new_state = simulate_backward_one_cycle(SimulationState(p.state.t, sim_state) ,a)

                if new_state is None:
                    counts[-1]['nan'].append(sim_state)
                    particle_msg('got NaN')
                    continue

                out_of_bounds_msg = out_of_bounds(new_state.state.to_vec())
                if  out_of_bounds_msg is not None:
                    counts[-1]['out'].append(sim_state)
                    particle_msg(f'Out: {out_of_bounds_msg} from state: {new_state.state.state_string()}')
                    continue

                counts[-1]['notnan'].append(sim_state)

                extra_cost = cost_function(new_state.state.v, a)
                cycle_amps = p.amplitudes.copy()
                cycle_amps.append(a)
                new_perturbation = p.perturbation.copy()
                new_perturbation.append(diff)
                add_particle = Particle(cycle_amps, new_state, p.period-1, p.cost+extra_cost,perturbation=new_perturbation, prev_state=sim_state)
                insert_particle(new_particles, voxels, add_particle)



    print(f"add:{len(counts[-1]['add'])} nan:{len(counts[-1]['nan'])} out:{len(counts[-1]['out'])} reject:{len(counts[-1]['reject'])}")

    for p in densify_point_cloud(new_particles, densify_size, neighbor_radius = densify_neighbor_radius):
        insert_particle(new_particles, voxels, p)

    print(f"add:{len(counts[-1]['add'])} nan:{len(counts[-1]['nan'])} out:{len(counts[-1]['out'])} reject:{len(counts[-1]['reject'])}")

    return new_particles

def out_of_bounds(state_vec):
    for i, (value, lo, hi) in enumerate(zip(state_vec, space_lower_bounds, space_upper_bounds)):
        if not (lo <= value <= hi):
            return f"{['v','m','h','n'][i]} = {value} is not in [{lo}, {hi}]"

def voxel_index(state):
    """
    Assume state is in bounds
    """
    state_vec = state.to_vec()
    return tuple(np.minimum(np.array((state_vec - space_lower_bounds) / voxel_dim, dtype=int), n_voxels - 1))

def insert_particle(particles, voxels, new_particle):
    """
    Add new_particle to particles but only if the voxel it is better than what's in the voxel
    """
    index = voxel_index(new_particle.state.state)

    voxel_particle_idx = voxels.get(index)

    if voxel_particle_idx is not None:
    # if False:
        voxel_particle = particles[voxel_particle_idx]

        if voxel_particle.cost < new_particle.cost:
            particle_msg('rejected: costly particle')
            counts[-1]['reject'].append(new_particle.state.state)
            return

        counts[-1]['add'].remove(voxel_particle.state.state)
        counts[-1]['reject'].append(voxel_particle.state.state)
        counts[-1]['add'].append(new_particle.state.state)

        particle_msg(f'particle replaced: {new_particle.state.state.state_string()} from {voxel_particle.state.state.state_string()}')
        particles[voxel_particle_idx] = new_particle
        return

    voxels[index] = len(particles)
    particles.append(new_particle)

    particle_msg(f'particle added: {new_particle.state.state.state_string()}')
    counts[-1]['add'].append(new_particle.state.state)


### Densify ###

def to_point(p: Particle):
    return p.state.state.to_vec() / voxel_dim # point in terms of voxels

def interpolate_line(p1: Particle, p2: Particle, step_size):
    """Return interpolated points from p1 to p2 spaced by step_size."""
    pt1 = to_point(p1)
    pt2 = to_point(p2)
    dist = np.linalg.norm(pt1 - pt2)
    if dist < step_size:
        return [], []  # No interpolation needed
    n_steps = int(np.floor(dist / step_size))
    return [interpolate_particles(p1, p2, (i / n_steps)) for i in range(1, n_steps)], np.array([pt1 + (pt2 - pt1) * (i / n_steps) for i in range(1, n_steps)])

def interpolate_particles(p1, p2, t):
    amplitudes = (1-t)*np.array(p1.amplitudes) + t*np.array(p2.amplitudes)
    state = (1-t)*p1.state.state.to_vec() + t*p2.state.state.to_vec()
    prev_state = None
    if p1.prev_state is not None and p2.prev_state is not None:
        prev_state = State(*((1-t)*p1.prev_state.to_vec() + t*p2.prev_state.to_vec()))
    state = (1-t)*p1.state.state.to_vec() + t*p2.state.state.to_vec()
    sim_state = SimulationState((1-t)*p1.state.t + t*p1.state.t, State(*state))
    period = p1.period
    cost = (1-t)*p1.cost + t*p2.cost
    perturbation = (1-t)*np.array(p1.perturbation) + t*np.array(p2.perturbation)
    return Particle(amplitudes, sim_state, period, cost, perturbation.tolist(), interpolated=True, prev_state=prev_state)

def densify_point_cloud(particles, voxel_size, neighbor_radius=None):
    """
    Fills in new points between existing points to achieve uniform density.
    
    Args:
        particles: array of Particles
        voxel_size (float): Desired spacing between points.
        neighbor_radius (float): Optional. Radius to search for neighbors. Default: 2 * voxel_size.
    
    Returns:
        array of Particles to add
    """

    if len(particles) == 0:
        return []
    
    if neighbor_radius is None:
        neighbor_radius = 2 * voxel_size

    points = np.array([to_point(p) for p in particles])

    tree = cKDTree(points)
    add_particles = []
    add_points = []
    seen_pairs = set()

    for i in range(len(particles)):
        p = particles[i]
        pt = points[i]

        indices = tree.query_ball_point(pt, r=neighbor_radius)
        for j in indices:
            if i >= j:
                continue  # Avoid duplicate or self-pair
            pair = tuple(sorted((i, j)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            p2 = particles[j]
            # print('interpolate!')
            new_particles, new_pts = interpolate_line(p, p2, voxel_size)
            # print(len(new_pts))
            if len(new_pts):
                add_particles.extend(new_particles)
                add_points.extend(new_pts)

    # add_particles = np.array(add_particles).flatten()
    # return np.array(add_particles).flatten()
    return add_particles


### Particles iterations ###

initialize_result, initial, final = stim_steady_state(final_amplitude)
print(f"initial state {initial.state.state_string()}")
print(f"goal state {final.state.state_string()}")

current_particles = [Particle([], final, n_cycles, 0)]

all_p = []

all_p.extend(current_particles)

for i in range(max_iter):
    print(f'===== step {i}, {len(current_particles)} particles =====')

    current_particles = next_grid_iteration(current_particles)
    all_p.extend(current_particles)


trajectories = {'v':[], 'm':[], 'h':[], 'n':[], 'dv':[], 'dm':[], 'dh':[], 'dn':[]}

for p in all_p:
    if p.prev_state is None:
        continue
    trajectories['v'].append(p.prev_state.v)
    trajectories['m'].append(p.prev_state.m)
    trajectories['h'].append(p.prev_state.h)
    trajectories['n'].append(p.prev_state.n)
    trajectories['dv'].append(p.state.state.v)
    trajectories['dm'].append(p.state.state.m)
    trajectories['dh'].append(p.state.state.h)
    trajectories['dn'].append(p.state.state.n)
                    

sim_state_dt =np.dtype([
        ('t', float),
        ('v', float),
        ('m', float),
        ('h', float),
        ('n', float)
    ])

def to_sim_state_dtype(results):
    """convert list of simulation states into a numpy array"""
    return np.array([(x.t, x.state.v, x.state.m, x.state.h, x.state.n) for x in results],
                    dtype=sim_state_dt)


### Simulations ###

step_amplitudes = np.repeat(final_amplitude, n_cycles)
linear_amplitudes = np.linspace(0,final_amplitude, n_cycles)
empty_amplitudes = np.zeros(n_cycles)

step_sim = simulate_i_wave(step_amplitudes)
linear_sim = simulate_i_wave(linear_amplitudes)
empty_sim = simulate_i_wave(empty_amplitudes)

step_cost = cost_of_path(step_amplitudes, step_sim)
linear_cost = cost_of_path(linear_amplitudes, linear_sim)
print(f'cost of step {step_cost}')
print(f'cost of ramp {linear_cost}')

# visualize rejects and outs
nans = []
notnans = []
outs = []
rejects = []
for c in counts:
    nans.extend(c['nan'])
    notnans.extend(c['notnan'])
    outs.extend(c['out'])
    rejects.extend(c['out'])

state_dt =np.dtype([
        ('v', float),
        ('m', float),
        ('h', float),
        ('n', float)
    ])

def voxel_deduplicate(states):
    """Remove points within the same voxel in 4D space."""
    # points = np.array([s.to_vec() / voxel_dim for s in states], dtype=int)
    # _, unique_indices = np.unique(points, axis=0, return_index=True)
    # filtered_states = np.array(states)[unique_indices]
    return np.array([(x.v, x.m, x.h, x.n) for x in states],
                        dtype=state_dt)

nans = voxel_deduplicate(nans)
notnans = voxel_deduplicate(notnans)
outs = voxel_deduplicate(outs)
rejects = voxel_deduplicate(rejects)


################
### Graphing ###
################

def cube(dims, corner):
    # Define cube corners (8 vertices)
    # Starting from origin
    x = [0, dims[0], dims[0], 0, 0, dims[0], dims[0], 0]
    y = [0, 0, dims[1], dims[1], 0, 0, dims[1], dims[1]]
    z = [0, 0, 0, 0, dims[2], dims[2], dims[2], dims[2]]

    # Define the 12 edges as pairs of indices
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # bottom face
        [4,5], [5,6], [6,7], [7,4],  # top face
        [0,4], [1,5], [2,6], [3,7]   # vertical edges
    ]

    # Build line segments
    x_lines, y_lines, z_lines = [], [], []

    for i, j in edges:
        x_lines += [corner[0]+x[i], corner[0]+x[j], None]
        y_lines += [corner[1]+y[i], corner[1]+y[j], None]
        z_lines += [corner[2]+z[i], corner[2]+z[j], None]
    return (x_lines, y_lines, z_lines)

step_sim = to_sim_state_dtype(step_sim)
linear_sim = to_sim_state_dtype(linear_sim)
empty_sim = to_sim_state_dtype(empty_sim)
all_p_array = to_sim_state_dtype([p.state for p in all_p])

# Create subplot grid: 2 rows, 2 columns
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}],
           [{'type': 'xy'}, {'type': 'xy'}]],
    subplot_titles=["3D Swarm", "3D Phase Space", "Voltage vs Time", "AC Amplitudes"],
    row_heights = [0.65, 0.35],
    horizontal_spacing=0.1,  # default is 0.2
    vertical_spacing=0.07,    # default is 0.3
)

hover = [
f"""v={p['v']}<br>
m={p['m']}<br>
h={p['h']}<br>
n={p['n']}<br>
amp={'<br>'.join([f'{x}' for x in all_p[i].amplitudes])}<br>
p={all_p[i].period}<br>
interp={all_p[i].interpolated}""" 
for (i, p) in enumerate(all_p_array)
]
# pert={'<br>'.join(['[' + ', '.join(f'{num:.4f}' for num in row) + ']' for row in all_p[i].perturbation])}<br>

# Initial point
trace_initial = go.Scatter3d(
    x=[initial.state.v], y=[initial.state.m], z=[initial.state.h],
    mode='markers',
    marker=dict(size=8, color='green', symbol='circle-open'),
    name='Initial'
)
trace_final = go.Scatter3d(
    x=[final.state.v], y=[final.state.m], z=[final.state.h],
    mode='markers',
    marker=dict(size=8, color='red', symbol='circle-open'),
    name='Final'
)

# 3D swarm plot
fig.add_trace(go.Scatter3d(
    x=all_p_array['v'],
    y=all_p_array['m'],
    z=all_p_array['h'],
    mode='markers',
    marker=dict(
        size=4,
        color=all_p_array['t'],
        colorscale='Viridis',
        colorbar=dict(
          title='t',
          len=0.75,
          yanchor='bottom',
          y=0.0
        ),
    ),
    text=hover,
    hoverinfo='text',
    name='Swarm'
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=nans['v'],
    y=nans['m'],
    z=nans['h'],
    mode='markers',
    marker=dict(
        size=2,
        color='red',
    ),
    visible='legendonly',
    name='Nan'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=notnans['v'],
    y=notnans['m'],
    z=notnans['h'],
    mode='markers',
    marker=dict(
        size=2,
        color='green',
    ),
    visible='legendonly',
    name='NotNan'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=outs['v'],
    y=outs['m'],
    z=outs['h'],
    mode='markers',
    marker=dict(
        size=2,
        color='orange',
    ),
    visible='legendonly',
    name='Outs'
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=rejects['v'],
    y=rejects['m'],
    z=rejects['h'],
    mode='markers',
    marker=dict(
        size=2,
        color='yellow',
    ),
    visible='legendonly',
    name='Rejects'
), row=1, col=1)

x_lines = []
y_lines = []
z_lines = []

for i in range(len(trajectories['v'])):
    x_lines += [trajectories['v'][i], trajectories['dv'][i], None]
    y_lines += [trajectories['m'][i], trajectories['dm'][i], None]
    z_lines += [trajectories['h'][i], trajectories['dh'][i], None]

# Plot
fig.add_trace(go.Scatter3d(
    x=x_lines, y=y_lines, z=z_lines,
    mode='lines',
    line=dict(color='blue', width=1),
    visible='legendonly',
    name='Traj'
), row=1, col=1)

cube_lines = cube(voxel_dim, final.state.to_vec())
fig.add_trace(go.Scatter3d(
    x=cube_lines[0], y=cube_lines[1], z=cube_lines[2],
    mode='lines',
    line=dict(color='purple', width=1),
    name='Voxel'
), row=1, col=1)

fig.add_trace(trace_initial, row=1, col=1)
fig.add_trace(trace_final, row=1, col=1)

# 3D phase space
fig.add_trace(go.Scatter3d(
    x=step_sim['v'], y=step_sim['m'], z=step_sim['h'],
    mode='markers',
    marker=dict(size=4, color=step_sim['t'], colorscale='Viridis'),
    name='Step sim'
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=linear_sim['v'], y=linear_sim['m'], z=linear_sim['h'],
    mode='markers',
    marker=dict(size=4, color=linear_sim['t'], colorscale='Plotly3'),
    name='Ramp sim'
), row=1, col=2)

fig.add_trace(trace_initial, row=1, col=2)
fig.add_trace(trace_final, row=1, col=2)

# Voltage vs time
fig.add_trace(go.Scatter(
    x=step_sim['t'], y=step_sim['v'], mode='lines', name='Step V'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=linear_sim['t'], y=linear_sim['v'], mode='lines', name='Ramp V'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=empty_sim['t'], y=empty_sim['v'], mode='lines', name='Zero V'
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=[empty_sim['t'][0], empty_sim['t'][-1]],
    y=[final.state.v, final.state.v],
    mode='lines',
    line=dict(dash='dash'),
    name='Goal'
), row=2, col=1)

# Amplitude vs cycle
fig.add_trace(go.Scatter(
    x=np.arange(n_cycles), y=step_amplitudes, mode='lines', name='Step amps'
), row=2, col=2)
fig.add_trace(go.Scatter(
    x=np.arange(n_cycles), y=linear_amplitudes, mode='lines', name='Ramp amps'
), row=2, col=2)

# Axis titles
fig.update_layout(
    scene1=dict(
        xaxis_title='V',
        yaxis_title='m',
        zaxis_title='h',
        # xaxis = dict(range=[space_lower_bounds[0], space_upper_bounds[0]]),
        yaxis = dict(range=[space_lower_bounds[1], space_upper_bounds[1]]),
        zaxis = dict(range=[space_lower_bounds[2], space_upper_bounds[2]])
    ),
    scene2=dict(
        xaxis_title='V',
        yaxis_title='m',
        zaxis_title='h'
    ),
    xaxis1=dict(title='t (ms)'),
    yaxis1=dict(title='V (mV)'),
    xaxis2=dict(title='Cycle'),
    yaxis2=dict(title='Amplitude (nA)'),
    showlegend=True,
)

# fig.show()
fig.show(config={"responsive": True})

