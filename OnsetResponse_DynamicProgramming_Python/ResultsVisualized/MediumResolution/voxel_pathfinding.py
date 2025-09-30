import time
import numpy as np
import itertools
from neuron import h, gui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import pandas as pd
# import psutil, os
# import gc

### Main parameters ###

# 1/n ms period => n kHz frequency
period = 1/2 # ms
dt = 0.01 # 20 dt per cycle as is

load_map = False
load_path = False
do_paths = True

v_init = -64.975

n_cycles = 100
final_amplitude = 150

# discrete_amplitudes = np.array([0,0.25,0.5,0.75,1,1.5])
discrete_amplitudes = np.linspace(0,1.5,num=16)

max_iter = n_cycles

# voxels in (v,m,h,n) space
# voxel_dim = np.array([1, 0.05, 0.1, 0.1])
# space_lower_bounds = np.array([-80, 0, 0, 0])
# space_upper_bounds = np.array([ 0, 1, 1, 1])
# 
voxel_dim = np.array([0.05, 0.01, 0.01, 0.01])
space_lower_bounds = np.array([-80, 0, 0.55, 0.3])
space_upper_bounds = np.array([-55, 0.1, 0.6, 0.35])

n_voxels = np.array(np.round((space_upper_bounds - space_lower_bounds)/voxel_dim), dtype=int)
voxel_ranges = [range(n) for n in n_voxels]

print(f'voxels {n_voxels}, max transitions: {np.prod(n_voxels)*len(discrete_amplitudes)}')

### Other definitions ###

transitions_file = f'transitions-{"-".join([str(x) for x in n_voxels])}'
path_grid_file = f'path-grid-for-{transitions_file}.npy'

steps_per_period = int(period / dt)
print(f'{steps_per_period} steps per period')

axon = h.Section(name='axon')
axon.insert('hh')
axon_seg = axon(0.5)

h.secondorder=1

stim = h.IClamp(axon_seg)
stim.delay = 0
stim.dur = 1e9   # always on

print_particles = False
def particle_msg(msg):
    if print_particles:
        print(msg)

# Space of a simulation: voltage, and ion channels
state_dt = np.dtype([
    ('v', float),
    ('m', float),
    ('h', float),
    ('n', float)
])

# Voxel indices of a state
state_index_dt = np.dtype([
    ('vi', np.int32),
    ('mi', np.int32),
    ('hi', np.int32),
    ('ni', np.int32)
])

# Edge of a directed graph whose vertices are the state indices
transition_dt = np.dtype([
    ('start', state_index_dt),
    ('end', state_index_dt),
    ('power', np.float32),             # amplitude index
])

# A path at a point stores where it came from and with what amplitude
path_dt = np.dtype([
    ('from', state_index_dt),
    ('ai', int),
    ('cost', float)
])

# A simulation has a state at a particular time
sim_dt = np.dtype([
    ('t', float),
    ('state', state_dt)
])

empty_state = np.zeros(1, dtype=state_dt)[0]
empty_index = np.void((-1,-1,-1,-1), dtype=state_index_dt)
empty_path = np.void((empty_index, 0, 10000), dtype=path_dt)

def state_string(state, truncate=True):
    """ This returns the full decimal places """
    if truncate:
        return f'({state["v"]:.3f}, {state["m"]:.3f}, {state["h"]:.3f}, {state["n"]:.3f})'
    else:
        return f'({state["v"]}, {state["m"]}, {state["h"]}, {state["n"]})'

def sim_string(sim, truncate=True):
    return f"{state_string(sim['state'],truncate=truncate)} t={sim['t']:.3f}"

def all_state_indices():
    return itertools.product(*voxel_ranges)

weight_v = 1
weight_current = 0.1
def cost_function(voltage, current):
    return weight_v*max(voltage-5,0)**2 + weight_current*(current/final_amplitude)**2

def cost_of_path(cycle_amplitudes, results):
    cost = 0
    for i in range(len(cycle_amplitudes)):
        cost += cost_function(results['state']['v'][1+i], cycle_amplitudes[i])
    return cost

### Simulation functions ###

def load_sim(state):
    (t, (v, m_var, h_var, n_var)) = state
    
    # Call finitialize() to make sure hh.gna and the like are not nan
    h.finitialize(v)
    h.t = t
    axon_seg.hh.m = m_var
    axon_seg.hh.h = h_var
    axon_seg.hh.n = n_var

def get_sim():
    return np.void((h.t, (axon_seg.v, axon_seg.hh.m, axon_seg.hh.h, axon_seg.hh.n)), dtype=sim_dt)

def simulate_forward_one_cycle(state, amplitude, avg_power=False):
    h.dt = dt
    load_sim(state)
    power = 0

    for step in range(0, steps_per_period):
        stim.amp = amplitude * np.sin(2 * np.pi * step / steps_per_period)
        power += -stim.amp * axon_seg.v
        h.fadvance()

    if avg_power:
        return get_sim(), power / dt
    return get_sim()

# def simulate_backward_one_cycle(state, amplitude):
#     """
#     Could return None if the simulation diverges
#     """
#     h.dt = -dt
#     load_state(state)

#     if math.isnan(axon_seg.hh.gna):
#         raise "Already NaN"
 
#     for step in range(steps_per_period, 0, -1):
#         stim.amp = amplitude * np.sin(2 * np.pi * step / steps_per_period)
#         h.fadvance()
    
#     if math.isnan(axon_seg.v):
#         return None

#     return get_state()

def simulate_stim_waveform(waveform, make_rounds=False):
   
    # there's a point before the first cycle and after each
    n_points = 1+len(waveform) 

    results = np.empty(n_points, dtype=sim_dt)

    h.finitialize(v_init)
    h.t = 0
    results[0] = get_sim()

    if make_rounds:
        results[0] = round_sim(results[0])

    for (i, a) in enumerate(waveform):
        results[1+i] = simulate_forward_one_cycle(results[i], a)

        if make_rounds:
            results[1+i] = round_sim(results[1+i])

    return results


def get_steady_state_for_amplitude(amplitude):

    max_cycles = 500

    h.finitialize(v_init)
    h.t = 0

    # there's a point before the first cycle and after each
    n_points = 1+max_cycles 
    results = np.zeros(n_points, dtype=sim_dt)
    results[0] = get_sim()

    steps_within = 20
    threshhold = 0.00001

    for i in range(max_cycles-1):
        results[1+i] = simulate_forward_one_cycle(results[i], amplitude)

        if i < steps_within:
            continue

        dstate = np.abs(np.subtract(np.array(results['state'][(i-steps_within):(1+i)].tolist()), np.array(results['state'][1+i].tolist())))

        if np.sum(dstate > threshhold) == 0:
            print(f'converged after {i} cycles, t={results["t"][i]}')
            return results, results[0], results[1+i]
    raise "Did not converge"

def out_of_bounds(state):
    for i, (value, lo, hi) in enumerate(zip(state, space_lower_bounds, space_upper_bounds)):
        if not (lo <= value <= hi):
            return f"{['v','m','h','n'][i]} = {value} is not in [{lo}, {hi}]"

def state_to_index(state):
    """ Assume state is in bounds """
    return tuple(np.minimum(np.array((np.array(state.tolist()) - space_lower_bounds) / voxel_dim, dtype=int), n_voxels - 1))

def index_to_state(voxel):
    return np.void(tuple(np.array(tuple(voxel))*voxel_dim + space_lower_bounds + 0.5*voxel_dim), dtype=state_dt)

def round_state(state):
    return index_to_state(state_to_index(state))

def round_sim(sim_state):
    (t, state) = sim_state
    return np.void((t, round_state(state)), dtype=sim_dt)

def amplitude_from_index(ai):
    return final_amplitude * discrete_amplitudes[ai]

### Grid iterations ###

def make_transitions():
    max_transitions = np.product(n_voxels) * discrete_amplitudes.size
    transitions = np.empty(max_transitions, dtype=transition_dt)
    transitions_len = 0

    i = 0

    # Iterate through the array and print the value and its corresponding tuple index
    for state_index in all_state_indices():
        for ai in range(len(discrete_amplitudes)):
            start_state = index_to_state(state_index)
            amp = amplitude_from_index(ai)
        
            end_state, avg_power = simulate_forward_one_cycle(np.void((final['t'], tuple(start_state)), dtype=sim_dt), amp, avg_power=True)
            # _ = simulate_forward_one_cycle(np.void((final['t'], tuple(start_state)), dtype=sim_dt), amp)

            if i % 100230 == 0:
                print(start_state)
                print(f'{i}/{max_transitions}')
                np.save(transitions_file, transitions)
                # gc.collect()
                # print(f"Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

            if end_state is not None and out_of_bounds(end_state['state']) is None:
                end_state_index = state_to_index(end_state['state'])
                if state_index != end_state_index:
                    transitions[transitions_len] = (state_index, end_state_index, avg_power)
                    transitions_len += 1
    
            i += 1

    return transitions[0:transitions_len]

def make_path_grid(final_state, transitions):
    final_index = state_to_index(final_state)

    path_grid = np.full(n_voxels, empty_path, dtype=path_dt)
    path_grid[final_index] = (final_index,0,0)
    
    quick_lookup = dict()
    for t in transitions:
        (_, end, _) = t
        end = tuple(end)
        if end not in quick_lookup:
            quick_lookup[end] = []
        quick_lookup[end].append(t)

    print(transitions.size)
    print(path_grid[final_index])

    for i in range(max_iter):
        print(f"===== step {i} =====")
        changes = dict()
        for index in all_state_indices():
            (from_state_index, _, cost) = path_grid[index]

            if from_state_index == empty_index:
                continue

            # loop through transitions that end here
            if index not in quick_lookup:
                continue
            for (start_state_index,_,added_cost) in quick_lookup[index]:
                new_cost = cost + added_cost
                if new_cost < path_grid[tuple(start_state_index)]['cost']:
                    changes_value = changes.get(tuple(start_state_index))
                    if changes_value is None or new_cost < changes_value[2]:
                        changes[tuple(start_state_index)] = (index, added_cost, new_cost)
                    
        if len(changes) == 0:
            print(f'no more changes after {i} steps')
            break
        print(f'{len(changes)} changes')

        for ind, val in changes.items():
            path_grid[ind] = val

    return path_grid

#########

initialize_result, initial, final = get_steady_state_for_amplitude(final_amplitude)
print(f"initial state {state_string(initial['state'])}")
print(f"goal state {state_string(final['state'])}")
print(f"initial state {state_string(round_state(initial['state']))}")
print(f"goal state {state_string(round_state(final['state']))}")

if load_map:
    transitions = np.load(transitions_file)
    transitions = np.array(transitions, dtype=transition_dt)
else:
    start_time = time.perf_counter()
    transitions = make_transitions()
    end_time = time.perf_counter()
    print(f'transitions took {end_time - start_time}')
    np.save(transitions_file, transitions)


x_paths = []
y_paths = []
z_paths = []

x_transition_lines = []
y_transition_lines = []
z_transition_lines = []

if do_paths:
    if load_path:
        path_grid = np.load(path_grid_file)
    else:
        path_grid = make_path_grid(final['state'], transitions)
        np.save(path_grid_file, path_grid)

    start_index = state_to_index(initial['state'])
    final_index = np.void(state_to_index(final['state']), dtype=state_index_dt)

    path_start = path_grid[start_index]
    optimal_path = []
    optimal_path_amplitudes = []
    optimal_amplitudes = np.zeros(n_cycles, dtype=float)

    if path_start != empty_path:
        current_index = np.void(start_index, dtype=state_index_dt)
        (_, _, cost) = path_grid[tuple(current_index)]
        optimal_path.append(index_to_state(current_index))
        print(f'Success! cost {cost}')

        while current_index != final_index:
            (current_index, ai,  _) = path_grid[tuple(current_index)]

            optimal_path.append(index_to_state(current_index))
            optimal_path_amplitudes.append(amplitude_from_index(ai))

        print(optimal_path_amplitudes)
        optimal_amplitudes[n_cycles-len(optimal_path_amplitudes):] = optimal_path_amplitudes

    optimal_path = np.array(optimal_path, dtype=state_dt)
    print(optimal_path)


    # for index in all_state_indices():
    #     (from_state_index, _, _) = path_grid[index]
    #     if from_state_index == empty_index:
    #         continue

    #     # print(np.array(point.tolist()))
    #     start_state = index_to_state(index)
    #     end_state = index_to_state(from_state_index)
    #     x_paths += [start_state['v'], end_state['v'], None]
    #     y_paths += [start_state['m'], end_state['m'], None]
    #     z_paths += [start_state['h'], end_state['h'], None]

    # for (start,end,amp) in transitions:
    #     start_state = index_to_state(start)
    #     end_state = index_to_state(end)
    #     x_transition_lines += [start_state['v'], end_state['v'], None]
    #     y_transition_lines += [start_state['m'], end_state['m'], None]
    #     z_transition_lines += [start_state['h'], end_state['h'], None]

### Simulations ###

step_amplitudes = np.repeat(final_amplitude, n_cycles)
linear_amplitudes = np.linspace(0,final_amplitude, n_cycles)
empty_amplitudes = np.zeros(n_cycles)

step_sim = simulate_stim_waveform(step_amplitudes)
linear_sim = simulate_stim_waveform(linear_amplitudes)
empty_sim = simulate_stim_waveform(empty_amplitudes)
optimal_sim = simulate_stim_waveform(optimal_amplitudes)
optimal_rounded_sim = simulate_stim_waveform(optimal_amplitudes, make_rounds=True)

step_cost = cost_of_path(step_amplitudes, step_sim)
linear_cost = cost_of_path(linear_amplitudes, linear_sim)
optimal_cost = cost_of_path(optimal_amplitudes, optimal_sim)
print(f'cost of step {step_cost}')
print(f'cost of ramp {linear_cost}')
print(f'cost of optimal {optimal_cost}')

################
### Graphing ###
################


def cube(dims, center):
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

    corner = [c - d/2 for d, c in zip(dims, center)]

    for i, j in edges:
        x_lines += [corner[0]+x[i], corner[0]+x[j], None]
        y_lines += [corner[1]+y[i], corner[1]+y[j], None]
        z_lines += [corner[2]+z[i], corner[2]+z[j], None]
    return (x_lines, y_lines, z_lines)

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


# Initial point
trace_initial = go.Scatter3d(
    x=[initial['state']['v']], y=[initial['state']['m']], z=[initial['state']['h']],
    hovertext=state_string(initial['state']),
    mode='markers',
    marker=dict(size=8, color='green', symbol='circle-open'),
    name='Initial',
)
trace_final = go.Scatter3d(
    x=[final['state']['v']], y=[final['state']['m']], z=[final['state']['h']],
    mode='markers',
    hovertext=state_string(final['state']),
    marker=dict(size=8, color='red', symbol='circle-open'),
    name='Final',
)

# Plot
# fig.add_trace(go.Scatter3d(
#     x=grid_map['end']['vi']*voxel_dim[0] + space_lower_bounds[0] + voxel_dim[0]/2,
#     y=grid_map['end']['mi']*voxel_dim[1] + space_lower_bounds[1] + voxel_dim[1]/2,
#     z=grid_map['end']['hi']*voxel_dim[2] + space_lower_bounds[2] + voxel_dim[2]/2,
#     mode='markers',
#     marker=dict(
#         size=2,
#         color=grid_map['cost'],
#         colorscale='Viridis',
#         colorbar=dict(
#           title='t',
#           len=0.75,
#           yanchor='bottom',
#           y=0.0
#         ),
#     ),
#     name='Swarm',
# ), row=1, col=1)


fig.add_trace(go.Scatter3d(
    x=optimal_path['v'],
    y=optimal_path['m'],
    z=optimal_path['h'],
    mode='lines',
    line=dict(
        color='darkblue',
        width=2
    ),
    hovertext=[state_string(s) for s in optimal_path],
    name='FullPath',
), row=1, col=1)

fig.add_trace(go.Scatter3d(
    x=x_transition_lines, y=y_transition_lines, z=z_transition_lines,
    mode='lines',
    line=dict(
        color=path_grid['cost'],
        colorscale='Viridis',
        width=1
    ),
    visible='legendonly',
    text=[f'cost: {a}' for a in path_grid['cost']],
    hoverinfo='text',
    name='Traj'
), row=1, col=1)


fig.add_trace(go.Scatter3d(
    x=x_paths, y=y_paths, z=z_paths,
    mode='lines',
    line=dict(
        color='red',
        width=2
    ),
    name='Paths'
), row=1, col=1)


cube_lines = cube(voxel_dim, round_state(final['state']))
fig.add_trace(go.Scatter3d(
    x=cube_lines[0], y=cube_lines[1], z=cube_lines[2],
    mode='lines',
    line=dict(color='purple', width=1),
    name='Voxel'
), row=1, col=1)

fig.add_trace(trace_initial, row=1, col=2)
fig.add_trace(trace_final, row=1, col=2)

# 3D phase space
fig.add_trace(go.Scatter3d(
    x=step_sim['state']['v'], y=step_sim['state']['m'], z=step_sim['state']['h'],
    mode='markers',
    marker=dict(size=4, color=step_sim['t'], colorscale='Viridis'),
    name='Step sim',
    hovertext=[sim_string(s) for s in step_sim],
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=linear_sim['state']['v'], y=linear_sim['state']['m'], z=linear_sim['state']['h'],
    mode='markers',
    marker=dict(size=4, color=linear_sim['t'], colorscale='Blues'),
    name='Ramp sim',
    hovertext=[sim_string(s) for s in linear_sim],
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=optimal_sim['state']['v'], y=optimal_sim['state']['m'], z=optimal_sim['state']['h'],
    mode='markers',
    marker=dict(size=4, color=optimal_sim['t'], colorscale='Peach'),
    name='Optimal sim',
    hovertext=[sim_string(s) for s in optimal_sim],
), row=1, col=2)

fig.add_trace(go.Scatter3d(
    x=optimal_rounded_sim['state']['v'], y=optimal_rounded_sim['state']['m'], z=optimal_rounded_sim['state']['h'],
    mode='markers',
    marker=dict(size=4, color=optimal_rounded_sim['t'], colorscale='Teal'),
    name='Optimal sim',
    hovertext=[sim_string(s) for s in optimal_rounded_sim],
), row=1, col=2)

fig.add_trace(trace_initial, row=1, col=1)
fig.add_trace(trace_final, row=1, col=1)

# Voltage vs time
fig.add_trace(go.Scatter(
    x=step_sim['t'], y=step_sim['state']['v'], mode='lines', name='Step V',
    hovertext=[sim_string(s) for s in step_sim]
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=linear_sim['t'], y=linear_sim['state']['v'], mode='lines', name='Ramp V',
    hovertext=[sim_string(s) for s in linear_sim]
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=empty_sim['t'], y=empty_sim['state']['v'], mode='lines', name='Zero V',
    hovertext=[sim_string(s) for s in empty_sim]
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=optimal_sim['t'], y=optimal_sim['state']['v'], mode='lines', name='Optimal V',
    hovertext=[sim_string(s) for s in optimal_sim]
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=optimal_rounded_sim['t'], y=optimal_rounded_sim['state']['v'], mode='lines', name='Optimal Rounded V',
    hovertext=[sim_string(s) for s in optimal_rounded_sim]
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=[empty_sim['t'][0], empty_sim['t'][-1]],
    y=[final['state']['v'], final['state']['v']],
    mode='lines',
    line=dict(dash='dash'),
    name='Goal'
), row=2, col=1)

# Amplitude vs cycle
fig.add_trace(go.Scatter(
    x=np.arange(n_cycles), y=step_amplitudes, mode='lines', name='Step amps'
), row=2, col=2)
fig.add_trace(go.Scatter(
    x=np.arange(n_cycles), y=optimal_amplitudes, mode='lines', name='Step amps'
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
        xaxis = dict(range=[space_lower_bounds[0], space_upper_bounds[0]]),
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

