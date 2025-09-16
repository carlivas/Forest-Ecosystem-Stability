import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mods.files import *
from mods.utilities import get_unique_aliases, format_float
from mods.buffers import StateBuffer, DataBuffer
from mods.simulation import Simulation
from mods.fields import DensityFieldCustom
from mods.plant import Plant, PlantCollection


def spatial_correlation_function(field, L=1):
    """Calculate the spatial correlation function for a 2D field."""
    mean, var = np.mean(field), np.var(field)
    M, N = field.shape
    distances = np.arange(1, min(M, N) // 2 + 1)
    corr = np.array([
        (M * N / ((M - d) * (N - d))) * np.sum((field[:-d, :-d] - mean) * (field[d:, d:] - mean)) / var
        for d in distances
    ])
    distances = distances * L / (min(M,N) // 2)  # Scale distances to the box size
    return distances, corr


def spatial_powr_spectrum(field):
    """Calculate the spatial power spectrum for a 2D field."""
    M, N = field.shape
    DFT = np.fft.fft2(field) / (M * N)
    power_spectrum = np.fft.fftshift(np.abs(DFT)**2)
    return power_spectrum


def calculate_r_spectrum(power_spectrum, num_bins=50):
    """Calculate the radial spectrum from the power spectrum using bins."""
    M, N = power_spectrum.shape
    X, Y = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    distances = np.linalg.norm(np.vstack([X.flatten(), Y.flatten()]).T, axis=1)
    power_spectrum = power_spectrum.flatten()[distances != 0]  # Exclude zero
    distances = distances[distances != 0]  # Exclude zero
    power_spectrum = power_spectrum.flatten()[distances <= max(M//2,N//2)]  # Exclude zero
    distances = distances[distances <= max(M//2,N//2)]  # Exclude zero
    bins = np.linspace(distances.min(), distances.max(), num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    r_spectrum = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers) - 1):
        mask = (distances >= bin_centers[i]) & (distances < bin_centers[i + 1])
        r_spectrum[i] = np.sum(power_spectrum.flatten()[mask])
    
    return bin_centers, r_spectrum


def calculate_theta_spectrum(power_spectrum, num_bins=50):
    """Calculate the angular spectrum from the power spectrum using bins."""
    M, N = power_spectrum.shape
    X, Y = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    radians = np.arctan2(Y.flatten(), X.flatten())
    power_spectrum = power_spectrum.flatten()[radians != 0]  # Exclude zero
    radians = radians[radians != 0]  # Exclude zero
    bins = np.linspace(-np.pi, np.pi, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    theta_spectrum = np.zeros(len(bin_centers))
    
    for i in range(len(bin_centers) - 1):
        mask = (radians >= bin_centers[i]) & (radians < bin_centers[i + 1])
        theta_spectrum[i] = np.sum(power_spectrum.flatten()[mask])
    
    return bin_centers, theta_spectrum

def find_mean_biomass(sims, verbose=False):
    # Initialize DataFrames
    biomasses = pd.DataFrame()
    mean_biomass = pd.DataFrame()
    max_index = 0
    
    # Loop through each simulation and put biomass data into the biomasses DataFrame
    for i, sim in enumerate(sims):
        density_scheme = sim.density_scheme
        if verbose:
            print(f'find_mean_biomass(): Loading simulation biomass {i+1}/{len(sims)}......', end='\r')
        
        data_buffer_path = f'{sim.folder}/data_buffer-{sim.alias}.csv'
        data = DataBuffer.get_data_static(file_path=data_buffer_path)
        
        if data.empty:
            if verbose:
                print(f'find_mean_biomass(): No data for alias: {sim.alias}')
            continue
        else:    
            # biomasses[i, :len(data['Biomass'].values)] = data['Biomass'].values
            d = {
                f'biomass_{density_scheme}_{sim.alias}': data['Biomass'].values,
            }
            biomasses = pd.concat([biomasses, pd.DataFrame.from_dict(d)], axis=1)
            max_index = max(max_index, len(data['Time']))
                
        # Truncate trailing zeros from biomasses
        biomasses = biomasses.iloc[:, :max_index]
        if biomasses.shape[0] == 0 or biomasses.isnull().all(axis=1).any():
            if verbose:
                print(f'find_mean_biomass(): Biomass data is empty or contains all NaN values in at least one row.')
            continue
        
        means = {
                f'mean_biomass_{density_scheme}': biomasses[biomasses.columns[biomasses.columns.str.contains(f'biomass_{density_scheme}')]].mean(axis=1),
            }
        mean_biomass = pd.concat([mean_biomass, pd.DataFrame.from_dict(means)], axis=1)
    
    if verbose:
        print(f'find_mean_biomass(): Biomass data shape: {biomasses.shape}')
        print(f'find_mean_biomass(): Mean biomass data shape: {mean_biomass.shape}')
    return mean_biomass

def estimate_tipping_point(mean_biomass, window_size = 1500, step_size = 1):
    time = mean_biomass.index.values
    biomass = mean_biomass.values.flatten()
    dt = time[1] - time[0]
    
    biomass_diff = []
    window_starts = np.arange(500, time[-1], step_size).astype(int)
    window_centers = window_starts + window_size//2
    for start in window_starts:
        if start + window_size > time[-1]:
            break
        
        dB = abs(biomass[start+window_size] - biomass[start])
        dB /= window_size * dt
        biomass_diff.append(dB)
    biomass_diff = np.array(biomass_diff)
    
    where_max_diff = np.argmax(biomass_diff)
    tp_time = time[window_centers[where_max_diff]]
    tp_biomass = biomass[window_centers[where_max_diff]]
    
    return tp_time, tp_biomass

def find_targets(mean_biomass, ntimes, verbose=False):
    transient_time = 1000
    tp_time, tp_biomass = estimate_tipping_point(mean_biomass)
    target_times = np.linspace(transient_time, tp_time, ntimes).astype(int)
    target_biomass = mean_biomass.loc[target_times].values.flatten()
    
    print(f'find_targets(): {target_times=}')
    print(f'find_targets(): {target_biomass=}')
    return target_times, target_biomass
    

# def find_targets(mean_biomass, ntimes, verbose=False):
#     target_times = np.array([])
#     if mean_biomass.empty or mean_biomass.isnull().all(axis=1).any():
#         print('find_targets(): mean_biomass is empty or contains all NaN values in at least one row.')
#     else:
#         # Calculate mean biomass across simulations and find target biomass values
#         bmin = 0.15
#         bmax = 0.30
#         target_biomass = np.linspace(bmin, bmax, ntimes)
#         if verbose:
#             print(f'find_targets(): {target_biomass=}')
#         ii = np.array([np.abs(mean_biomass - tb).idxmin().iloc[-1] for tb in target_biomass])
#         target_times = np.concat([target_times, ii])
        
#         sort = np.argsort(target_times)
#         target_times = target_times[sort]
#         target_biomass = target_biomass[sort]
#     if verbose:
#         print(f'find_targets(): {target_times=}')
        
#     return target_times, target_biomass


def calculate_individual_statistics(sim, target_time, resolution, verbose=False):
    density_scheme = sim.density_scheme
    
    # In the state buffer, find closest time to the target time
    sb_times = sim.data_buffer.get_data()['Time'].values
    t = sb_times[np.argmin(np.abs(sb_times-target_time))] if target_time not in sb_times else target_time

    # Load the state buffer data
    file_path = f'{sim.folder}/state_buffer-{sim.alias}.csv'
    sb_data = StateBuffer.get_specific_data_static(file_path, t)
    if sb_data.empty:
        print(f'calculate_individual_statistics(): sb_data is empty for alias: {sim.alias} at time {t}')
        return None, None
    
    
    images_data = pd.DataFrame()
    stats_data = pd.DataFrame()
    
    # if isinstance(sb_data, pd.DataFrame):
    #     sb_data = sb_data.reset_index(drop=True)
    # else:
    #     sb_data = pd.concat(sb_data, ignore_index=True)
    
    plants = PlantCollection()
    for _, row in sb_data.iterrows():
        plant = Plant(id=row['id'], x=row['x'], y=row['y'], r=row['r'], **sim.species_list[0].__dict__)
        plants.add_plant(plant)
    
    density_field = DensityFieldCustom(box=sim.box, resolution=resolution)
    density_field.update(plants)
    field = density_field.values.reshape((density_field.resolution, density_field.resolution))

    distances_corr, spat_corr = spatial_correlation_function(field, sim.L)
    power_spectrum = spatial_powr_spectrum(field)
    unique_distances, r_spectrum = calculate_r_spectrum(power_spectrum, num_bins=density_field.resolution//2)
    unique_radians, θ_spectrum = calculate_theta_spectrum(power_spectrum, num_bins=density_field.resolution//2)
    
    images_data[f'field_{density_scheme}_{format_float(t)}_{sim.alias}'] = field.flatten()
    images_data[f'power_spectrum_{density_scheme}_{format_float(t)}_{sim.alias}'] = power_spectrum.flatten()
    
    S = {
        f'spat_corr_{density_scheme}_{format_float(t)}_{sim.alias}': spat_corr,
        f'r_spectrum_{density_scheme}_{format_float(t)}_{sim.alias}': r_spectrum,
        f'θ_spectrum_{density_scheme}_{format_float(t)}_{sim.alias}': θ_spectrum
    }
    stats_data = pd.concat([stats_data, pd.DataFrame.from_dict(S)], axis=1)
    images_data = pd.concat([images_data, pd.DataFrame.from_dict(images_data)], axis=1)
    return stats_data, images_data

def get_containers(sims, resolution, ntimes, verbose=False):

    if len(sims) == 0:
        print('calculate_statistics(): No simulations provided for statistics calculation.')
        return None
    
    sims_container = {
        'local': [sim for sim in sims if sim.density_scheme == 'local'],
        'global': [sim for sim in sims if sim.density_scheme == 'global']
    }
    mean_biomass_container = {
        'local': find_mean_biomass(sims_container['local'], verbose),
        'global': find_mean_biomass(sims_container['global'], verbose)
    }
    
    targets_container = {
        'local': find_targets(mean_biomass_container['local'].mean(axis = 1), ntimes, verbose),
        'global': find_targets(mean_biomass_container['global'].mean(axis = 1), ntimes, verbose)
    }
    
    if verbose:
        print(f'get_containers(): {targets_container=}')
        print(f'get_containers(): {mean_biomass_container=}')
    return targets_container, sims_container, mean_biomass_container

def calculate_results(sims, resolution, ntimes, save_figs=False, figs_folder='./', verbose=False):
    ### GET CONTAINERS OF SIMULATION DATA ###
    targets_container, sims_container, mean_biomass_container = get_containers(sims, resolution, ntimes, verbose=False)
    
    ### PLOT TARGETS ###
    plot_targets(targets_container, mean_biomass_container, save=save_figs, folder=figs_folder, verbose=verbose)
    plt.show()
    
    ### CALCULATE STATISTICS ###
    stats = pd.DataFrame()
    images_data = pd.DataFrame()
    stats_data = pd.DataFrame()
    for d, density_scheme in enumerate(['local', 'global']):
        target_time, target_biomass = targets_container[density_scheme]
        sims = sims_container[density_scheme]
        for j, t in enumerate(target_time):
            for i, sim in enumerate(sims):
                if verbose:
                    print(f'calculate_results(): Processing {d+1}/2 at time step {j+1}/{len(target_time)} for simulation {i+1}/{len(sims)} with density_scheme "{density_scheme}"......', end='\r')
                stats, images = calculate_individual_statistics(sim, t, resolution, verbose)
                stats_data = pd.concat([stats_data, stats], axis=1)
                # images_data = pd.concat([images_data, images], axis=1)
    
    print(f'calculate_results(): stats_data shape: {stats_data.shape}')
    
    ### CALCULATE AGGREGATE STATISTICS ???
    for variable_name in ['spat_corr', 'r_spectrum', 'θ_spectrum']: 
        for density_scheme in ['local', 'global']:
            target_time, target_biomass = targets_container[density_scheme]
            for j, t in enumerate(target_time):
                
                cols = [c for c in stats_data.columns if  f'{variable_name}_{density_scheme}_{format_float(t)}' in c]
                stats_data_variable = stats_data[cols].dropna(axis=1).values
                
                if stats_data_variable.size == 0:
                    print(f'calculate_results(): No stats for {variable_name} at time {t}')
                    continue
                
                mean_variable = np.mean(stats_data_variable, axis=1)
                std_variable = np.std(stats_data_variable, axis=1)
                
                stats[f'{variable_name}_{density_scheme}_{format_float(t)}_mean'] = mean_variable
                stats[f'{variable_name}_{density_scheme}_{format_float(t)}_std'] = std_variable 
    ### APPEND EXTRA STATISTICS AND GATHER OUTPUTS ###
    extra_stats = {
        f'distances_corr': np.arange(resolution//2)*sim.L/(resolution//2),
        f'unique_distances': np.arange(resolution//2)*sim.L/(resolution//2),
        f'unique_radians': np.linspace(-np.pi, np.pi, resolution//2),
    }
    stats = pd.concat([stats, pd.DataFrame.from_dict(extra_stats)], axis=1)
    
    target_times = pd.DataFrame({
        'local': targets_container['local'][0],
        'global': targets_container['global'][0]
    })
    target_biomass = pd.DataFrame({
        'local': targets_container['local'][1],
        'global': targets_container['global'][1]
    })
    return target_times, target_biomass, stats

def plot_results(results, save_figs, folder, title=''):
    """Plot and save results."""
    
    target_times, target_biomass, stats = results
    var_names = ['spat_corr', 'r_spectrum', 'θ_spectrum']
    fig, ax = plt.subplots(len(var_names), 2, figsize=(6, 8))
    if len(target_times['local']) == 1:
        ax = np.array([ax])
    for v, variable_name in enumerate(var_names):
        for k, density_scheme in enumerate(['local', 'global']):
            tt = target_times[density_scheme]
            for t, time in enumerate(tt):
                if t > 0:
                    ax[v, k].sharex(ax[v, 0])
                    ax[v, k].sharey(ax[v, 0])
                label = f't={int(time)}'
                if variable_name == 'spat_corr':
                    xscale = 'log'
                    yscale = 'linear'
                    xlabel = 'distance (m)'
                    ylabel = 'spatial correlation'
                    xx = stats['distances_corr']
                elif variable_name == 'r_spectrum':
                    xscale = 'log'
                    yscale = 'linear'
                    xlabel = 'distance (m)'
                    ylabel = '$r$-spectrum (power)'
                    xx = stats['unique_distances']
                elif variable_name == 'θ_spectrum':
                    xscale = 'linear'
                    yscale = 'linear'
                    xlabel = 'radians (π)'
                    ylabel = '$θ$-spectrum (power)'
                    xx = stats['unique_radians']                
                cmap = plt.get_cmap('magma') if density_scheme == 'local' else plt.get_cmap('viridis')
                color = cmap((t + 1)/(len(tt) + 1))
            
                if len(stats) == 0 and np.isnan(stats).all():
                    print(f'plot_results(): No stats for {variable_name}_{density_scheme}_{int(time)}')
                    continue
                mean = stats[f'{variable_name}_{density_scheme}_{format_float(time)}_mean']
                std = stats[f'{variable_name}_{density_scheme}_{format_float(time)}_std']
                
                ax[v, k].plot(xx, mean, color=color, label=label)
                ax[v, k].fill_between(xx, mean - std, mean + std, alpha=0.2, color=color)
                ax[v, k].set_xlabel(xlabel)
                ax[v, k].set_ylabel(ylabel)
                ax[v, k].set_xscale(xscale)
                ax[v, k].set_yscale(yscale)
                columns = stats.columns[stats.columns.str.contains(f'{variable_name}_{density_scheme}_{format_float(time)}')]
                
                ax[v, k].grid(alpha=0.5)
                ax[v, k].tick_params(axis='both', which='major')
                # ax[v, t].locator_params(axis='x', nbins=4)
                ax[v, k].locator_params(axis='y', nbins=4)
                if k > 0:
                    ax[v, k].tick_params(labelleft=False)
                    ax[v, k].set_ylabel('')
                if v == 0:
                    ax[v, k].set_title(f'{density_scheme} model')
                    # ax[v, k].set_xlabel('')
                    # ax[v, k].tick_params(labelbottom=False)
                elif v == 1:
                    ax[v, k].legend(handlelength=0.5, handletextpad=0.5, loc='best', markerscale=1, fancybox=True, shadow=False, frameon=True)
                    for legend_handle in ax[v, k].get_legend().legend_handles:
                        legend_handle.set_marker('s')  # Set marker to square
                        legend_handle.set_linestyle('')  # Remove line
                    
                    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    if save_figs:
        fig.savefig(f'{folder}/EWS_spatial_agg.png', dpi=300, bbox_inches='tight')

def load_simulations(input_folder, verbose=False):
    aliases = get_unique_aliases(input_folder)
    print(f'load_simulations: {aliases = }')
    
    sims=[]
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, alias in enumerate(aliases):
        if verbose:
            print(f'load_simulations(): Processing alias: {alias}, {i+1}/{len(aliases)}......', end='\r')
            
        kwargs = pd.read_json(f'{input_folder}/kwargs-{alias}.json', typ='series').to_dict()
        sim = Simulation(folder=input_folder, alias=alias, verbose=False, **kwargs)
        
        db_data = sim.data_buffer.get_data()
        if db_data.empty:
            print(f'db_data is empty for alias: {alias}')
            continue

        b15000 = db_data['Biomass'][db_data['Time'] == 15000].values[0]    
        if b15000 < 0.1:
            color = 'red' if sim.density_scheme == 'local' else 'blue'
            sims.append(sim)
        else:    
            color = 'k'
        
        ax.plot(sim.data_buffer.get_data()['Time'].values, sim.data_buffer.get_data()['Biomass'].values, color=color)
    return sims


def plot_targets(targets_container, mean_biomass_container, save=False, folder='./', verbose=False):
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    ax.set_xlim(-150, 15000)
    xticks = np.array([])
    yticks = np.array([])
    for density_scheme in ['local', 'global']:
        target_times, target_biomass = targets_container[density_scheme]
        print(f'plot_targets(): {density_scheme} target times: {target_times}')
        print(f'plot_targets(): {density_scheme} target biomass: {target_biomass}')
    
        color = 'blue' if density_scheme == 'local' else 'orange'
        
        mean_mean_biomass = mean_biomass_container[density_scheme].mean(axis=1)
        mean_std_biomass = mean_biomass_container[density_scheme].std(axis=1)
        ax.plot(mean_mean_biomass, color=color, label=f'{density_scheme} model')
        ax.fill_between(mean_mean_biomass.index, mean_mean_biomass - mean_std_biomass, mean_mean_biomass + mean_std_biomass, alpha=0.2, color=color)
        
        # Get axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        print(f'plot_targets(): {x_min=}, {x_max=}, {y_min=}, {y_max=}')
        print(f'plot_targets(): {target_times=}, {target_biomass=}')

        for t, b in zip(target_times, target_biomass):
            print(f'plot_targets(): {density_scheme} target time: {t}, target biomass: {b}')

            # Normalize b and t
            normalized_ymax = (b - y_min) / (y_max - y_min)
            normalized_xmax = (t - x_min) / (x_max - x_min)
            # Plot with normalized values
            ax.axvline(x=t, ymax=normalized_ymax, color=color, ls='-', lw=2, zorder = -1, alpha=1)
            ax.axhline(y=b, xmax=normalized_xmax, color=color, ls='-', lw=2, zorder = -1, alpha=1)
        ax.scatter(target_times, target_biomass, color='r', marker='x', s=50, zorder = 2, label=f'target states')
        
        xticks = np.concat([xticks, target_times])
        yticks = np.concat([yticks, target_biomass])
    ax.set_xlabel('Time')
    ax.set_ylabel('Biomass')
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{x:.0f}' for x in xticks], rotation=45)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{y:.3f}' for y in yticks])
    ax.set_title('Target states')
    ax.legend(loc='upper right')
    fig.tight_layout()
    if save:
        path = f'{folder}'+'/EWS_spatial_targets.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'plot_targets(): Saved figure to {path}')

def main():
    ### SET THE PARAMETERS FOR PLOTTING ###
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'axes.titlesize': 12})
    plt.rcParams.update({'axes.labelsize': 12})
    plt.rcParams.update({'xtick.labelsize': 12})
    plt.rcParams.update({'ytick.labelsize': 12})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'figure.titlesize': 12})
    plt.rcParams.update({'axes.formatter.useoffset': False})
    plt.rcParams.update({'axes.formatter.use_mathtext': True})
    plt.rcParams.update({'axes.formatter.limits': (-3, 3)})
    

    ### SET THE PARAMETERS FOR THE SCRIPT ###
    input_folder = 'C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/LaTeX/Figures/Data/lin_prec_L2000'
    output_folder = f'C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/LaTeX/Figures'
    save_figs = True
    generate_new_data = False
    save_new_data = False
    
    ### IMPORT THE SIMULATIONS ###
    if generate_new_data:
        sims = load_simulations(input_folder)
        if len(sims) == 0:
            print('No simulations found.')
            return
        print('Generating new data...')
        target_times, target_biomass, stats = calculate_results(sims, resolution=500, ntimes=3, save_figs=save_figs, figs_folder=output_folder, verbose=True)
        
        ### SAVE THE RESULTS ###
        targets = pd.DataFrame({
            'local_times': target_times['local'],
            'local_biomass': target_biomass['local'],
            'global_times': target_times['global'],
            'global_biomass': target_biomass['global']
        })
        targets.to_csv(f'{input_folder}/EWS_spatial_agg_targets.csv', index=False)
        stats.to_csv(f'{input_folder}/EWS_spatial_agg_stats.csv', index=False)

    else:
        targets = pd.read_csv(f'{input_folder}/EWS_spatial_agg_targets.csv')
        target_times = {
            'local': targets['local_times'].values,
            'global': targets['global_times'].values
        }
        target_biomass = {
            'local': targets['local_biomass'].values,
            'global': targets['global_biomass'].values
        }
        stats = pd.read_csv(f'{input_folder}/EWS_spatial_agg_stats.csv')

    plot_results((target_times, target_biomass, stats), save_figs=save_figs, folder=output_folder)

    plt.show()

if __name__ == "__main__":
    main()