import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from mods.files import *
from mods.utilities import get_unique_aliases
from mods.buffers import StateBuffer, DataBuffer
from mods.simulation import Simulation
from mods.fields import DensityFieldCustom
from mods.plant import Plant, PlantCollection

def spatial_correlation_function(field):
    """Calculate the spatial correlation function for a 2D field."""
    mean = np.mean(field)
    var = np.var(field)
    M, N = field.shape
    
    distances = np.arange(1, min(M, N) // 2 + 1)
    corr = np.zeros_like(distances, dtype=float)
    
    for d in distances:
        W = (M - d) * (N - d)    
        if W <= 0:
            continue
        corr[d-1] = (M * N / W) * np.sum((field[:-d, :-d] - mean) * (field[d:, d:] - mean)) / var
    return distances, corr

# Define the discrete fourier transform function for a 2D field
def spatial_power_spectrum(dft):
    """Calculate the spatial power spectrum for a 2D field."""
    M, N = field.shape
    DFT = np.fft.fft2(field) / (M * N)
    power_spectrum = np.abs(DFT)**2  # Power spectrum
    power_spectrum = np.fft.fftshift(power_spectrum)  # Shift zero frequency to center
    return power_spectrum

def calculate_r_spectrum(power_spectrum):
    """Calculate the radial spectrum from the power spectrum."""
    M, N = power_spectrum.shape
    X, Y = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    grid_indices = np.vstack([X.flatten(), Y.flatten()]).T
    
    distances = np.linalg.norm(grid_indices, axis=1)
    unique_distances = np.unique(distances)
    unique_distances = np.sort([d for d in unique_distances if d > 0])
    r_spectrum = np.zeros_like(unique_distances)
    
    for i, d in enumerate(unique_distances):
        mask = distances == d
        if np.sum(mask) == 0:
            continue
        r_spectrum[i] = np.sum(power_spectrum.flatten()[mask])
    
    return unique_distances, r_spectrum

def calculate_theta_spectrum(power_spectrum):
    """Calculate the angular spectrum from the power spectrum."""
    M, N = power_spectrum.shape
    X, Y = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
    mask_dist = (X/M)**2 + (Y/N)**2 < 0.5**2
    power_spectrum = power_spectrum * mask_dist
    radians = np.arctan2(Y.flatten(), X.flatten())
    
    unique_radians = np.unique(radians)
    unique_radians = np.sort([r for r in unique_radians if r != 0])
    theta_spectrum = np.zeros_like(unique_radians)
        
    for i, theta in enumerate(unique_radians):
        mask = radians == theta
        if np.sum(mask) == 0:
            continue
        theta_spectrum[i] = np.sum(power_spectrum.flatten()[mask])
    
    return unique_radians, theta_spectrum

folder = 'Data/lin_prec_test'
alias = 'linprec_global_L2000_250430_114412'
save_fig = False

kwargs = pd.read_json(
    f'{folder}/kwargs-{alias}.json', typ='series').to_dict()
sim = Simulation(folder=folder, alias=alias, **kwargs)

tt = np.linspace(10000, 13500, 4).astype(int)
db_data = sim.data_buffer.get_data()
fig, axs, title = DataBuffer.plot(db_data)
for t in tt:
    for ax in axs:
        ax.axvline(x=t, color='r', linestyle='--', label=f't = {t}')
        ax.legend()

density_field = DensityFieldCustom(box = sim.box, resolution = 100)
field_list = []
distance_corr_list = []
corr_list = []
power_spectrum_list = []
unique_distances_list = []
unique_radians_list = []
r_spectrum_list = []
θ_spectrum_list = []
for i, t in enumerate(tt):
    print()
    print(f'Processing t = {t}...')
    sb_data = sim.state_buffer.get_specific_data(t)
    if sb_data.empty:
        tt = np.delete(tt, np.where(tt == t))
        continue
    plants = PlantCollection()
    x, y = sb_data['x'], sb_data['y']
    r = sb_data['r']
    ids = sb_data['id']
    for j in range(len(x)):
        # Create a new plant object for each plant in the simulation
        # and add it to the collection
        plant = Plant(id=ids[j], x=x[j], y=y[j], r=r[j], **sim.species_list[0].__dict__)
        plants.add_plant(plant)
    density_field.update(plants)
    
    field = density_field.values.reshape((density_field.resolution, density_field.resolution))
    distances_corr, corr = spatial_correlation_function(field)
    power_spectrum = spatial_power_spectrum(field)
    unique_distances, r_spectrum = calculate_r_spectrum(power_spectrum)
    unique_radians, θ_spectrum = calculate_theta_spectrum(power_spectrum)
    
    field_list.append(field)
    distance_corr_list.append(distances_corr)
    corr_list.append(corr)
    power_spectrum_list.append(power_spectrum)
    unique_distances_list.append(unique_distances)
    unique_radians_list.append(unique_radians)
    r_spectrum_list.append(r_spectrum)
    θ_spectrum_list.append(θ_spectrum)


fig_field, ax_field = plt.subplots(len(tt), 1, figsize=(4, 6), sharex=True, sharey=True)
fig_corre, ax_corre = plt.subplots(len(tt), 1, figsize=(4, 6), sharex=True, sharey=True)
fig_pspec, ax_pspec = plt.subplots(len(tt), 1, figsize=(4, 6), sharex=True, sharey=True)
fig_rspec, ax_rspec = plt.subplots(len(tt), 1, figsize=(4, 6), sharex=True, sharey=True)
fig_tspec, ax_tspec = plt.subplots(len(tt), 1, figsize=(4, 6), sharex=True, sharey=True)
fig_field.suptitle('Density Field')
fig_corre.suptitle('Spatial Correlation Function')
fig_pspec.suptitle('Power Spectrum')
fig_rspec.suptitle('Radial Power Spectrum')
fig_tspec.suptitle('Angular Power Spectrum')
figs = [fig_field, fig_corre, fig_pspec, fig_rspec, fig_tspec]

plt.set_cmap('Greys')
for i, t in enumerate(tt):
    print(f'Plotting t = {t}...')
    if isinstance(ax_field, plt.Axes):
        ax_field = [ax_field]
    if isinstance(ax_corre, plt.Axes):
        ax_corre = [ax_corre]
    if isinstance(ax_pspec, plt.Axes):
        ax_pspec = [ax_pspec]
    if isinstance(ax_rspec, plt.Axes):
        ax_rspec = [ax_rspec]
    if isinstance(ax_tspec, plt.Axes):
        ax_tspec = [ax_tspec]
    
    ax_field[i].set_title(f't = {t}', fontsize=10)
    ax_corre[i].set_title(f't = {t}', fontsize=10)
    ax_pspec[i].set_title(f't = {t}', fontsize=10)
    ax_rspec[i].set_title(f't = {t}', fontsize=10)
    ax_tspec[i].set_title(f't = {t}', fontsize=10)
    
    
        
    ax_field[i].imshow(field_list[i], origin='lower')
    ax_field[i].set_xticks([])
    ax_field[i].set_yticks([])
    
    ax_corre[i].plot(distance_corr_list[i]/density_field.resolution * sim.L, corr_list[i], '-o', markersize=2)
    ax_corre[i].set_xlabel('Distance', fontsize=7)
    ax_corre[i].set_ylabel('Correlation', fontsize=7)
    # ax_corre[i].set_yscale('log')
       
    power_spectrum_log = np.log10(power_spectrum_list[i] + 1e-10)  # Log scale for better visibility
    ax_pspec[i].imshow(power_spectrum_log/np.var(field_list[i]))
    ax_pspec[i].set_xticks([])
    ax_pspec[i].set_yticks([])
    ax_pspec[i].set_aspect('equal')

    ax_rspec[i].plot(unique_distances_list[i]/density_field.resolution*sim.L, r_spectrum_list[i], 'g-o', markersize=2)
    ax_rspec[i].set_xlabel('Distance', fontsize=7)
    ax_rspec[i].set_ylabel('Power', fontsize=7)
    # ax_rspec[i].set_yscale('log')

    ax_tspec[i].plot(unique_radians_list[i], θ_spectrum_list[i], 'g-o', markersize=2)
    ax_tspec[i].set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi/2))
    ax_tspec[i].set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax_tspec[i].set_xlabel('Angle', fontsize=7)
    ax_tspec[i].set_ylabel('Power', fontsize=7)
    # ax_tspec[i].set_yscale('log')
    
for fig in figs:
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.5)
    if save_fig:
        fig.savefig(f'{folder}/figs/{alias}_spatial_analysis.png', dpi=300, bbox_inches='tight')

if save_fig:
    plt.close('all')
else:
    plt.show()
    
        