import numpy as np
import matplotlib.pyplot as plt


def grazing_model_step(r, K, c, h, σ, x, dt):
    """
    Perform one step of the grazing model using the Euler method with white noise.

    Parameters:
    r (float): Intrinsic growth rate of the plant.
    K (float): Carrying capacity of the environment.
    c (float): Grazing rate.
    h (float): Half-saturation constant.
    x (float): Current plant biomass.
    dt (float): Time step for the simulation.
    σ (float): Intensity of the white noise.

    Returns:
    x_new (float): Updated plant biomass after one time step.
    """
    dx = r * x * (1 - x / K) - c * x**2 / (x**2 + h**2)
    noise = σ * x * np.random.randn() / np.sqrt(dt)
    x_new = x + dx * dt + noise
    return x_new


def inflow_step(T, β, I, x):
    """
    Calculate the inflow of resource biomass using red noise.

    Parameters:
    T (float): Time scale over which noise becomes uncorrelated.
    β (float): Standard deviation of the normally distributed error term.
    I (float): Previous inflow value.
    x (float): Current plant biomass.

    Returns:
    I_new (float): Updated inflow value.
    """
    η = np.random.randn()
    I_new = ((1 - 1/T) * I + β * η) * x
    return I_new


# SCENARIO 1: Critical slowing down
# Initialize variables for simulation
x0 = 0.1  # Initial plant biomass
dt = 1  # Time step
T = 1000  # Total time
n_steps = int(T / dt)
t = np.linspace(0, T, n_steps)
x = np.zeros(n_steps)
x[0] = x0

# Parameters
r = 1.0  # Intrinsic growth rate
K = 10.0  # Carrying capacity
cs = np.linspace(1, 2.7, n_steps)  # Grazing rate
h = 1.0  # Half-saturation constant
σ = 0.03  # Noise strength

np.random.seed(2)

# Run the simulation
for i in range(1, n_steps):
    c = cs[i]
    x[i] = grazing_model_step(r, K, c, h, σ, x[i-1], dt)

x_obs = x + np.random.normal(0, 0.1, n_steps)

# Plot the results
plt.figure()
plt.plot(t, x_obs, label='Plant Biomass')
plt.plot(t, cs, label='Grazing Rate')
plt.axhline(2.604, color='r', linestyle='--',
            label='Critical Grazing Rate')
plt.xlabel('Time')
plt.ylabel('Biomass')
plt.title('Grazing Model (Noy-Meir, 1975), Critical Slowing Down')
plt.legend()
plt.show()

# # SCENARIO 2: Flickering
# # Initialize variables for simulation
# x0 = 0.5  # Initial plant biomass
# dt = 1  # Time step
# T = 10000  # Total time
# n_steps = int(T / dt)
# t = np.linspace(0, T, n_steps)
# x = np.zeros(n_steps)
# x[0] = x0

# # Parameters
# r = 1.0  # Intrinsic growth rate
# K = 10.0  # Carrying capacity
# cs = np.linspace(1, 2.7, n_steps)  # Grazing rate
# h = 1.0  # Half-saturation constant
# σ = 0.15  # Noise strength

# β = 0.07
# T_inflow = 20

# I = np.zeros(n_steps)
# I[0] = inflow_step(T_inflow, β, 0, x0)

# np.random.seed(2)

# # Run the simulation
# for i in range(1, n_steps):
#     c = cs[i]
#     I[i] = inflow_step(T_inflow, β, I[i-1], x[i-1])
#     x[i] = grazing_model_step(r, K, c, h, σ, x[i-1], dt) + I[i]

# # Plot the results
# plt.figure()
# plt.plot(t, x, label='Plant Biomass')
# plt.plot(t, cs, label='Grazing Rate')
# plt.axhline(2.604, color='r', linestyle='--',
#             label='Critical Grazing Rate')
# plt.xlabel('Time')
# plt.ylabel('Biomass')
# plt.title('Grazing Model (Noy-Meir, 1975), Flickering')
# plt.legend()
# plt.show()
