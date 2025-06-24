import numpy as np

def dispersion_coefficients(x, stability_class):
    stability_classes = {
        'A': (0.22, 0.20),
        'B': (0.16, 0.12),
        'C': (0.11, 0.08),
        'D': (0.08, 0.06),
        'E': (0.06, 0.03),
        'F': (0.04, 0.016)
    }
    a_y, a_z = stability_classes.get(stability_class.upper(), (0.08, 0.06))

    # Prevent invalid values by clipping x
    x_clipped = np.clip(x, 1e-3, None)
    sigma_y = a_y * x_clipped ** 0.9
    sigma_z = a_z * x_clipped ** 0.9
    return sigma_y, sigma_z


def gaussian_plume(x, y, z, Q, u, H, stability_class='D',
                   v_d=0.0, v_s=0.0, decay_rate=0.0):
    """
    Gaussian plume model with deposition, settling, and decay.

    Parameters:
    - x, y, z: Coordinates (can be numpy arrays)
    - Q: Emission rate (mass/time)
    - u: Wind speed (m/s)
    - H: Effective stack height
    - stability_class: A-F
    - v_d: Deposition velocity (m/s)
    - v_s: Settling velocity (m/s)
    - decay_rate: First-order decay constant (1/s)
    """
    sigma_y, sigma_z = dispersion_coefficients(x, stability_class)
    
    # Settling shift in vertical coordinate (z axis)
    z_eff = z - v_s * x / u

    part1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
    part2 = np.exp(-y**2 / (2 * sigma_y**2))
    part3 = np.exp(-(z_eff - H)**2 / (2 * sigma_z**2)) + np.exp(-(z_eff + H)**2 / (2 * sigma_z**2))

    # Decay (first-order loss)
    decay = np.exp(-decay_rate * x / u)

    # Deposition acts like an additional loss term
    deposition = np.exp(-v_d * x / u)

    return part1 * part2 * part3 * decay * deposition

def gaussian_puff(x, y, z, Q, u, H, stability_class, t, v_s=0, v_d=0, decay_rate=0):
    """
    Gaussian puff dispersion model with optional settling, deposition, and decay.

    - t: time since release [s]
    - Q: total mass [g] released in the puff (not rate)
    """

    # Puff spreads more slowly than plume
    sigma_y, sigma_z = dispersion_coefficients(u * t, stability_class)
    
    # Settling shifts puff downward
    z_eff = z - v_s * t

    # Base Gaussian terms
    part1 = Q / ((2 * np.pi)**1.5 * sigma_y * sigma_y * sigma_z)
    part2 = np.exp(-(x - u * t)**2 / (2 * sigma_y**2))
    part3 = np.exp(-y**2 / (2 * sigma_y**2))
    part4 = np.exp(-(z_eff - H)**2 / (2 * sigma_z**2)) + np.exp(-(z_eff + H)**2 / (2 * sigma_z**2))

    # Exponential decay
    decay = np.exp(-decay_rate * t)

    return part1 * part2 * part3 * part4 * decay
