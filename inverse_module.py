import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.gaussian import gaussian_plume

# --- Precompute grids and wind rotation ---
def compute_grid(lat0, lon0, zoom, u_east, u_north):
    extent, res = 500, 1
    x = np.linspace(-extent, extent, int(2 * extent / res))
    y = np.linspace(-extent, extent, int(2 * extent / res))
    X, Y = np.meshgrid(x, y)
    u_north_img = -u_north
    u = np.hypot(u_east, u_north_img)
    theta = np.arctan2(u_north_img, u_east)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    return X_rot, Y_rot, u

# --- Inverse Tab Builder ---
def build_inverse_tab(parent,
                      lat_entry, lon_entry,
                      ue_entry, un_entry,
                      vs_entry, vd_entry,
                      decay_entry, zoom_entry,
                      stability_var):
    # Sensor data holders
    sensor_lats = np.array([])
    sensor_lons = np.array([])
    C_obs = np.array([])

    def load_sensors():
        nonlocal sensor_lats, sensor_lons, C_obs
        path = filedialog.askopenfilename(
            title="Select sensor CSV", filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        sensor_lats, sensor_lons, C_obs = data[:,0], data[:,1], data[:,2]
        messagebox.showinfo("Loaded", f"{len(C_obs)} sensors loaded.")

    ttk.Button(parent, text="Load Sensors CSV", command=load_sensors).pack(pady=10)
    tk.Label(parent, text="Noise σ", bg="#f4f4f4").pack()
    sigma_entry = ttk.Entry(parent); sigma_entry.insert(0, "5.0"); sigma_entry.pack(pady=5)

    def run_inverse():
        lat0 = float(lat_entry.get()); lon0 = float(lon_entry.get())
        ue = float(ue_entry.get()); un = float(un_entry.get())
        v_s = float(vs_entry.get()); v_d = float(vd_entry.get())
        decay = float(decay_entry.get()); zoom = int(zoom_entry.get())
        stability = stability_var.get().upper()

        X_rot, Y_rot, u = compute_grid(lat0, lon0, zoom, ue, un)
        coords = np.linspace(-500, 500, X_rot.shape[0])
        sensor_idxs = []
        for la, lo in zip(sensor_lats, sensor_lons):
            dx = (lo - lon0) * 111000; dy = (la - lat0) * 111000
            ix = np.abs(coords - dx).argmin(); iy = np.abs(coords - dy).argmin()
            sensor_idxs.append((iy, ix))

        sigma = float(sigma_entry.get())

        with pm.Model() as model:
            x_src = pm.Uniform("x_src", 0, 100)
            y_src = pm.Uniform("y_src", 0, 100)
            Q = pm.LogNormal("Q", mu=np.log(50), sigma=1)
            preds = []
            for iy, ix in sensor_idxs:
                preds.append(
                    gaussian_plume(
                        X_rot[iy, ix], Y_rot[iy, ix], 0,
                        Q, u, 10, stability,
                        v_d=v_d, v_s=v_s, decay_rate=decay
                    )
                )
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=sigma)
            pm.Normal("obs", mu=preds, sigma=sigma_obs, observed=C_obs)

            idata = pm.sample(draws=1000, tune=1000, chains=4, cores=1, target_accept=0.95)

        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        pm.plot_posterior(idata, var_names=["x_src", "y_src", "Q"], ax=axes)
        plt.tight_layout()

        win = tk.Toplevel(parent)
        win.title("Posterior Distributions")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True)

    ttk.Button(parent, text="Run Inverse (PyMC)", command=run_inverse).pack(pady=15)