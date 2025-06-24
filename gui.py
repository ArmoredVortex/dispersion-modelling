import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.font import Font

from map import fetch_tile, point_to_pixels, pixels_to_point, TILE_SIZE
from model import run_model
import inverse_module  # import the separated inverse code

# --- Tkinter GUI ---
def start_gui():
    root = tk.Tk()
    root.title("Gaussian Dispersion Visualizer + Inverse (PyMC)")
    root.geometry("600x820")
    root.configure(bg="#f4f4f4")

    title_font = Font(family="Helvetica", size=16, weight="bold")
    label_font = Font(family="Helvetica", size=10)

    tk.Label(root, text="Gaussian Dispersion Visualizer + Inverse (PyMC)",
             font=title_font, bg="#f4f4f4").pack(pady=10)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # --- Forward Model Tab ---
    forward_frame = tk.Frame(notebook, bg="#f4f4f4")
    notebook.add(forward_frame, text="Forward Model")
    frame_f = tk.Frame(forward_frame, bg="#f4f4f4")
    frame_f.pack(pady=5)

    # Input fields for forward
    fields_f = [
        ("Latitude", "26.208938"), ("Longitude", "78.183051"),
        ("Emission Rate Q [g/s]", "1.0"),
        ("Wind East u_east [m/s]", "2.0"),
        ("Wind North u_north [m/s]", "-3.0"),
        ("Stack Height H [m]", "10.0"), ("Zoom Level", "16"),
        ("Settling v_s [m/s]", "0.01"),
        ("Deposition v_d [m/s]", "0.005"),
        ("Decay Rate [1/s]", "0.001")
    ]
    entries_f = {}
    for i,(lbl,df) in enumerate(fields_f):
        tk.Label(frame_f, text=lbl, font=label_font, bg="#f4f4f4").grid(
            row=i, column=0, sticky="e", pady=3, padx=5)
        e = ttk.Entry(frame_f, width=25); e.insert(0,df)
        e.grid(row=i, column=1, pady=3)
        entries_f[lbl] = e

    lat_f = entries_f["Latitude"]; lon_f = entries_f["Longitude"]
    Q_f = entries_f["Emission Rate Q [g/s]"]
    ue_f = entries_f["Wind East u_east [m/s]"]
    un_f = entries_f["Wind North u_north [m/s]"]
    H_f  = entries_f["Stack Height H [m]"]
    zoom_f = entries_f["Zoom Level"]
    vs_f = entries_f["Settling v_s [m/s]"]
    vd_f = entries_f["Deposition v_d [m/s]"]
    decay_f = entries_f["Decay Rate [1/s]"]

    stability_f = tk.StringVar(value='A')
    tk.Label(frame_f, text="Stability Class (A-F)", font=label_font,bg="#f4f4f4").grid(
        row=len(fields_f), column=0, sticky="e", pady=3, padx=5)
    ttk.Combobox(frame_f, textvariable=stability_f, values=list("ABCDEF"), width=22).grid(
        row=len(fields_f), column=1, pady=3)

    mt_f = tk.StringVar(value='Plume')
    tk.Label(frame_f, text="Model Type", font=label_font,bg="#f4f4f4").grid(
        row=len(fields_f)+1, column=0, sticky="e", pady=3, padx=5)
    ttk.Combobox(frame_f, textvariable=mt_f, values=["Plume","Puff"], width=22).grid(
        row=len(fields_f)+1, column=1, pady=3)

    ttk.Button(forward_frame, text="Run Forward Model",
               command=lambda: run_model(
                   float(lat_f.get()), float(lon_f.get()),
                   float(Q_f.get()), float(ue_f.get()), float(un_f.get()),
                   float(H_f.get()), stability_f.get(), int(zoom_f.get()),
                   v_s=float(vs_f.get()), v_d=float(vd_f.get()),
                   decay_rate=float(decay_f.get()), model_type=mt_f.get()
               )).pack(pady=15)

    # --- Inverse Estimation Tab ---
    inv_frame = tk.Frame(notebook, bg="#f4f4f4")
    notebook.add(inv_frame, text="Inverse Estimation")
    frame_i = tk.Frame(inv_frame, bg="#f4f4f4")
    frame_i.pack(pady=5)

    # New inverse reference lat/lon inputs
    tk.Label(frame_i, text="Inverse Ref Latitude", font=label_font, bg="#f4f4f4").grid(
        row=0, column=0, sticky="e", padx=5)
    lat_i = ttk.Entry(frame_i, width=20); lat_i.insert(0, "26.208938")
    lat_i.grid(row=0, column=1, pady=3)

    tk.Label(frame_i, text="Inverse Ref Longitude", font=label_font, bg="#f4f4f4").grid(
        row=1, column=0, sticky="e", padx=5)
    lon_i = ttk.Entry(frame_i, width=20); lon_i.insert(0, "78.183051")
    lon_i.grid(row=1, column=1, pady=3)

    # Delegate building the rest of the inverse tab, passing new lat_i, lon_i
    inverse_module.build_inverse_tab(
        parent=inv_frame,
        lat_entry=lat_i,
        lon_entry=lon_i,
        ue_entry=ue_f,
        un_entry=un_f,
        vs_entry=vs_f,
        vd_entry=vd_f,
        decay_entry=decay_f,
        zoom_entry=zoom_f,
        stability_var=stability_f
    )

    root.mainloop()

if __name__ == '__main__':
    start_gui()
