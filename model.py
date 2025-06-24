from map import fetch_tile, point_to_pixels, pixels_to_point, TILE_SIZE
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from models.gaussian import gaussian_plume, gaussian_puff


def run_model(lat0, lon0, Q, u_east, u_north, H, stability, zoom,
              v_s=0.0, v_d=0.0, decay_rate=0.0, model_type='Plume'):

    # --- Get center pixel coordinates and tile info ---
    x0_pix, y0_pix = point_to_pixels(lon0, lat0, zoom)
    center_tile_x = int(x0_pix // TILE_SIZE)
    center_tile_y = int(y0_pix // TILE_SIZE)

    # --- Fetch 3x3 tile grid ---
    tiles = {}
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            tx, ty = center_tile_x + dx, center_tile_y + dy
            try:
                tiles[(dx, dy)] = fetch_tile(tx, ty, zoom)
            except Exception as e:
                print(f"Tile fetch error at ({tx}, {ty}): {e}")
                tiles[(dx, dy)] = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255, 255, 255, 255))

    # --- Stitch tiles ---
    full_map = Image.new("RGBA", (3 * TILE_SIZE, 3 * TILE_SIZE))
    for (dx, dy), tile in tiles.items():
        full_map.paste(tile, ((dx + 1) * TILE_SIZE, (dy + 1) * TILE_SIZE))

    # --- Grid for simulation (in meters) ---
    extent = 500  # meters
    res = 1       # meter resolution
    x = np.linspace(-extent, extent, int(2 * extent / res))
    y = np.linspace(-extent, extent, int(2 * extent / res))
    X, Y = np.meshgrid(x, y)
    Z = 0  # ground level

    # --- Wind adjustment ---
    u_north = -u_north  # Convert to image coordinates
    u = np.hypot(u_east, u_north)
    theta = np.arctan2(u_north, u_east)

    # --- Rotate grid with wind ---
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # --- Compute overlay offsets ---
    x_in_tile = x0_pix % TILE_SIZE
    y_in_tile = y0_pix % TILE_SIZE
    x_offset = TILE_SIZE + int(x_in_tile)
    y_offset = TILE_SIZE + int(y_in_tile)
    x_draw = x_offset - X.shape[1] // 2
    y_draw = y_offset - X.shape[0] // 2

    # --- Geographic bounds for plotting ---
    top_left_px = (center_tile_x - 1) * TILE_SIZE
    top_left_py = (center_tile_y - 1) * TILE_SIZE
    bottom_right_px = top_left_px + 3 * TILE_SIZE
    bottom_right_py = top_left_py + 3 * TILE_SIZE
    min_lon, max_lat = pixels_to_point(top_left_px, top_left_py, zoom)
    max_lon, min_lat = pixels_to_point(bottom_right_px, bottom_right_py, zoom)

    if model_type == 'Plume':
        # --- compute and composite exactly as you had it ---
        C = gaussian_plume(X_rot, Y_rot, Z, Q, u, H, stability,
                        v_d=v_d, v_s=v_s, decay_rate=decay_rate)
        C_norm = C / np.max(C) if np.max(C) > 0 else C
        C_rgba = plt.cm.plasma(C_norm)
        C_rgba[..., 3] = np.sqrt(C_norm)
        plume_img = Image.fromarray((C_rgba * 255).astype(np.uint8))

        map_with_plume = full_map.copy()
        map_with_plume.paste(plume_img, (x_draw, y_draw), plume_img)

        # --- now show that composite once, with the correct geographic extent ---
        overlay_arr = np.asarray(map_with_plume)

        fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        im = ax.imshow(
            overlay_arr,
            extent=[min_lon, max_lon, min_lat, max_lat]
        )

        # --- build a ScalarMappable only for the colorbar ---
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(C))
        sm   = mpl.cm.ScalarMappable(cmap='plasma', norm=norm)
        sm.set_array([])    # required placeholder
        cbar = fig.colorbar(sm, ax=ax, label='Concentration')

        # --- plot the source and finish up ---
        ax.scatter(lon0, lat0, color='cyan', s=50, marker='x', label='Source')
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Gaussian Plume (Q={Q:.1f} g/s)")
        ax.grid(True)
        ax.legend()
        plt.show()

    elif model_type == 'Puff':
        # --- Animated Gaussian puff ---
        t_max = 100
        dt = 5
        times = np.arange(0, t_max + dt, dt)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Gaussian Puff Dispersion")
        ax.grid(True)
        ax.scatter([lon0], [lat0], color='cyan', marker='x', s=50, label='Source')
        ax.legend()

        img_artist = ax.imshow(np.zeros((X.shape[0], X.shape[1], 4)),
                               extent=[min_lon, max_lon, min_lat, max_lat])

        def update(t):
            C = gaussian_puff(X_rot, Y_rot, Z, Q, u, H, stability, t=t,
                              v_d=v_d, v_s=v_s, decay_rate=decay_rate)
            C_norm = C / np.max(C) if np.max(C) > 0 else C
            C_rgba = plt.cm.plasma(C_norm)
            C_rgba[..., 3] = np.sqrt(C_norm)
            puff_img = Image.fromarray((C_rgba * 255).astype(np.uint8))

            overlay = full_map.copy()
            overlay.paste(puff_img, (x_draw, y_draw), puff_img)
            img_artist.set_data(np.asarray(overlay))
            return [img_artist]

        ani = FuncAnimation(fig, update, frames=times, interval=200, blit=False)
        plt.tight_layout()
        plt.show()
