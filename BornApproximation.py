import numpy as np
import pyopenvdb as vdb
import os
from numba import njit, prange
from tqdm import tqdm
import time

# Start the timer
start_time = time.time()

# --- Global Constants ---
c = 1.0
frames = 100
pulse_periods = 5
fixed_beam_width = 30.0
scatterer_gain = 10
scatterer_domain = 10
pitch = 10.0
normalize_accumulated = True

# --- Incident Wave Direction ---
theta_deg = -20
phi_deg = 0
theta_xy = np.radians(theta_deg)
phi_z = np.radians(phi_deg)

nx = np.cos(phi_z) * np.cos(theta_xy)
ny = np.cos(phi_z) * np.sin(theta_xy)
nz = np.sin(phi_z)

# --- Grid Setup ---
x = np.linspace(0, 320, 256)
y = np.linspace(-80, 80, 128)
z = np.linspace(-80, 80, 128)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
t_vals = np.linspace(0, 300, frames)
scatterer_x = np.mean(x) - scatterer_domain

# --- Wavelengths to simulate ---
wavelengths = np.linspace(5, 15, 121)

@njit(parallel=True)
def compute_scattered_field(X, Y, Z, t, scatterer_positions, k, omega, c, gain, nx, ny, nz):
    Nx, Ny, Nz = X.shape
    field = np.zeros((Nx, Ny, Nz), dtype=np.float64)

    for i in prange(len(scatterer_positions)):
        x0, y0, z0 = scatterer_positions[i]

        # Calculate when the incident wavefront reaches this scatterer
        inc_distance = x0 * nx + y0 * ny + z0 * nz  # Dot product with wave direction
        t_inc = inc_distance / c

        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    dx = X[ix, iy, iz] - x0
                    dy = Y[ix, iy, iz] - y0
                    dz = Z[ix, iy, iz] - z0
                    R = np.sqrt(dx*dx + dy*dy + dz*dz)

                    t_required = t_inc + R / c  # Total time to get from source → scatterer → point

                    if t < t_required:
                        continue  # Scatterer hasn't had time to send wave to this point yet

                    # Incident phase at the scatterer when it was hit
                    phase_inc = k * inc_distance - omega * t_inc
                    inc_amp = np.cos(phase_inc)

                    # Scattered phase at current time
                    phase_scat = k * R - omega * (t - t_inc)
                    val = inc_amp * np.cos(phase_scat) / (R + 1e-3)
                    field[ix, iy, iz] += val * gain

    return field

def incident_field(X, Y, Z, t, k, omega):
    x0 = 2 + c * t * nx
    y0 = -(scatterer_x - 2) * ny / nx + c * t * ny
    z0 = 0 + c * t * nz
    sigma = fixed_beam_width
    R2 = (X - x0)**2 + 2*(Y - y0)**2 + 2*(Z - z0)**2
    envelope = np.exp(-R2 / (2 * sigma**2))
    phase = k * (nx * X + ny * Y + nz * Z) - omega * t
    return envelope * np.cos(phase)

# --- Main Simulation Loop ---
# Note that this main loop generates way more OpenVDB files then might be needed. It generated an animation for all wavelengths while you might only be interested in one.
counter = 0
for wavelength in wavelengths:
    print(f"\nSimulating wavelength: {wavelength:.2f}")
    k = 2 * np.pi / wavelength
    omega = k * c

    x_vals = [scatterer_x - pitch, scatterer_x, scatterer_x + pitch]
    y_vals = np.arange(-scatterer_domain, scatterer_domain + pitch, pitch)
    z_vals = np.arange(-scatterer_domain, scatterer_domain + pitch, pitch)
    scatterer_positions = np.array([(x, y, z) for x in x_vals for y in y_vals for z in z_vals], dtype=np.float64)

    output_dir = "output"
    os.makedirs(f"{output_dir}/{wavelength:.5f}/total", exist_ok=True)
    os.makedirs(f"{output_dir}/{wavelength:.5f}/inc", exist_ok=True)
    os.makedirs(f"{output_dir}/{wavelength:.5f}/scat", exist_ok=True)

    os.makedirs(f"{output_dir}/wavelength_sweep/scat", exist_ok=True)
    os.makedirs(f"{output_dir}/wavelength_sweep/inc", exist_ok=True)
    os.makedirs(f"{output_dir}/wavelength_sweep/total", exist_ok=True)

    accumulated_total = np.zeros_like(X)
    accumulated_inc = np.zeros_like(X)
    accumulated_scat = np.zeros_like(X)

    # Frame loop with export per frame
    for frame_idx, t in enumerate(tqdm(t_vals, desc=f"Wavelength {wavelength:.2f}")):
        inc = incident_field(X, Y, Z, t, k, omega)
        scat = compute_scattered_field(X, Y, Z, t, scatterer_positions, k, omega, c, scatterer_gain, nx, ny, nz)
        total = np.abs(inc + scat)**2

        # Accumulate intensities
        accumulated_total += total
        accumulated_scat += scat**2 
        accumulated_inc += inc**2

        # Write VDB for each component
        grid = vdb.FloatGrid()
        grid.copyFromArray(accumulated_total.astype(np.float32))
        grid.name = "intensity"
        vdb.write(f"{output_dir}/{wavelength:.5f}/total/accumulated_total_{frame_idx:03d}.vdb", grids=[grid])

        grid = vdb.FloatGrid()
        grid.copyFromArray(accumulated_scat.astype(np.float32))
        grid.name = "intensity"
        vdb.write(f"{output_dir}/{wavelength:.5f}/scat/accumulated_scat_{frame_idx:03d}.vdb", grids=[grid])

        grid = vdb.FloatGrid()
        grid.copyFromArray(accumulated_inc.astype(np.float32))
        grid.name = "intensity"
        vdb.write(f"{output_dir}/{wavelength:.5f}/inc/accumulated_inc_{frame_idx:03d}.vdb", grids=[grid])

        if t == t_vals[-1]:
            counter += 1
            # Write VDB for each component
            print("Last frame reached, saving in dedicated folder")
            print(f"Total min: {np.min(accumulated_total)}, total max: {np.max(accumulated_total)}")
            print(f"Scat min: {np.min(accumulated_scat)}, Scat max: {np.max(accumulated_scat)}")
            print(f"Inc min: {np.min(accumulated_inc)}, Inc max: {np.max(accumulated_inc)}")

            grid = vdb.FloatGrid()
            grid.copyFromArray(accumulated_total.astype(np.float32))
            grid.name = "intensity"
            vdb.write(f"{output_dir}/wavelength_sweep/total/accumulated_total_{counter:03d}.vdb", grids=[grid])

            grid = vdb.FloatGrid()
            grid.copyFromArray(accumulated_scat.astype(np.float32))
            grid.name = "intensity"
            vdb.write(f"{output_dir}/wavelength_sweep/scat/accumulated_scat_{counter:03d}.vdb", grids=[grid])

            grid = vdb.FloatGrid()
            grid.copyFromArray(accumulated_inc.astype(np.float32))
            grid.name = "intensity"
            vdb.write(f"{output_dir}/wavelength_sweep/inc/accumulated_inc_{counter:03d}.vdb", grids=[grid])

print("--- %s seconds ---" % (time.time() - start_time))
