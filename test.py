import numpy as np
import matplotlib.pyplot as plt

# --- 1. PHYSICAL CONSTANTS ---
m_p = 1.67e-27          # Mass of proton (kg)
q = 1.602e-19           # Charge of proton (C)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space
Re = 6.371e6            # Earth radius (m)

# Dipole magnitude approx 8e22 A*m^2
# The assignment asks for a "tilted" dipole.
# Let's tilt it 20 degrees relative to the Z-axis (ecliptic normal).
tilt_deg = 20
tilt_rad = np.radians(tilt_deg)
M_mag = 8.0e22 
M_vec = np.array([M_mag * np.sin(tilt_rad), 0, M_mag * np.cos(tilt_rad)])

# --- 2. HELPER FUNCTIONS ---

def get_B_field(pos):
    """
    Calculates the magnetic field B at position 'pos' (x, y, z)
    using the dipole formula: B(r) = (mu0 / 4pi) * (3(m . r_hat)r_hat - m) / r^3
    """
    r = np.linalg.norm(pos)
    
    # Avoid division by zero at the center
    if r < Re: 
        return np.array([0.0, 0.0, 0.0]) # Inside Earth (simplified)
        
    r_hat = pos / r
    
    # Dot product (m . r_hat)
    m_dot_r = np.dot(M_vec, r_hat)
    
    # Vector math for the dipole
    factor = (mu0 / (4 * np.pi)) / (r**3)
    B = factor * (3 * m_dot_r * r_hat - M_vec)
    return B

def get_acceleration(vel, B):
    """
    Calculates acceleration via Lorentz force: F = q(v x B).
    Since F = ma, a = (q/m)(v x B).
    Note: Magnetic fields do NO work, so speed |v| should be constant[cite: 21].
    """
    cross_prod = np.cross(vel, B)
    acc = (q / m_p) * cross_prod
    return acc

# --- 3. SIMULATION SETUP ---

# Initial Conditions [cite: 12]
# Start 10 Earth Radii away on the X-axis (Sun side)
r0 = np.array([10.0 * Re, 0.0, 0.0])

# Solar wind velocity (approx 400 km/s) moving TOWARD Earth (-x direction)
v0 = np.array([-400000.0, 0.0, 0.0]) 

# Time settings
# CRITICAL: Euler's method needs a tiny step to handle the magnetic spiral.
# If the path flies off to infinity, reduce dt.
dt = 0.0001       # Time step (seconds)
t_max = 100.0   # Total simulation time (seconds)
steps = int(t_max / dt)

# Arrays to store trajectory for plotting
positions = np.zeros((steps, 3))
velocities = np.zeros((steps, 3))
time_array = np.linspace(0, t_max, steps)

# Set initial values
positions[0] = r0
velocities[0] = v0

# --- 4. EULER'S METHOD LOOP  ---

print(f"Simulating {steps} steps...")

for i in range(steps - 1):
    curr_pos = positions[i]
    curr_vel = velocities[i]
    
    # A. Calculate Field and Acceleration
    B = get_B_field(curr_pos)
    acc = get_acceleration(curr_vel, B)
    
    # B. Euler Step: x(t+dt) = x(t) + v(t)dt
    next_pos = curr_pos + curr_vel * dt
    
    # C. Euler Step: v(t+dt) = v(t) + a(t)dt
    next_vel = curr_vel + acc * dt
    
    # D. Store results
    positions[i+1] = next_pos
    velocities[i+1] = next_vel

# --- 5. TEST CONDITION: ENERGY CONSERVATION [cite: 15] ---

# Calculate speed at every step: |v| = sqrt(vx^2 + vy^2 + vz^2)
speeds = np.linalg.norm(velocities, axis=1)

# Plot Speed vs Time to check accuracy
plt.figure(figsize=(10, 4))
plt.plot(time_array, speeds)
plt.title("Test Condition: Speed vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid(True)
# If this line goes UP, your simulation is gaining fake energy (dt is too big).
plt.show()

# --- 6. PLOTTING THE TRAJECTORY [cite: 17] ---

# Normalized to Earth Radii for easier viewing
pos_Re = positions / Re 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(pos_Re[:,0], pos_Re[:,1], pos_Re[:,2], label='Proton Path')

# Draw Earth at the center for reference
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_e = np.cos(u)*np.sin(v)
y_e = np.sin(u)*np.sin(v)
z_e = np.cos(v)
ax.plot_wireframe(x_e, y_e, z_e, color="green", alpha=0.3, label="Earth")

ax.set_xlabel("X (Earth Radii)")
ax.set_ylabel("Y (Earth Radii)")
ax.set_zlabel("Z (Earth Radii)")
ax.set_title("Proton Trajectory in Dipole Field")
ax.legend()
plt.show()