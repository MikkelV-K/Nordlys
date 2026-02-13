import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Constants & Physics Setup ---
R_E = 6371e3        # Earth Radius (m)
m_p = 1.672e-27     # Proton Mass (kg)
q = 1.602e-19       # Elementary Charge (C)
mu_0 = 4 * np.pi * 1e-7
B0 = 3.12e-5        # Surface magnetic field (T)

# Dipole Moment Magnitude (~8e22)
M_mag = 8.0e22      

# Tilt Angle (23.5 deg + 11 deg offset approx)
tilt = np.radians(34.5)
M_vec = M_mag * np.array([np.sin(tilt), 0, np.cos(tilt)])

def get_B_field(pos):
    """Calculates B-field vector at position pos (x,y,z)."""
    r = np.linalg.norm(pos)
    if r < R_E: return np.zeros(3) # Inside Earth
    
    r_hat = pos / r
    # Dipole Formula: (mu0/4pi) * (3(m.r)r - m) / r^3
    factor = 1e-7 / (r**3) # 1e-7 is mu0/4pi
    B = factor * (3 * np.dot(M_vec, r_hat) * r_hat - M_vec)
    return B

def get_derivatives(t, state):
    pos = state[:3]
    vel = state[3:]
    B = get_B_field(pos)
    acc = (q / m_p) * np.cross(vel, B)
    return np.concatenate([vel, acc])

# --- 2. The RK4 Solver ---
def rk4_step(t, state, dt):
    k1 = get_derivatives(t, state)
    k2 = get_derivatives(t + dt/2, state + k1 * dt/2)
    k3 = get_derivatives(t + dt/2, state + k2 * dt/2)
    k4 = get_derivatives(t + dt, state + k3 * dt)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- 3. Simulation Parameters ---
# Initial Position: 4 Earth Radii out on the negative X-axis
pos_0 = np.array([-10.0 * R_E, 0.0, 0.0]) 

# Initial Velocity: angled to create a spiral
v_mag = 1000000 
v_dir = np.array([1.0, 0, 0]) 
v_dir = v_dir / np.linalg.norm(v_dir)
vel_0 = v_mag * v_dir

state = np.concatenate([pos_0, vel_0])

dt = 0.001      # Time step
t_max = 300    # Increased time slightly to see more motion
steps = int(t_max / dt)

history = np.zeros((steps, 6))
history[0] = state

# --- 4. Run Loop ---
print(f"Simulating {steps} steps with RK4...")
for i in range(steps - 1):
    state = rk4_step(i*dt, state, dt)
    history[i+1] = state
    
    if np.linalg.norm(state[:3]) < R_E:
        print("Crashed into Earth!")
        history = history[:i+1]
        break

# --- 5. Visualization ---
pos_hist = history[:, :3] / R_E # Normalize to Earth Radii
x, y, z = pos_hist[:, 0], pos_hist[:, 1], pos_hist[:, 2]

# --- NEW: 2D PROJECTIONS ---
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))

# Common helper to draw Earth circle
def draw_earth(ax):
    circle = plt.Circle((0, 0), 1, color='green', alpha=0.1)
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

# 1. XY Plane (Top View)
axes[0].plot(x, y, lw=0.8, color='blue')
draw_earth(axes[0])
axes[0].set_title("XY Projection (Top View)")
axes[0].set_xlabel("X ($R_E$)")
axes[0].set_ylabel("Y ($R_E$)")

# 2. XZ Plane (Side View)
axes[1].plot(x, z, lw=0.8, color='red')
draw_earth(axes[1])
axes[1].set_title("XZ Projection (Side View)")
axes[1].set_xlabel("X ($R_E$)")
axes[1].set_ylabel("Z ($R_E$)")

# 3. YZ Plane (Front View)
axes[2].plot(y, z, lw=0.8, color='purple')
draw_earth(axes[2])
axes[2].set_title("YZ Projection (Front View)")
axes[2].set_xlabel("Y ($R_E$)")
axes[2].set_ylabel("Z ($R_E$)")

plt.tight_layout()
plt.show()

# --- 3D Visualization (Your original plot) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, lw=0.8, label="Proton Path")

# Earth Sphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
xe = np.cos(u)*np.sin(v)
ye = np.sin(u)*np.sin(v)
ze = np.cos(v)
ax.plot_wireframe(xe, ye, ze, color="blue", alpha=0.1)

ax.set_xlabel("X ($R_E$)")
ax.set_ylabel("Y ($R_E$)")
ax.set_zlabel("Z ($R_E$)")
ax.set_title("3D Trajectory")
plt.show()

# Energy Check (Your original plot)
velocities = history[:, 3:]
speeds = np.linalg.norm(velocities, axis=1)
plt.figure(figsize=(10, 3))
plt.plot(speeds)
plt.title("Test Condition: Speed vs Time (Energy Conservation)")
plt.ylabel("Speed (m/s)")
plt.xlabel("Step")
plt.ylim(min(speeds)*0.99, max(speeds)*1.01)
plt.show()