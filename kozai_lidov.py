#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:03:56 2026

@author: resulayberksahpaz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# --- Physical Constants (SI) ---
G = 6.67430e-11
MSUN = 1.989e30
MJUP = 1.898e27
RJUP = 7.149e7
AU = 1.496e11
C = 2.9979e8
YEAR_SEC = 31557600

def secular_dynamics(t, y, m1, m2, m3, a_out, e_out, tv2_yr):
    a, e, inc, omega = y
    
    # Boundary clamping to prevent numerical instability
    e = max(min(e, 0.9999), 1e-6)
    
    # Kozai-Lidov (KL) Timescales
    p_in = 2 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))
    p_out = 2 * np.pi * np.sqrt(a_out**3 / (G * (m1 + m2 + m3)))
    tau_kl = (2 * p_out**2 / (3 * np.pi * p_in)) * ((m1 + m2 + m3) / m3) * (1 - e_out**2)**1.5
    
    # KL Dynamics (Eccentricity and Inclination trade-off)
    de_dt_kl = (15/8) * (e * np.sqrt(1 - e**2) / tau_kl) * np.sin(inc)**2 * np.sin(2 * omega)
    di_dt_kl = -(15/8) * (e**2 / (tau_kl * np.sqrt(1 - e**2))) * np.sin(inc) * np.cos(inc) * np.sin(2 * omega)
    domega_dt_kl = (3 / (4 * tau_kl * np.sqrt(1 - e**2))) * (2 * (1 - e**2) + 5 * np.sin(omega)**2 * (e**2 - np.sin(inc)**2))
    
    # GR Precession (Suppresses KL if too close to star)
    n = np.sqrt(G * (m1 + m2) / a**3)
    domega_dt_gr = (3 * G * (m1 + m2) * n) / (a * C**2 * (1 - e**2))
    
    # Tidal Migration (Dissipation)
    tv_sec = tv2_yr * YEAR_SEC
    tf2 = tv_sec * (9/8) * (a / RJUP)**6 * (m1 / m2)
    
    v2 = (9 / tf2) * (1 + (15/4)*e**2 + (15/8)*e**4 + (5/64)*e**6) / (1 - e**2)**6.5
    w2 = (1 / tf2) * (1 + (15/2)*e**2 + (45/8)*e**4 + (5/16)*e**6) / (1 - e**2)**6.5
    
    da_dt_tide = -a * (2 * v2 * (e**2 / (1 - e**2)) + 2 * w2)
    de_dt_tide = -e * (v2 + 9 * w2)
    
    return [da_dt_tide, de_dt_kl + de_dt_tide, di_dt_kl, domega_dt_kl + domega_dt_gr]

def run_migration_sim():
    # --- System Configuration ---
    m1, m2, m3 = 0.6 * MSUN, 0.390 * MJUP, 0.31 * MSUN
    a_in, e_in = 5.0 * AU, 0.128
    inc_tot = np.radians(87.05)
    omega_in = 0
    
    a_out, e_out = 80 * AU, 0.2
    tv2 = 0.1 # Viscous timescale parameter
    
    # --- Integration ---
    t_end = 2e6 * YEAR_SEC # 2 Million Years
    y0 = [a_in, e_in, inc_tot, omega_in]
    
    print("Simulating evolution...")
    sol = solve_ivp(secular_dynamics, (0, t_end), y0, 
                    args=(m1, m2, m3, a_out, e_out, tv2),
                    rtol=1e-8, atol=1e-10)
    
    visualize_migration(sol, m1, m3, a_out, e_out)

def get_orbit_path(a, e, inc, omega):
    theta = np.linspace(0, 2 * np.pi, 100)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    
    # Rotations for 3D plot
    x = x_orb * np.cos(omega) - y_orb * np.sin(omega)
    y_temp = x_orb * np.sin(omega) + y_orb * np.cos(omega)
    y = y_temp * np.cos(inc)
    z = y_temp * np.sin(inc)
    return x, y, z

def visualize_migration(sol, m1, m3, a_out, e_out):
    t_myr = sol.t / (1e6 * YEAR_SEC)
    y = sol.y
    p_out = 2 * np.pi * np.sqrt(a_out**3 / (G * (m1 + m3)))
    
    fig = plt.figure(figsize=(15, 8), facecolor='white')
    gs = fig.add_gridspec(3, 2)
    
    # Left Panels: Orbital Parameters
    ax_a = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[2, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
    
    ax_a.plot(t_myr, y[0]/AU, 'b'); ax_a.set_ylabel('a (AU)'); ax_a.set_title('Planet Semi-major Axis'); ax_a.grid(True)
    ax_e.plot(t_myr, y[1], 'r'); ax_e.set_ylabel('e'); ax_e.set_title('Planet Eccentricity'); ax_e.grid(True)
    
    r_sep = a_out * (1 - e_out * np.cos(2 * np.pi * sol.t / p_out))
    ax_s.plot(t_myr, r_sep/AU, color='gray'); ax_s.set_ylabel('r_sep (AU)'); ax_s.set_xlabel('Time (Myr)'); ax_s.grid(True)
    
    # Markers for animation
    h_a_mark, = ax_a.plot([], [], 'ro')
    h_e_mark, = ax_e.plot([], [], 'bo')
    h_s_mark, = ax_s.plot([], [], 'ko')
    
    # 3D Setup
    a1, a3 = a_out * (m3/(m1+m3)), a_out * (m1/(m1+m3))
    limit = (a3/AU) * 1.5
    
    h_star1, = ax_3d.plot([0], [0], [0], 'yo', markersize=12, label='Host Star')
    h_star3, = ax_3d.plot([0], [0], [0], 'ko', markersize=8, label='Companion')
    h_orbit, = ax_3d.plot([], [], [], 'k-', alpha=0.4)
    h_planet, = ax_3d.plot([], [], [], 'ro', markersize=6)
    
    ax_3d.set_xlim([-limit, limit]); ax_3d.set_ylim([-limit, limit]); ax_3d.set_zlim([-limit/2, limit/2])
    ax_3d.view_init(30, 45)

    def animate(i):
        # Sampling the data for smoother animation
        idx = i * (len(sol.t) // 350)
        if idx >= len(sol.t): idx = len(sol.t) - 1
        
        # Companion and Host star positions
        phase = 2 * np.pi * (sol.t[idx] / p_out)
        r_f = (1 - e_out * np.cos(phase))
        x1, y1 = -a1 * r_f * np.cos(phase), -a1 * r_f * np.sin(phase)
        x3, y3 =  a3 * r_f * np.cos(phase),  a3 * r_f * np.sin(phase)
        
        h_star1.set_data([x1/AU], [y1/AU]); h_star1.set_3d_properties([0])
        h_star3.set_data([x3/AU], [y3/AU]); h_star3.set_3d_properties([0])
        
        # Planet Orbit Path
        ox, oy, oz = get_orbit_path(y[0,idx], y[1,idx], y[2,idx], y[3,idx])
        h_orbit.set_data((ox+x1)/AU, (oy+y1)/AU); h_orbit.set_3d_properties(oz/AU)
        
        # Planet Position (at periastron for visualization)
        rp = y[0,idx] * (1 - y[1,idx])
        px = (rp * np.cos(y[3,idx]) + x1) / AU
        py = (rp * np.sin(y[3,idx]) * np.cos(y[2,idx]) + y1) / AU
        pz = (rp * np.sin(y[3,idx]) * np.sin(y[2,idx])) / AU
        h_planet.set_data([px], [py]); h_planet.set_3d_properties(pz)
        
        # Graph Markers
        h_a_mark.set_data([t_myr[idx]], [y[0,idx]/AU])
        h_e_mark.set_data([t_myr[idx]], [y[1,idx]])
        h_s_mark.set_data([t_myr[idx]], [r_sep[idx]/AU])
        
        return h_star1, h_star3, h_orbit, h_planet, h_a_mark, h_e_mark, h_s_mark

    ani = FuncAnimation(fig, animate, frames=350, interval=30, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_migration_sim()