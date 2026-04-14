#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:58:31 2026

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

def secular_dynamics_reverse(t, y, m1, m2, m3, a_out, e_out, tv2_yr):
    a, e, inc, omega = y
    
    # Boundary clamping for eccentricity
    e = max(min(e, 0.9999), 1e-6)
    
    # Kozai-Lidov Timescale
    p_in = 2 * np.pi * np.sqrt(a**3 / (G * (m1 + m2)))
    p_out = 2 * np.pi * np.sqrt(a_out**3 / (G * (m1 + m2 + m3)))
    tau_kl = (2 * p_out**2 / (3 * np.pi * p_in)) * ((m1 + m2 + m3) / m3) * (1 - e_out**2)**1.5
    
    # KL Dynamics
    de_dt_kl = (15/8) * (e * np.sqrt(1 - e**2) / tau_kl) * np.sin(inc)**2 * np.sin(2 * omega)
    di_dt_kl = -(15/8) * (e**2 / (tau_kl * np.sqrt(1 - e**2))) * np.sin(inc) * np.cos(inc) * np.sin(2 * omega)
    domega_dt_kl = (3 / (4 * tau_kl * np.sqrt(1 - e**2))) * (2 * (1 - e**2) + 5 * np.sin(omega)**2 * (e**2 - np.sin(inc)**2))
    
    # GR Precession
    n = np.sqrt(G * (m1 + m2) / a**3)
    domega_dt_gr = (3 * G * (m1 + m2) * n) / (a * C**2 * (1 - e**2))
    
    # Tidal Migration (Anti-dissipation logic)
    tv_sec = tv2_yr * YEAR_SEC
    tf2 = tv_sec * (9/8) * (a / RJUP)**6 * (m1 / m2)
    
    v2 = (9 / tf2) * (1 + (15/4)*e**2 + (15/8)*e**4 + (5/64)*e**6) / (1 - e**2)**6.5
    w2 = (1 / tf2) * (1 + (15/2)*e**2 + (45/8)*e**4 + (5/16)*e**6) / (1 - e**2)**6.5
    
    da_dt_tide = -a * (2 * v2 * (e**2 / (1 - e**2)) + 2 * w2)
    de_dt_tide = -e * (v2 + 9 * w2)
    
    return [da_dt_tide, de_dt_kl + de_dt_tide, di_dt_kl, domega_dt_kl + domega_dt_gr]

def run_simulation():
    # --- System Setup ---
    m1, m2, m3 = 0.6 * MSUN, 0.390 * MJUP, 0.31 * MSUN
    a_out, e_out, tv2 = 80 * AU, 0.2, 0.1
    
    # Current state
    p_days = 10
    target_p_sec = p_days * 24 * 3600
    a_start = ((target_p_sec * np.sqrt(G * (m1 + m2))) / (2 * np.pi))**(2/3)
    y0 = [a_start, 0.01, np.radians(85), np.radians(90)]
    
    # Integration range (0 to -100 Myr)
    t_span = (0, -100e12 * YEAR_SEC)
    a_birth_limit = 5.0 * AU
    
    # Stop event: when a reaches the birth limit
    def stop_at_birth(t, y, *args):
        return y[0] - a_birth_limit
    stop_at_birth.terminal = True
    stop_at_birth.direction = 1 # 'a' increases moving backwards
    
    print("Running Time-Reversal Integration...")
    sol = solve_ivp(secular_dynamics_reverse, t_span, y0, 
                    args=(m1, m2, m3, a_out, e_out, tv2),
                    events=stop_at_birth, rtol=1e-8, atol=1e-10)
    
    # Print Stats
    t_myr = np.abs(sol.t / (1e6 * YEAR_SEC))
    print(f"Simulation went back to: {t_myr[-1]:.2f} Myr ago")
    print(f"Original Semi-major Axis: {sol.y[0,-1]/AU:.4f} AU")
    print(f"Original Eccentricity: {sol.y[1,-1]:.4f}")
    
    visualize(sol, m1, m3, a_out, e_out)

def get_orbit_path(a, e, inc, omega):
    theta = np.linspace(0, 2 * np.pi, 100)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x_orb = r * np.cos(theta)
    y_orb = r * np.sin(theta)
    
    # Rotations
    x = x_orb * np.cos(omega) - y_orb * np.sin(omega)
    y_temp = x_orb * np.sin(omega) + y_orb * np.cos(omega)
    y = y_temp * np.cos(inc)
    z = y_temp * np.sin(inc)
    return x, y, z

def visualize(sol, m1, m3, a_out, e_out):
    t_myr = np.abs(sol.t / (1e6 * YEAR_SEC))
    y = sol.y
    p_out = 2 * np.pi * np.sqrt(a_out**3 / (G * (m1 + m3)))
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 2)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_s = fig.add_subplot(gs[2, 0])
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
    
    # Plot static lines
    ax_a.plot(t_myr, y[0]/AU, 'b'); ax_a.set_ylabel('a (AU)'); ax_a.grid(True)
    ax_e.plot(t_myr, y[1], 'r'); ax_e.set_ylabel('e'); ax_e.grid(True)
    
    r_sep = a_out * (1 - e_out * np.cos(2 * np.pi * sol.t / p_out))
    ax_s.plot(t_myr, r_sep/AU, color='gray'); ax_s.set_ylabel('r_sep (AU)'); ax_s.set_xlabel('Myr Ago'); ax_s.grid(True)
    
    # 3D setup
    a1, a3 = a_out * (m3/(m1+m3)), a_out * (m1/(m1+m3))
    limit = (a3/AU) * 1.2
    
    h_star1, = ax_3d.plot([0], [0], [0], 'yo', markersize=10)
    h_star3, = ax_3d.plot([0], [0], [0], 'ko', markersize=8)
    h_orbit, = ax_3d.plot([], [], [], 'k-', alpha=0.5)
    h_planet, = ax_3d.plot([], [], [], 'ro')
    
    ax_3d.set_xlim([-limit, limit]); ax_3d.set_ylim([-limit, limit]); ax_3d.set_zlim([-limit/2, limit/2])
    
    def update(i):
        idx = i * (len(sol.t) // 200) # Speed up animation
        if idx >= len(sol.t): idx = len(sol.t) - 1
        
        phase = 2 * np.pi * (sol.t[idx] / p_out)
        r_f = (1 - e_out * np.cos(phase))
        x1, y1 = -a1 * r_f * np.cos(phase), -a1 * r_f * np.sin(phase)
        x3, y3 =  a3 * r_f * np.cos(phase),  a3 * r_f * np.sin(phase)
        
        h_star1.set_data([x1/AU], [y1/AU]); h_star1.set_3d_properties([0])
        h_star3.set_data([x3/AU], [y3/AU]); h_star3.set_3d_properties([0])
        
        ox, oy, oz = get_orbit_path(y[0,idx], y[1,idx], y[2,idx], y[3,idx])
        h_orbit.set_data((ox+x1)/AU, (oy+y1)/AU); h_orbit.set_3d_properties(oz/AU)
        
        return h_star1, h_star3, h_orbit

    # In Python, we usually just show the final plot or save an animation
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()