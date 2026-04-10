#!/usr/bin/env python3
"""
TurtleBot4 System Identification — Data Analysis & Visualization
================================================================
Loads a pickle file produced by turtlebot_sysid.py and generates
mode-specific plots and animations.

Usage:
  python3 turtlebot_sysid_plot.py sysid_lidar.pkl    [--save]
  python3 turtlebot_sysid_plot.py sysid_velocity.pkl [--save]
  python3 turtlebot_sysid_plot.py sysid_latency.pkl  [--save]

--save  Write figures / animation to files instead of displaying.
"""
import sys
import argparse
import os
import pickle
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def monotonic_to_rel(wall_times, t0=None):
    arr = np.asarray(wall_times)
    if t0 is None:
        t0 = arr[0]
    return arr - t0


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — LiDAR  (Section 1)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_lidar(data: dict, save: bool):
    scans = data['scans']
    n_scans = len(scans)
    n_rays  = len(scans[0]['ranges'])

    # ── build range matrix [n_scans × n_rays] ──────────────────────────────
    R = np.stack([s['ranges'] for s in scans], axis=0)          # float32
    bad_mask = ~np.isfinite(R)                                   # NaN or inf → True

    # ── Per-ray statistics ─────────────────────────────────────────────────
    R_valid = np.where(bad_mask, np.nan, R.astype(np.float64))
    ray_mean = np.nanmean(R_valid, axis=0)
    ray_std  = np.nanstd(R_valid,  axis=0)
    dropout_rate = bad_mask.mean(axis=0)                         # fraction bad per ray

    # ── Markov dropout model: p_start, p_continue (per-ray) ───────────────
    # p_start   = P(bad at t+1 | good at t)
    # p_continue = P(bad at t+1 | bad  at t)
    b = bad_mask.astype(np.int8)                                 # 0=good, 1=bad
    prev = b[:-1]   # [n_scans-1, n_rays]
    curr = b[1:]    # [n_scans-1, n_rays]

    GG = ((prev == 0) & (curr == 0)).sum(axis=0).astype(float)
    GB = ((prev == 0) & (curr == 1)).sum(axis=0).astype(float)
    BG = ((prev == 1) & (curr == 0)).sum(axis=0).astype(float)
    BB = ((prev == 1) & (curr == 1)).sum(axis=0).astype(float)

    p_start    = np.where((GG + GB) > 0, GB / (GG + GB), 0.0)
    p_continue = np.where((BG + BB) > 0, BB / (BG + BB), 0.0)

    # Global scalar estimates (used in annotation)
    p_start_global    = GB.sum() / (GG.sum() + GB.sum() + 1e-12)
    p_continue_global = BB.sum() / (BG.sum() + BB.sum() + 1e-12)

    # ── Scan geometry ──────────────────────────────────────────────────────
    meta      = scans[0]
    angles    = meta['angle_min'] + np.arange(n_rays) * meta['angle_increment']
    wall_t    = monotonic_to_rel([s['wall_time'] for s in scans])

    print(f"[LiDAR] {n_scans} scans × {n_rays} rays")
    print(f"  Global dropout rate    : {bad_mask.mean() * 100:.2f} %")
    print(f"  p_start  (global mean) : {p_start_global:.4f}")
    print(f"  p_continue (global mean): {p_continue_global:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Figure 1 — Per-ray statistics
    # ══════════════════════════════════════════════════════════════════════
    fig1, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig1.suptitle("LiDAR Per-Ray Statistics", fontsize=13, fontweight='bold')
    ang_deg = np.degrees(angles)

    axes[0].plot(ang_deg, ray_mean, lw=0.8, color='steelblue')
    axes[0].fill_between(ang_deg, ray_mean - ray_std, ray_mean + ray_std,
                         alpha=0.25, color='steelblue', label='±1 σ')
    axes[0].set_ylabel('Range [m]')
    axes[0].set_title('Per-ray mean ± std')
    axes[0].legend(fontsize=8)

    axes[1].plot(ang_deg, ray_std, lw=0.8, color='darkorange')
    axes[1].set_ylabel('Std [m]')
    axes[1].set_title('Per-ray standard deviation')

    axes[2].plot(ang_deg, dropout_rate * 100, lw=0.8, color='firebrick')
    axes[2].set_ylabel('Dropout [%]')
    axes[2].set_title('Per-ray NaN/Inf dropout rate')

    axes[3].plot(ang_deg, p_start,    lw=0.8, color='purple',  label='p_start')
    axes[3].plot(ang_deg, p_continue, lw=0.8, color='crimson', label='p_continue')
    axes[3].set_ylabel('Probability')
    axes[3].set_xlabel('Angle [°]')
    axes[3].set_title('Markov dropout model  —  p_start & p_continue per ray')
    axes[3].legend(fontsize=8)
    axes[3].text(0.01, 0.92,
                 f"Global:  p_start={p_start_global:.4f}   p_continue={p_continue_global:.4f}",
                 transform=axes[3].transAxes, fontsize=8, color='black',
                 bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray', alpha=0.8))

    for ax in axes:
        ax.grid(True, lw=0.3, alpha=0.5)
    fig1.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    # Figure 2 — Scan image (range-time heatmap)
    # ══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    vmin, vmax = np.nanpercentile(R_valid, 2), np.nanpercentile(R_valid, 98)
    im = ax2.imshow(R_valid, aspect='auto', origin='lower',
                    extent=[ang_deg[0], ang_deg[-1], wall_t[0], wall_t[-1]],
                    cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, ax=ax2, label='Range [m]')
    ax2.set_xlabel('Angle [°]')
    ax2.set_ylabel('Time [s]')
    ax2.set_title('LiDAR Range Image  (scan × ray,  NaN = masked)')
    fig2.tight_layout()

    # ══════════════════════════════════════════════════════════════════════
    # Animation — polar LiDAR scans
    # ══════════════════════════════════════════════════════════════════════
    # Subsample to ~100 frames maximum so it stays responsive
    stride = max(1, n_scans // 100)
    frame_idx = np.arange(0, n_scans, stride)

    fig_anim = plt.figure(figsize=(7, 7))
    ax_pol = fig_anim.add_subplot(111, polar=True)
    ax_pol.set_title('LiDAR Scan Animation', pad=14)
    ax_pol.set_ylim(0, meta['range_max'])
    ax_pol.set_theta_zero_location('N')
    ax_pol.set_theta_direction(-1)

    # Static: mean scan in the background
    mean_valid = np.where(np.isfinite(ray_mean), ray_mean, 0.0)
    ax_pol.fill(angles, mean_valid, alpha=0.12, color='steelblue')
    ax_pol.plot(angles, mean_valid, lw=0.6, color='steelblue', alpha=0.4, label='mean')

    line_live, = ax_pol.plot([], [], lw=0.8, color='firebrick', alpha=0.85)
    time_text  = ax_pol.text(0.01, 1.02, '', transform=ax_pol.transAxes, fontsize=9)
    ax_pol.legend(loc='upper right', fontsize=8)

    def _anim_init():
        line_live.set_data([], [])
        return line_live, time_text

    def _anim_update(fi):
        i   = frame_idx[fi]
        r   = np.where(np.isfinite(R[i]), R[i], 0.0)
        line_live.set_data(angles, r)
        time_text.set_text(f"t = {wall_t[i]:.2f} s  (scan {i+1}/{n_scans})")
        return line_live, time_text

    anim = FuncAnimation(fig_anim, _anim_update, init_func=_anim_init,
                         frames=len(frame_idx), interval=60, blit=True)

    if save:
        out_stats = 'lidar_stats.png'
        out_image = 'lidar_range_image.png'
        out_anim  = 'lidar_animation.gif'
        fig1.savefig(out_stats, dpi=150)
        fig2.savefig(out_image, dpi=150)
        print(f"  Saved {out_stats}, {out_image}")
        writer = PillowWriter(fps=15)
        anim.save(out_anim, writer=writer, dpi=100)
        print(f"  Saved {out_anim}")
        plt.close('all')
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Velocity Step-Response  (Section 3)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_velocity(data: dict, save: bool):
    odom    = data['odom_buffer']
    trials  = data['trials']
    v_cmd   = data['v_target']
    n_steps = data['n_steps']

    # ── Arrays from odom buffer ────────────────────────────────────────────
    wall  = np.array([o['wall_time'] for o in odom])
    vx    = np.array([o['vx']        for o in odom])
    phase = [o['phase']    for o in odom]
    sidx  = np.array([o['step_idx']  for o in odom])

    # ── Remove odom spike outliers (e.g. first-sample encoder glitch) ──────
    # Keep only samples where |vx| ≤ 2× v_cmd + 0.1 m/s
    outlier_mask = np.abs(vx) <= 2.0 * v_cmd + 0.1
    n_removed = (~outlier_mask).sum()
    if n_removed:
        print(f"  [velocity] Removed {n_removed} outlier sample(s) "
              f"(|vx| > {2.0*v_cmd+0.1:.2f} m/s, max was {np.abs(vx).max():.2f} m/s)")
    wall  = wall[outlier_mask]
    vx    = vx[outlier_mask]
    phase = [p for p, k in zip(phase, outlier_mask) if k]
    sidx  = sidx[outlier_mask]

    t0   = wall[0]
    t_rel = wall - t0

    # ── Build piecewise reference command signal ───────────────────────────
    cmd_times  = np.array([tr['cmd_wall_time'] for tr in trials]) - t0
    cmd_signal = np.zeros_like(t_rel)
    for i, tr in enumerate(trials):
        ct = tr['cmd_wall_time'] - t0
        # Command is v_cmd from ct until the next rest (hold_time after command)
        hold = data.get('hold_time', 4.0)
        mask = (t_rel >= ct) & (t_rel < ct + hold)
        cmd_signal[mask] = v_cmd

    # ── Figure 1: full timeline ─────────────────────────────────────────────
    fig1, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig1.suptitle("Velocity Step-Response  —  Full Timeline", fontsize=13, fontweight='bold')

    axes[0].step(t_rel, cmd_signal, where='post', color='black', lw=1.2, label='cmd v')
    axes[0].plot(t_rel, vx, lw=0.7, color='steelblue', alpha=0.85, label='measured vx')
    for ct in cmd_times:
        axes[0].axvline(ct, color='green', lw=0.6, ls='--', alpha=0.5)
    axes[0].set_ylabel('v [m/s]')
    axes[0].legend(fontsize=8)
    axes[0].set_title('Command vs measured linear velocity')
    axes[0].grid(True, lw=0.3, alpha=0.5)

    vy = np.array([o['vy'] for o in odom])[outlier_mask]
    wz = np.array([o['wz'] for o in odom])[outlier_mask]
    axes[1].plot(t_rel, vy, lw=0.7, color='darkorange', label='vy (lateral)')
    axes[1].plot(t_rel, wz, lw=0.7, color='purple',     label='wz (yaw rate)')
    axes[1].set_ylabel('m/s  /  rad/s')
    axes[1].legend(fontsize=8)
    axes[1].set_title('Lateral velocity & yaw rate (should be ≈ 0 during straight steps)')
    axes[1].grid(True, lw=0.3, alpha=0.5)

    px = np.array([o['px'] for o in odom])[outlier_mask]
    py = np.array([o['py'] for o in odom])[outlier_mask]
    axes[2].plot(t_rel, px, lw=0.7, color='steelblue', label='x')
    axes[2].plot(t_rel, py, lw=0.7, color='firebrick', label='y')
    axes[2].set_ylabel('Position [m]')
    axes[2].set_xlabel('Time [s]')
    axes[2].legend(fontsize=8)
    axes[2].set_title('Odometry position')
    axes[2].grid(True, lw=0.3, alpha=0.5)

    fig1.tight_layout()

    # ── Figure 2: per-step aligned overlay ─────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.suptitle("Velocity Steps — Aligned Overlay", fontsize=13, fontweight='bold')
    hold = data.get('hold_time', 4.0)
    rest = data.get('rest_time', 2.0)
    window = rest + hold + 1.0   # seconds to show around each step

    colors = plt.cm.tab10(np.linspace(0, 1, n_steps))
    for i, tr in enumerate(trials):
        ct = tr['cmd_wall_time']
        t_start = ct - rest
        t_end   = ct + hold + 1.0
        mask = (wall >= t_start) & (wall <= t_end)
        if not mask.any():
            continue
        t_loc = wall[mask] - ct
        ax.plot(t_loc, vx[mask], lw=0.9, color=colors[i], alpha=0.8, label=f'step {i+1}')

    ax.axvline(0, color='black', lw=1.2, ls='--', label='step onset')
    ax.axhline(v_cmd, color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel('Time relative to step [s]')
    ax.set_ylabel('vx [m/s]')
    ax.set_title(f'All {n_steps} step responses aligned at command time')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, lw=0.3, alpha=0.5)
    fig2.tight_layout()

    # ── Figure 3: try first-order fit for each step ─────────────────────────
    # vx(t) ≈ v_cmd * (1 − exp(−t/τ))  for t ≥ 0
    fig3, axes3 = plt.subplots(
        math.ceil(n_steps / 3), min(n_steps, 3),
        figsize=(14, 4 * math.ceil(n_steps / 3)), sharey=True)
    axes3_flat = np.asarray(axes3).ravel() if n_steps > 1 else [axes3]
    fig3.suptitle("Per-Step First-Order Fit  (vx ≈ v_cmd·[1−e^(−t/τ)])",
                  fontsize=12, fontweight='bold')

    taus = []
    for i, tr in enumerate(trials):
        ax3 = axes3_flat[i]
        ct  = tr['cmd_wall_time']
        # Use samples from cmd time onwards up to cmd+hold
        mask = (wall >= ct) & (wall <= ct + hold)
        if not mask.any():
            ax3.set_visible(False)
            continue
        t_loc = wall[mask] - ct
        v_meas = vx[mask]

        # Nonlinear least squares for τ
        # Only use the rising portion (up to peak) to avoid post-peak decay
        # confusing the fit.  Normalise by v_peak rather than v_cmd.
        tau_est = None
        try:
            from scipy.optimize import curve_fit
            peak_idx = int(np.argmax(v_meas))
            v_peak   = v_meas[peak_idx]
            if peak_idx > 2 and v_peak > 0.05:
                t_rise  = t_loc[:peak_idx + 1]
                v_rise  = v_meas[:peak_idx + 1]
                def first_order(t, tau):
                    return v_peak * (1.0 - np.exp(-t / tau))
                p0, _ = curve_fit(first_order, t_rise, v_rise, p0=[0.3],
                                   bounds=(1e-4, 5.0), maxfev=2000)
                tau_est = p0[0]
                # Reject if τ hit the upper bound (poor fit)
                if tau_est < 4.99:
                    taus.append(tau_est)
                    t_fit = np.linspace(0, t_loc[-1], 300)
                    ax3.plot(t_fit, first_order(t_fit, tau_est),
                             lw=1.5, color='firebrick', ls='--',
                             label=f'fit τ={tau_est*1000:.0f} ms  (v_pk={v_peak:.2f})')
                else:
                    tau_est = None
        except Exception:
            pass

        ax3.plot(t_loc, v_meas, lw=0.8, color='steelblue', label='measured')
        ax3.axhline(v_cmd, color='gray', lw=0.7, ls=':', alpha=0.6)
        ax3.set_title(f'Step {i+1}' + (f'  τ={tau_est*1000:.0f} ms' if tau_est else ''))
        ax3.set_xlabel('t [s]')
        ax3.set_ylabel('vx [m/s]')
        ax3.legend(fontsize=7)
        ax3.grid(True, lw=0.3, alpha=0.5)

    # Hide unused subplots
    for j in range(n_steps, len(axes3_flat)):
        axes3_flat[j].set_visible(False)

    if taus:
        print(f"[Velocity] Estimated τ per step: {[f'{t*1000:.1f} ms' for t in taus]}")
        print(f"           Mean τ = {np.mean(taus)*1000:.1f} ms ± {np.std(taus)*1000:.1f} ms")

    fig3.tight_layout()

    if save:
        fig1.savefig('velocity_timeline.png', dpi=150)
        fig2.savefig('velocity_overlay.png',  dpi=150)
        fig3.savefig('velocity_fit.png',      dpi=150)
        print("  Saved velocity_timeline.png, velocity_overlay.png, velocity_fit.png")
        plt.close('all')
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Latency + Drift  (Section 4)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_latency(data: dict, save: bool):
    odom       = data['odom_buffer']
    lat_trials = data['lat_trials']
    v_cruise   = data['v_cruise']
    vx_thresh  = data.get('vx_threshold', 0.02)

    wall = np.array([o['wall_time'] for o in odom])
    vx   = np.array([o['vx']        for o in odom])
    vy   = np.array([o['vy']        for o in odom])
    px   = np.array([o['px']        for o in odom])
    py   = np.array([o['py']        for o in odom])
    ph   = [o['phase'] for o in odom]

    # ── Figure 1: per-trial velocity around step ───────────────────────────
    n_trials = len(lat_trials)
    ncols = min(n_trials, 4)
    nrows = math.ceil(n_trials / ncols)
    fig1, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharey=True)
    axes_flat  = np.asarray(axes).ravel() if n_trials > 1 else [axes]
    fig1.suptitle(f"Actuator Latency Trials  (v_cmd = {v_cruise} m/s,  threshold = {vx_thresh} m/s)",
                  fontsize=12, fontweight='bold')

    win_pre, win_post = 0.2, 0.8   # seconds before/after command to show
    delays = []

    for i, tr in enumerate(lat_trials):
        ax = axes_flat[i]
        ct = tr['cmd_wall_time']
        mask = (wall >= ct - win_pre) & (wall <= ct + win_post)
        if not mask.any():
            ax.set_visible(False)
            continue
        t_loc  = wall[mask] - ct
        vx_loc = vx[mask]

        ax.axvline(0, color='black', lw=1.2, ls='--', label='cmd')
        ax.axhline(vx_thresh, color='gray', lw=0.7, ls=':', alpha=0.7, label=f'thresh {vx_thresh}')
        ax.plot(t_loc, vx_loc, lw=0.9, color='steelblue')

        delay = tr.get('detected_delay_s')
        if delay is not None:
            ax.axvline(delay, color='firebrick', lw=1.2, ls='-.', label=f'Δt={delay*1000:.1f} ms')
            delays.append(delay)
        ax.set_title(f'Trial {i+1}' + (f'  Δt={delay*1000:.1f} ms' if delay else '  (no response)'))
        ax.set_xlabel('t [s]')
        ax.set_ylabel('vx [m/s]')
        ax.legend(fontsize=7)
        ax.grid(True, lw=0.3, alpha=0.4)

    for j in range(n_trials, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig1.tight_layout()

    # ── Figure 2: delay summary ────────────────────────────────────────────
    if delays:
        delays_ms = np.array(delays) * 1000
        print(f"[Latency] {len(delays)}/{n_trials} trials with detected response")
        print(f"  Mean Δt = {delays_ms.mean():.1f} ms ± {delays_ms.std():.1f} ms")
        print(f"  Min/Max = {delays_ms.min():.1f} / {delays_ms.max():.1f} ms")

        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
        fig2.suptitle("Latency Summary", fontsize=12, fontweight='bold')

        axes2[0].bar(np.arange(1, len(delays_ms) + 1), delays_ms,
                     color='steelblue', edgecolor='black', lw=0.5)
        axes2[0].axhline(delays_ms.mean(), color='firebrick', lw=1.2, ls='--',
                         label=f'mean = {delays_ms.mean():.1f} ms')
        axes2[0].set_xlabel('Trial')
        axes2[0].set_ylabel('Delay [ms]')
        axes2[0].set_title('Per-trial detected delay')
        axes2[0].legend(fontsize=9)
        axes2[0].grid(True, lw=0.3, axis='y', alpha=0.5)

        axes2[1].boxplot(delays_ms, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightsteelblue'))
        axes2[1].scatter(np.ones(len(delays_ms)) + 0.05 * np.random.randn(len(delays_ms)),
                         delays_ms, alpha=0.6, color='steelblue', zorder=5)
        axes2[1].set_ylabel('Delay [ms]')
        axes2[1].set_xticks([1])
        axes2[1].set_xticklabels([f'n={len(delays_ms)}'])
        axes2[1].set_title(f'Distribution  μ={delays_ms.mean():.1f}  σ={delays_ms.std():.1f} ms')
        axes2[1].grid(True, lw=0.3, axis='y', alpha=0.5)
        fig2.tight_layout()
    else:
        print("[Latency] No delays were detected — check VX_THRESHOLD.")
        fig2 = None

    # ── Figure 3: drift trajectory ─────────────────────────────────────────
    drift_mask = np.array([p == 'drift' for p in ph])
    if drift_mask.any():
        px_d = px[drift_mask]
        py_d = py[drift_mask]
        t_d  = wall[drift_mask] - wall[drift_mask][0]

        # Straight-line reference (along x from start position)
        x0, y0 = px_d[0], py_d[0]
        lateral = py_d - y0          # deviation from starting y

        print(f"[Drift] Distance covered: {math.hypot(px_d[-1]-x0, py_d[-1]-y0):.3f} m")
        print(f"  Final lateral deviation: {lateral[-1]*100:.1f} cm")
        print(f"  Max |lateral| deviation: {np.abs(lateral).max()*100:.1f} cm")

        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
        fig3.suptitle("Differential Drive Drift Analysis", fontsize=12, fontweight='bold')

        # XY trajectory
        axes3[0].plot(px_d - x0, py_d - y0, lw=1.0, color='steelblue', label='trajectory')
        axes3[0].plot([0, px_d[-1] - x0], [0, 0], lw=1.0, color='black', ls='--', label='ideal (straight)')
        axes3[0].scatter([0], [0], color='green', s=50, zorder=5, label='start')
        axes3[0].scatter([px_d[-1] - x0], [py_d[-1] - y0], color='red', s=50, zorder=5, label='end')
        axes3[0].set_xlabel('x [m]')
        axes3[0].set_ylabel('y [m]')
        axes3[0].set_title('XY Trajectory (odom frame)')
        axes3[0].legend(fontsize=8)
        axes3[0].set_aspect('equal')
        axes3[0].grid(True, lw=0.3, alpha=0.5)

        # Lateral deviation over distance
        dist_d = np.sqrt((px_d - x0)**2 + (py_d - y0)**2)
        axes3[1].plot(dist_d, lateral * 100, lw=1.0, color='firebrick')
        axes3[1].axhline(0, color='black', lw=0.7, ls='--')
        axes3[1].fill_between(dist_d, lateral * 100, 0, alpha=0.15, color='firebrick')
        axes3[1].set_xlabel('Distance travelled [m]')
        axes3[1].set_ylabel('Lateral deviation [cm]')
        axes3[1].set_title(f'Lateral deviation  (final = {lateral[-1]*100:.1f} cm)')
        axes3[1].grid(True, lw=0.3, alpha=0.5)
        fig3.tight_layout()
    else:
        print("[Drift] No drift phase found in odom buffer.")
        fig3 = None

    if save:
        fig1.savefig('latency_trials.png',  dpi=150)
        print("  Saved latency_trials.png")
        if fig2:
            fig2.savefig('latency_summary.png', dpi=150)
            print("  Saved latency_summary.png")
        if fig3:
            fig3.savefig('latency_drift.png', dpi=150)
            print("  Saved latency_drift.png")
        plt.close('all')
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 — Staircase velocity tracking  (Section 5)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_staircase(data: dict, save: bool):
    odom     = data['odom_buffer']
    sequence = data['sequence']   # list of (v_cmd, w_cmd, hold_s)

    wall  = np.array([o['wall_time'] for o in odom])
    vx    = np.array([o['vx']        for o in odom])
    wz    = np.array([o['wz']        for o in odom])
    v_cmd = np.array([o['v_cmd']     for o in odom])
    w_cmd = np.array([o.get('w_cmd', 0.0) for o in odom])  # backward compat
    sidx  = np.array([o['step_idx']  for o in odom])

    t_rel = wall - wall[0]

    # ── Outlier removal ──────────────────────────────────────────────────────
    max_v = max(abs(v) for v, _, _ in sequence)
    max_w = max(abs(w) for _, w, _ in sequence)
    thr_v = 2.0 * max_v + 0.1
    thr_w = 2.0 * max_w + 0.5
    ok = (np.abs(vx) <= thr_v) & (np.abs(wz) <= thr_w)
    n_rm = (~ok).sum()
    if n_rm:
        print(f"  [staircase] Removed {n_rm} outlier(s)")
    t_rel = t_rel[ok]
    vx    = vx[ok];   wz    = wz[ok]
    v_cmd = v_cmd[ok]; w_cmd = w_cmd[ok]
    sidx  = sidx[ok]

    # ── Determine which channels are actually exercised ──────────────────────
    has_lin = any(abs(v) > 1e-6 for v, _, _ in sequence)
    has_ang = any(abs(w) > 1e-6 for _, w, _ in sequence)
    n_panels = int(has_lin) + int(has_ang)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle("Staircase Tracking  —  Reference vs Measured",
                 fontsize=13, fontweight='bold')

    def _plot_channel(ax, t, cmd, meas, label_cmd, label_meas, ylabel, color):
        ax.step(t, cmd, where='post', color='black', lw=1.4,
                label=label_cmd, zorder=3)
        ax.plot(t, meas, color=color, lw=0.8, alpha=0.9, label=label_meas)
        ax.fill_between(t, cmd, meas, alpha=0.15, color='firebrick',
                        label='tracking error')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, lw=0.3, alpha=0.5)

    def _annotate_levels(ax, sequence, field):
        """Draw vertical separators and level labels at the top of the axis."""
        t_cursor = 0.0
        ylo, yhi = ax.get_ylim()
        for v, w, hold in sequence:
            val = v if field == 'v' else w
            ax.axvline(t_cursor, color='gray', lw=0.5, ls='--', alpha=0.45)
            ax.text(t_cursor + hold / 2, yhi - (yhi - ylo) * 0.05,
                    f'{val:.2f}', ha='center', va='top', fontsize=7, color='dimgray')
            t_cursor += hold

    panel = 0
    if has_lin:
        _plot_channel(axes[panel], t_rel, v_cmd, vx,
                      'v_cmd', 'vx measured', 'v [m/s]', 'steelblue')
        _annotate_levels(axes[panel], sequence, 'v')
        panel += 1
    if has_ang:
        _plot_channel(axes[panel], t_rel, w_cmd, wz,
                      'ω_cmd', 'ωz measured', 'ω [rad/s]', 'darkorange')
        _annotate_levels(axes[panel], sequence, 'w')
        panel += 1

    axes[-1].set_xlabel('Time [s]')

    # ── Per-level steady-state table ─────────────────────────────────────────
    print(f"[Staircase] {len(sequence)} levels  |  odom samples: {ok.sum()}")
    if has_lin:
        print(f"\n  Linear velocity:")
        print(f"  {'v_cmd':>8}  {'v_mean':>8}  {'v_std':>8}  {'error':>8}")
        for i, (v, w, hold) in enumerate(sequence):
            mask_i = sidx == i
            if not mask_i.any():
                continue
            t_i = t_rel[mask_i]
            steady = t_i >= (t_i[0] + hold * 0.5)
            meas = vx[mask_i]
            mu, sig = (meas[steady].mean(), meas[steady].std()) if steady.any() \
                      else (meas.mean(), meas.std())
            print(f"  {v:>8.3f}  {mu:>8.3f}  {sig:>8.3f}  {mu-v:>+8.3f}")
    if has_ang:
        print(f"\n  Angular velocity:")
        print(f"  {'w_cmd':>8}  {'w_mean':>8}  {'w_std':>8}  {'error':>8}")
        for i, (v, w, hold) in enumerate(sequence):
            mask_i = sidx == i
            if not mask_i.any():
                continue
            t_i = t_rel[mask_i]
            steady = t_i >= (t_i[0] + hold * 0.5)
            meas = wz[mask_i]
            mu, sig = (meas[steady].mean(), meas[steady].std()) if steady.any() \
                      else (meas.mean(), meas.std())
            print(f"  {w:>8.3f}  {mu:>8.3f}  {sig:>8.3f}  {mu-w:>+8.3f}")

    fig.tight_layout()

    if save:
        fig.savefig('staircase_tracking.png', dpi=150)
        print("  Saved staircase_tracking.png")
        plt.close('all')
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='TurtleBot4 SysId — Data Visualisation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('pickle_file', help='Path to .pkl file produced by turtlebot_sysid.py')
    parser.add_argument('--save', action='store_true',
                        help='Save figures/animation to files instead of displaying')
    args = parser.parse_args()

    path = args.pickle_file
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    print(f"Loading {path} …")
    data = load(path)
    mode = data.get('mode', '')
    print(f"Mode: {mode}")

    if args.save:
        matplotlib.use('Agg')

    if mode == 'lidar_sysid':
        analyse_lidar(data, args.save)
    elif mode == 'velocity_sysid':
        analyse_velocity(data, args.save)
    elif mode == 'latency_drift_sysid':
        analyse_latency(data, args.save)
    elif mode == 'staircase_sysid':
        analyse_staircase(data, args.save)
    else:
        print(f"Unknown mode '{mode}' in pickle file.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
