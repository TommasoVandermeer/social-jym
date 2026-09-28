#!/usr/bin/env python3
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from matplotlib import rc, rcParams

font = {'weight': 'regular', 'size': 18}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
TRACK_WIDTH = 0.235

def fit_coupled_slew_rate_with_delay(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
    cmd_data = data['cmd']
    odom_data = data['odom']
    cmd_t = np.array([d['timestamp'] for d in cmd_data])
    odom_t = np.array([d['timestamp'] for d in odom_data])
    cmd_v = np.array([d['v'] for d in cmd_data])
    cmd_w = np.array([d['w'] for d in cmd_data])
    odom_v = np.array([d['v'] for d in odom_data])
    odom_w = np.array([d['w'] for d in odom_data])
    t0 = min(cmd_t[0], odom_t[0])
    cmd_t -= t0
    odom_t -= t0

    def simulate_coupled_slew_rate(a_max_wheel, t, u_v, u_w, v_init, w_init):
        v_sim = np.zeros_like(t)
        w_sim = np.zeros_like(t)
        v_sim[0] = v_init
        w_sim[0] = w_init
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            if dt <= 0:
                v_sim[i] = v_sim[i-1]
                w_sim[i] = w_sim[i-1]
                continue
            a_req = (u_v[i-1] - v_sim[i-1]) / dt
            alpha_req = (u_w[i-1] - w_sim[i-1]) / dt
            effort = abs(a_req) + (TRACK_WIDTH / 2.0) * abs(alpha_req)
            scale = a_max_wheel / effort if effort > a_max_wheel and effort > 1e-6 else 1.0
            v_sim[i] = v_sim[i-1] + (a_req * scale) * dt
            w_sim[i] = w_sim[i-1] + (alpha_req * scale) * dt
        return v_sim, w_sim

    def optimize_amax_for_given_delay(delay):
        shifted_cmd_t = cmd_t + delay
        interp_v = interp1d(shifted_cmd_t, cmd_v, kind='previous', bounds_error=False, fill_value=(0.0, cmd_v[-1]))
        interp_w = interp1d(shifted_cmd_t, cmd_w, kind='previous', bounds_error=False, fill_value=(0.0, cmd_w[-1]))
        u_v_delayed = interp_v(odom_t)
        u_w_delayed = interp_w(odom_t)
        def loss_function(params):
            a_max_wheel = params[0] 
            if a_max_wheel <= 0.01 or a_max_wheel > 10.0:
                return 1e6
            v_sim, w_sim = simulate_coupled_slew_rate(a_max_wheel, odom_t, u_v_delayed, u_w_delayed, odom_v[0], odom_w[0])
            mse_v = np.mean((v_sim - odom_v)**2)
            mse_w = np.mean((w_sim - odom_w)**2) * (TRACK_WIDTH / 2.0)**2
            return mse_v + mse_w
        result = minimize(loss_function, [0.9], method='Nelder-Mead')
        return result.x[0], result.fun, u_v_delayed, u_w_delayed

    print("Starting Grid Search on Delay (Td) for COUPLED dynamics...")
    candidate_delays = np.arange(0.0, 0.5, 0.01)
    best_error = float('inf')
    best_amax_wheel, best_delay = None, 0.0
    best_u_v_sync, best_u_w_sync = None, None
    for delay in candidate_delays:
        a_max, error, u_v_del, u_w_del = optimize_amax_for_given_delay(delay)
        if error < best_error:
            best_error = error
            best_amax_wheel = a_max
            best_delay = delay
            best_u_v_sync = u_v_del
            best_u_w_sync = u_w_del

    print("\n" + "="*45)
    print("🎯 IDENTIFIED PARAMETERS (COUPLED SYSTEM):")
    print(f"   Model                         : Dynamic Acceleration Scaling (Rhombus)")
    print(f"   Pure Delay (T_d)              : {best_delay:.3f} s")
    print(f"   SINGLE WHEEL Max Accel (A_max): {best_amax_wheel:.4f} m/s^2")
    print("="*45 + "\n")

    v_sim_opt, w_sim_opt = simulate_coupled_slew_rate(best_amax_wheel, odom_t, best_u_v_sync, best_u_w_sync, odom_v[0], odom_w[0])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.15)
    
    interp_v_orig = interp1d(cmd_t, cmd_v, kind='previous', bounds_error=False, fill_value=(cmd_v[0], cmd_v[-1]))
    interp_w_orig = interp1d(cmd_t, cmd_w, kind='previous', bounds_error=False, fill_value=(cmd_w[0], cmd_w[-1]))
    
    ax1.step(odom_t, interp_v_orig(odom_t), label='Ref', where='post', color='green', alpha=0.7, linestyle='--')
    ax1.step(odom_t, best_u_v_sync, label='Delayed Ref', where='post', color='gray', alpha=0.5, linestyle=':')
    ax1.plot(odom_t, odom_v, label='State', color='blue', linewidth=2)
    ax1.plot(odom_t, v_sim_opt, label='Coupled Model', color='red', linestyle='-.', linewidth=2)
    ax1.set_ylabel('Linear [m/s]')
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(True)
    ax1.set_title(f'Coupled Slew Rate | Wheel $A_{{max}} = {best_amax_wheel:.2f}$ m/s$^2$, $T_d = {best_delay:.2f}$ s')

    ax2.step(odom_t, interp_w_orig(odom_t), label='Ref', where='post', color='green', alpha=0.7, linestyle='--')
    ax2.step(odom_t, best_u_w_sync, label='Delayed Ref', where='post', color='gray', alpha=0.5, linestyle=':')
    ax2.plot(odom_t, odom_w, label='State', color='blue', linewidth=2)
    ax2.plot(odom_t, w_sim_opt, label='Coupled Model', color='red', linestyle='-.', linewidth=2)
    ax2.set_ylabel('Angular [rad/s]')
    ax2.set_xlabel('Time [s]')
    ax2.legend(loc='upper right', fontsize=14)
    ax2.grid(True)

    eps_filename = 'coupled_slew_rate_fit.eps'
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), eps_filename), format='eps')
    print(f"Plot saved in: {eps_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Identify coupled slew rate parameters (Single Wheel Max Accel and Delay)")
    parser.add_argument('-f', '--file', type=str, default='coupled_excitation_data.pkl', help="Pkl file containing cmd and odom data.")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = args.file if os.path.isabs(args.file) else os.path.join(script_dir, args.file)
    if not os.path.exists(full_path):
        print(f"ERROR: file {full_path} does not exist.")
        return
    fit_coupled_slew_rate_with_delay(full_path)

if __name__ == '__main__':
    main()