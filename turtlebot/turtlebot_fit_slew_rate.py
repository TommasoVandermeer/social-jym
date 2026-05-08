#!/usr/bin/env python3
import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from matplotlib import rc, rcParams

font = {
    'weight' : 'regular',
    'size'   : 23
}
rc('font', **font)
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

def fit_slew_rate_with_delay(pkl_filename, axis='linear'):
    with open(pkl_filename, 'rb') as f:
        data = pickle.load(f)
        
    cmd_data = data['cmd']
    odom_data = data['odom']
    
    cmd_t = np.array([d['timestamp'] for d in cmd_data])
    odom_t = np.array([d['timestamp'] for d in odom_data])
    
    if axis == 'linear':
        cmd_v = np.array([d['v'] for d in cmd_data])
        odom_v = np.array([d['v'] for d in odom_data])
    else:
        cmd_v = np.array([d['w'] for d in cmd_data])
        odom_v = np.array([d['w'] for d in odom_data])

    t0 = min(cmd_t[0], odom_t[0])
    cmd_t -= t0
    odom_t -= t0

    # 1. Nuovo Modello: Slew Rate Limiter
    def simulate_slew_rate(a_max, t, u, v_init):
        v_sim = np.zeros_like(t)
        v_sim[0] = v_init
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            if dt <= 0:
                v_sim[i] = v_sim[i-1]
                continue
                
            error = u[i-1] - v_sim[i-1]
            max_delta_v = a_max * dt
            
            # Applica il limite Bang-Bang dell'accelerazione
            actual_delta_v = np.clip(error, -max_delta_v, max_delta_v)
            v_sim[i] = v_sim[i-1] + actual_delta_v
            
        return v_sim

    # 2. Ottimizzatore modificato per cercare a_max
    def optimize_amax_for_given_delay(delay):
        shifted_cmd_t = cmd_t + delay
        
        cmd_interp_func = interp1d(shifted_cmd_t, cmd_v, kind='previous', bounds_error=False, fill_value=(0.0, cmd_v[-1]))
        u_sync_delayed = cmd_interp_func(odom_t)
        
        def loss_function(params):
            a_max = params[0] 
            # Evita valori negativi o irrealisticamente alti (es. > 20 m/s^2 o rad/s^2)
            if a_max <= 0.01 or a_max > 20.0:
                return 1e6
            v_sim = simulate_slew_rate(a_max, odom_t, u_sync_delayed, odom_v[0])
            return np.mean((v_sim - odom_v)**2)
            
        # Fornisco un punto di partenza intuitivo a seconda dell'asse per velocizzare l'ottimizzazione
        initial_guess = [1.0] if axis == 'linear' else [5.0]
        result = minimize(loss_function, initial_guess, method='Nelder-Mead')
        return result.x[0], result.fun, u_sync_delayed

    print(f"Grid Search on Delay T_d for axis {axis}...")
    
    candidate_delays = np.arange(0.0, 0.5, 0.01)
    best_error = float('inf')
    best_amax = None
    best_delay = 0.0
    best_u_sync = None
    
    for delay in candidate_delays:
        a_max, error, u_sync_delayed = optimize_amax_for_given_delay(delay)
        if error < best_error:
            best_error = error
            best_amax = a_max
            best_delay = delay
            best_u_sync = u_sync_delayed

    amax_opt = best_amax

    print("\n" + "="*45)
    print("🎯 IDENTIFIED PARAMETERS:")
    print(f"   Model         : Slew Rate Limiter (Bang-Bang)")
    print(f"   Delay (T_d)   : {best_delay:.3f} s")
    acc_unit = "m/s^2" if axis == 'linear' else "rad/s^2"
    print(f"   Max Accel (a) : {amax_opt:.4f} {acc_unit}")
    print("="*45 + "\n")

    v_sim_opt = simulate_slew_rate(amax_opt, odom_t, best_u_sync, odom_v[0])
    
    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15)
    
    cmd_interp_original = interp1d(cmd_t, cmd_v, kind='previous', bounds_error=False, fill_value=(cmd_v[0], cmd_v[-1]))
    plt.step(odom_t, cmd_interp_original(odom_t), label='Ref', where='post', color='green', alpha=0.7, linestyle='--')
    
    plt.step(odom_t, best_u_sync, label=f'Delayed ref ({best_delay:.2f}s)', where='post', color='gray', alpha=0.7, linestyle=':')
    
    plt.plot(odom_t, odom_v, label='State', color='blue', linewidth=2)
    plt.plot(odom_t, v_sim_opt, label=f'Model (a_max={amax_opt:.2f}, Td={best_delay:.2f}s)', color='red', linestyle='-.', linewidth=2)
    
    ylabel = 'Linear Vel [m/s]' if axis == 'linear' else 'Angular Vel [rad/s]'
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    if axis == 'linear':
        plt.title(f'Slew Rate system identification ({axis})    ' + r'$ \dot{v} = a_{max} \cdot \mathrm{sgn}(u(t - T_d) - v) $')
    elif axis == 'angular':
        plt.title(f'Slew Rate system identification ({axis})    ' + r'$ \dot{\omega} = \alpha_{max} \cdot \mathrm{sgn}(u(t - T_d) - \omega) $')
    plt.legend(fontsize=16)
    plt.grid(True)
    
    eps_filename = f'slew_rate_fit_{axis}.eps'
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), eps_filename), format='eps')
    print(f"Grafico salvato in: {eps_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Identify slew rate system parameters for the robot velocity dynamics")
    parser.add_argument('-a', '--axis', type=str, choices=['linear', 'angular'], default='linear', help="Analyze 'linear' or 'angular' velocity dynamics.")
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = 'square_wave_data' + f"_{args.axis}" + '.pkl'

    if os.path.isabs(file):
        full_path = file
    else:
        full_path = os.path.join(script_dir, file)
    if not os.path.exists(full_path):
        print(f"ERROR: file {full_path} does not exist.")
        return

    fit_slew_rate_with_delay(full_path, axis=args.axis)

if __name__ == '__main__':
    main()