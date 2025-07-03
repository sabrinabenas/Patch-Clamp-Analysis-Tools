"""
Spike Analysis Tool for Patch Clamp Recordings

This script analyzes ABF (Axon Binary Format) files from patch clamp recordings
to detect and characterize action potentials (spikes), including:
- Spike amplitude
- Afterhyperpolarization (AHP) depth
- Voltage threshold
- Spike width and timing
- Spike count and frequency

Uses eFEL (Electrophys Feature Extraction Library) for feature extraction
combined with custom algorithms for threshold detection.

Author: Sabrina Benas
"""

import efel
import pyabf
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.interpolate import CubicSpline
import matplotlib.cm as cm
import pandas as pd


def find_nearest_value(array, target_value):
    """
    Find the nearest value and its index in an array.
    
    Args:
        array (array-like): Array of values to search
        target_value (float): Target value to find nearest match for
        
    Returns:
        tuple: (nearest_value, index) of the closest match
    """
    array = np.asarray(array)
    idx = (np.abs(array - target_value)).argmin()
    return array[idx], idx


def calculate_second_derivative_windowed(data, window_size=1):
    """
    Calculate second derivative using a windowed approach.
    Used for spike threshold detection.
    
    Args:
        data (list): Input voltage data
        window_size (int): Size of the window for derivative calculation
        
    Returns:
        list: Second derivative values
    """
    data_padded = list(data)
    
    # Add NaN padding based on window size
    for _ in range(window_size):
        data_padded.insert(0, np.nan)
        data_padded.append(np.nan)
    
    second_derivative = []
    for i in range(window_size, len(data_padded) - window_size):
        # Second derivative approximation: f(x+h) + f(x-h) - 2*f(x)
        derivative_val = (data_padded[i + window_size] + 
                         data_padded[i - window_size] - 
                         2 * data_padded[i])
        second_derivative.append(derivative_val)
    
    return second_derivative


def setup_efel_parameters(derivative_threshold=3):
    """
    Configure eFEL parameters for spike detection.
    
    Args:
        derivative_threshold (float): Threshold for spike detection sensitivity
    """
    efel.setDerivativeThreshold(derivative_threshold)
    efel.setDoubleSetting('interp_step', 0.1)
    efel.setDoubleSetting('DownDerivativeThreshold', -100)
    efel.setDoubleSetting('DerivativeWindow', -50)


def extract_spike_features(trace):
    """
    Extract spike features using eFEL library.
    
    Args:
        trace (dict): Trace dictionary with 'T', 'V', 'stim_start', 'stim_end'
        
    Returns:
        dict: Dictionary containing extracted features
    """
    feature_names = [
        'mean_frequency', 'min_AHP_indices', 'AP_begin_time', 
        'AP_begin_voltage', 'AP_begin_width', 'adaptation_index2', 
        'ISI_CV', 'doublet_ISI', 'time_to_first_spike', 'AP_height',
        'AHP_depth_abs', 'AHP_depth_abs_slow', 'AHP_slow_time', 
        'AP_width', 'peak_time'
    ]
    
    return efel.getFeatureValues([trace], feature_names)


def detect_spike_threshold(time_trace, voltage_trace, peak_time, peak_voltage, 
                          points_before_peak=50, spline_window=200):
    """
    Detect spike threshold using second derivative maximum.
    
    Args:
        time_trace (list): Time values in milliseconds
        voltage_trace (list): Voltage values in mV
        peak_time (float): Time of spike peak
        peak_voltage (float): Voltage at spike peak
        points_before_peak (int): Number of points to analyze before peak
        spline_window (int): Window size for spline derivative calculation
        
    Returns:
        tuple: (threshold_voltage, threshold_time)
    """
    # Find indices closest to peak
    _, time_idx = find_nearest_value(time_trace, peak_time)
    _, voltage_idx = find_nearest_value(voltage_trace, peak_voltage)
    
    # Extract mini-traces around the spike
    mini_time = time_trace[time_idx - points_before_peak:time_idx + 10]
    mini_voltage = voltage_trace[voltage_idx - points_before_peak:voltage_idx + 10]
    
    # Create high-resolution spline interpolation
    spline_func = CubicSpline(mini_time, mini_voltage, bc_type='natural')
    time_highres = np.linspace(mini_time[0], mini_time[-1], 1000)
    voltage_highres = spline_func(time_highres)
    
    # Calculate second derivatives
    derivative2_original = calculate_second_derivative_windowed(mini_voltage, w=7)
    derivative2_spline = calculate_second_derivative_windowed(voltage_highres, w=spline_window)
    
    # Find maximum of second derivative (threshold point)
    max_idx_spline = list(derivative2_spline).index(np.nanmax(derivative2_spline))
    
    # Map back to original time and voltage
    threshold_time, _ = find_nearest_value(mini_time, time_highres[max_idx_spline])
    threshold_voltage, _ = find_nearest_value(mini_voltage, voltage_highres[max_idx_spline])
    
    return threshold_voltage, threshold_time


def calculate_spike_width(voltage_trace, time_trace, threshold_voltage, 
                         threshold_time, min_voltage, min_time, peak_time):
    """
    Calculate spike width from threshold to repolarization.
    
    Args:
        voltage_trace (list): Voltage values
        time_trace (list): Time values
        threshold_voltage (float): Spike threshold voltage
        threshold_time (float): Spike threshold time
        min_voltage (float): Minimum voltage (AHP)
        min_time (float): Time of minimum voltage
        peak_time (float): Time of spike peak
        
    Returns:
        tuple: (spike_width, width_to_peak)
    """
    threshold_idx = find_nearest_value(voltage_trace, threshold_voltage)[1]
    
    if min_voltage < threshold_voltage:
        # Find repolarization point after peak
        search_start = threshold_idx + 20
        search_end = threshold_idx + 55
        repol_voltage, _ = find_nearest_value(
            voltage_trace[search_start:search_end], threshold_voltage
        )
        repol_idx = voltage_trace.index(repol_voltage)
        
        spike_width = time_trace[repol_idx] - threshold_time
        width_to_peak = peak_time - threshold_time
        
    else:
        # Alternative calculation when AHP is above threshold
        min_idx = find_nearest_value(voltage_trace, min_voltage)[1]
        search_start = min_idx - 80
        search_end = min_idx - 20
        repol_voltage, _ = find_nearest_value(
            voltage_trace[search_start:search_end], min_voltage
        )
        repol_idx = voltage_trace.index(repol_voltage)
        
        spike_width = min_time - time_trace[repol_idx]
        width_to_peak = peak_time - time_trace[repol_idx]
    
    return spike_width, width_to_peak


def plot_spike_analysis(time_trace, voltage_trace, peak_times, peak_voltages,
                       threshold_times, threshold_voltages, min_times, min_voltages):
    """
    Create a plot showing spike analysis results.
    
    Args:
        time_trace (list): Time values
        voltage_trace (list): Voltage values
        peak_times (list): Spike peak times
        peak_voltages (list): Spike peak voltages
        threshold_times (list): Threshold times
        threshold_voltages (list): Threshold voltages
        min_times (list): AHP times
        min_voltages (list): AHP voltages
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Plot main trace
    ax.plot(time_trace, voltage_trace, 'b-', linewidth=2, label='Voltage trace')
    
    # Plot detected features
    ax.plot(peak_times, peak_voltages, 'ro', markersize=8, label='Spike peaks')
    ax.plot(threshold_times, threshold_voltages, 'k*', markersize=10, label='Thresholds')
    ax.plot(min_times, min_voltages, 'r*', markersize=10, label='AHP minima')
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def analyze_single_file(filename, derivative_threshold=3, stim_start=115, stim_end=500):
    """
    Analyze a single ABF file for spike characteristics.
    
    Args:
        filename (str): Path to ABF file
        derivative_threshold (float): eFEL derivative threshold parameter
        stim_start (float): Stimulus start time (ms)
        stim_end (float): Stimulus end time (ms)
        
    Returns:
        dict: Analysis results for all sweeps
    """
    print(f"Analyzing file: {filename}")
    
    abf = pyabf.ABF(filename)
    num_sweeps = len(abf.sweepList)
    print(f"Number of sweeps: {num_sweeps}")
    
    # Setup eFEL parameters
    setup_efel_parameters(derivative_threshold)
    
    file_results = {}
    
    for sweep_num in range(num_sweeps):
        print(f"  Processing sweep {sweep_num}")
        
        abf.setSweep(sweep_num)
        time_values = [t * 1000 for t in abf.sweepX]  # Convert to milliseconds
        voltage_values = list(abf.sweepY)
        current_values = list(abf.sweepC)
        
        # Prepare trace for eFEL
        trace = {
            'T': time_values,
            'V': voltage_values,
            'stim_start': [stim_start],
            'stim_end': [stim_end]
        }
        
        # Get current at 200ms for I-threshold calculation
        current_at_200ms = current_values[time_values.index(200)]
        
        # Extract features using eFEL
        features = extract_spike_features(trace)
        
        # Initialize sweep results
        sweep_results = {
            'corriente': current_at_200ms,
            'Vthr': 0,
            'AHP': 0,
            'Amplitud pico': 0,
            'Cant de Picos': 0,
            'width': 0,
            'width_al_pico': 0
        }
        
        # Process if spikes were detected
        if len(features[0]['peak_time']) > 0:
            peak_times = features[0]['peak_time']
            spike_heights = features[0]['AP_height']
            ahp_indices = features[0]['min_AHP_indices']
            
            # Calculate AHP times and voltages
            ahp_times = [time_values[idx] for idx in ahp_indices]
            ahp_voltages = [voltage_values[idx] for idx in ahp_indices]
            
            # Detect thresholds for all spikes
            threshold_voltages = []
            threshold_times = []
            
            for i, (peak_time, spike_height) in enumerate(zip(peak_times, spike_heights)):
                points_before = 50 if i == 0 else 100  # More points for subsequent spikes
                
                thr_v, thr_t = detect_spike_threshold(
                    time_values, voltage_values, peak_time, spike_height, 
                    points_before_peak=points_before
                )
                threshold_voltages.append(thr_v)
                threshold_times.append(thr_t)
            
            # Calculate spike width for first spike
            if len(threshold_voltages) > 0 and len(ahp_voltages) > 0:
                width, width_to_peak = calculate_spike_width(
                    voltage_values, time_values, threshold_voltages[0], 
                    threshold_times[0], ahp_voltages[0], ahp_times[0], peak_times[0]
                )
                sweep_results['width'] = width
                sweep_results['width_al_pico'] = width_to_peak
            
            # Count significant spikes (amplitude > 10 mV above threshold)
            spike_count = 0
            for spike_height, threshold_v in zip(spike_heights, threshold_voltages):
                if abs(spike_height - threshold_v) > 10:
                    spike_count += 1
            
            # Store results
            sweep_results.update({
                'Vthr': threshold_voltages[0] if threshold_voltages else 0,
                'AHP': abs(ahp_voltages[0] - threshold_voltages[0]) if ahp_voltages and threshold_voltages else 0,
                'Amplitud pico': abs(spike_heights[0] - threshold_voltages[0]) if spike_heights and threshold_voltages else 0,
                'Cant de Picos': spike_count
            })
            
            # Create visualization
            plot_spike_analysis(
                time_values, voltage_values, peak_times, spike_heights,
                threshold_times, threshold_voltages, ahp_times, ahp_voltages
            )
            
            print(f"    Detected {spike_count} spikes")
        
        file_results[f'sweep{sweep_num}'] = sweep_results
    
    return file_results


def calculate_file_averages(file_results):
    """
    Calculate average values across sweeps for a single file.
    
    Args:
        file_results (dict): Results from analyze_single_file
        
    Returns:
        dict: Averaged measurements
    """
    df = pd.DataFrame(file_results).T
    
    # Initialize averages
    averages = {
        'amplitude': 0,
        'ahp': 0,
        'vthr': 0,
        'i_thr': 0,
        'max_spikes': 0,
        'width_mean': 0,
        'width_to_peak_mean': 0
    }
    
    # Calculate averages only for sweeps with detected spikes
    spikes_detected = df[df['Cant de Picos'] > 0]
    
    if len(spikes_detected) > 0:
        averages.update({
            'i_thr': spikes_detected['corriente'].iloc[0],
            'max_spikes': int(spikes_detected['Cant de Picos'].max()),
            'amplitude': spikes_detected[spikes_detected['Amplitud pico'] > 0]['Amplitud pico'].mean(),
            'ahp': spikes_detected[spikes_detected['AHP'] > 0]['AHP'].mean(),
            'vthr': spikes_detected[spikes_detected['Vthr'] < 0]['Vthr'].mean(),
            'width_mean': spikes_detected[spikes_detected['width'] > 0]['width'].mean(),
            'width_to_peak_mean': spikes_detected[spikes_detected['width_al_pico'] > 0]['width_al_pico'].mean()
        })
        
        print(f"Average threshold voltage: {averages['vthr']:.3f} mV")
        print(f"Average AHP depth: {averages['ahp']:.3f} mV")
        print(f"Average spike amplitude: {averages['amplitude']:.3f} mV")
        print(f"Maximum spike count: {averages['max_spikes']}")
    
    return averages


def main():
    """
    Main analysis function that processes all ABF files and generates summary.
    """
    # Find all ABF files
    abf_files = glob.glob('*.abf')
    print(f"Found {len(abf_files)} ABF files to analyze\n")
    
    # Initialize results storage
    all_results = {}
    summary_data = {
        'amplitude': [], 'ahp': [], 'vthr': [], 'i_thr': [], 
        'max_spikes': [], 'width_mean': [], 'width_to_peak_mean': []
    }
    filenames = []
    
    # Process each file
    for filename in abf_files:
        print(f"\n{'='*60}")
        file_results = analyze_single_file(filename)
        all_results[filename] = file_results
        
        # Calculate and display averages
        averages = calculate_file_averages(file_results)
        
        # Store for summary
        filenames.append(filename)
        for key in summary_data:
            summary_data[key].append(averages[key])
        
        print(f"{'='*60}\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'File': filenames,
        'amp': summary_data['amplitude'],
        'ahp': summary_data['ahp'],
        'vthr': summary_data['vthr'],
        'i_thr': summary_data['i_thr'],
        'Max Picos': summary_data['max_spikes'],
        'width_mean': summary_data['width_mean'],
        'width_al_pico_mean': summary_data['width_to_peak_mean']
    })
    
    # Round numerical values
    numeric_columns = ['amp', 'ahp', 'vthr', 'i_thr', 'width_mean', 'width_al_pico_mean']
    summary_df[numeric_columns] = summary_df[numeric_columns].astype(float).round(3)
    
    # Save results
    output_filename = 'output_spiking.csv'
    summary_df.to_csv(
        output_filename,
        columns=['File', 'amp', 'ahp', 'vthr', 'i_thr', 'Max Picos', 'width_mean', 'width_al_pico_mean'],
        decimal=',',
        index=False
    )
    
    print(f"Results saved to: {output_filename}")
    print("\nSummary of all files:")
    print(summary_df)
    
    return summary_df


if __name__ == "__main__":
    main()
