"""
Patch Clamp Analysis Tool for Input Resistance and Capacitance Measurement

This script analyzes ABF (Axon Binary Format) files from patch clamp recordings
to calculate input resistance, membrane capacitance, and spike amplitude.

Author: Sabrina Benas
"""

import pyabf
import matplotlib.pyplot as plt 
import numpy as np
import glob
import matplotlib.cm as cm
import pandas as pd
from numpy import trapz
from scipy.signal import find_peaks


def find_closest_value(value_list, target_value):
    """
    Find the closest value in a list to avoid sampling issues.
    
    Args:
        value_list (list): List of numerical values
        target_value (float): Target value to find closest match for
        
    Returns:
        float: Closest value in the list to target_value
    """
    return min(value_list, key=lambda x: abs(x - target_value))


def calculate_baseline_data(abf, sweep, cursor1=0.000, cursor2=0.0002):
    """
    Calculate baseline-corrected data for a specific sweep.
    
    Args:
        abf: ABF file object
        sweep (int): Sweep number to analyze
        cursor1 (float): Start time for baseline calculation (seconds)
        cursor2 (float): End time for baseline calculation (seconds)
        
    Returns:
        tuple: (time, current, voltage) lists
    """
    abf.setSweep(sweep, baseline=[cursor1, find_closest_value(list(abf.sweepX), cursor2)])
    time = list(abf.sweepX)  # Convert to list for easier manipulation
    current = list(abf.sweepY)
    voltage = list(abf.sweepC)
    return time, current, voltage


def calculate_current_difference(time, current, cursor1=0.008, cursor2=0.04):
    """
    Calculate the difference in current between two time points.
    This is used to determine input resistance.
    
    Args:
        time (list): Time values in seconds
        current (list): Current values in pA
        cursor1 (float): First time point (seconds)
        cursor2 (float): Second time point (seconds)
        
    Returns:
        float: Current difference in pA
    """
    # Calculate mean current around cursor2 (±5ms window)
    idx2_start = time.index(find_closest_value(time, cursor2 - 0.005))
    idx2_end = time.index(find_closest_value(time, cursor2 + 0.005))
    val1 = np.mean(current[idx2_start:idx2_end])
    
    # Calculate mean current around cursor1 (±0.5ms window)
    idx1_start = time.index(find_closest_value(time, cursor1 - 0.0005))
    idx1_end = time.index(find_closest_value(time, cursor1 + 0.0005))
    val2 = np.mean(current[idx1_start:idx1_end])
    
    return val1 - val2


def calculate_average_traces(abf, sweep_list):
    """
    Calculate averaged time, current, and voltage traces across multiple sweeps.
    
    Args:
        abf: ABF file object
        sweep_list (list): List of sweep numbers to average
        
    Returns:
        tuple: (avg_time, avg_current, avg_voltage) lists
    """
    current_traces = []
    time_traces = []
    voltage_traces = []
    
    for sweep_number in sweep_list:
        time, current, voltage = calculate_baseline_data(abf, sweep_number)
        current_traces.append(current)
        time_traces.append(time)
        voltage_traces.append(voltage)

    # Calculate averages across all sweeps
    avg_time = list(np.mean(time_traces, axis=0))
    avg_current = list(np.mean(current_traces, axis=0))
    avg_voltage = list(np.mean(voltage_traces, axis=0))
    
    return avg_time, avg_current, avg_voltage


def calculate_membrane_capacitance(time, current, voltage, avg_time, avg_current, cursor2=0.04):
    """
    Calculate membrane capacitance using transient current integration.
    
    Args:
        time (list): Time values in seconds
        current (list): Current values in pA
        voltage (list): Voltage values in mV
        avg_time (list): Average time trace
        avg_current (list): Average current trace
        cursor2 (float): End time for integration (seconds)
        
    Returns:
        list: Capacitance value in pF
    """
    capacitance = []  # pF (picofarads)

    # Find the baseline current at cursor2
    baseline_idx = avg_time.index(find_closest_value(avg_time, cursor2))
    baseline_current = avg_current[baseline_idx]
    
    # Find the start of voltage pulse (when voltage reaches -10 mV)
    pulse_start_idx = time.index(time[voltage.index(find_closest_value(voltage, -10))])
    pulse_end_idx = time.index(find_closest_value(time, cursor2))
    
    # Calculate current transient (absolute difference from baseline)
    transient_current = [
        abs(current_val - baseline_current) 
        for current_val in current[pulse_start_idx:pulse_end_idx]
    ]
    
    # Time step in milliseconds (convert from seconds)
    dt = 0.02  # ms (assuming 50 kHz sampling rate)
    
    # Calculate area under the curve (charge)
    area = trapz(transient_current, dx=dt)
    capacitance.append(area / 10)  # Convert to pF (pA*ms/mV = pF)

    return capacitance


def calculate_spike_amplitude(current_trace, threshold=50):
    """
    Calculate spike amplitude using peak detection.
    
    Args:
        current_trace (list): Current values in pA
        threshold (float): Minimum height for peak detection (mV)
        
    Returns:
        float: Spike amplitude in mV, or 0 if no peaks found
    """
    peaks, _ = find_peaks(current_trace, height=threshold)
    
    if len(peaks) > 0:
        return current_trace[peaks[0]]
    else:
        print("Warning: No peaks detected above threshold")
        return 0


def main():
    """
    Main analysis function that processes all ABF files in the current directory.
    """
    # Initialize result lists
    input_resistances = []
    capacitances = []
    spike_amplitudes = []
    filenames = []
    
    # Get all ABF files in current directory
    abf_files = glob.glob('*.abf')
    print(f"Found {len(abf_files)} ABF files to process\n")
    
    # Process each file
    for filename in np.sort(abf_files):
        print(f"Processing: {filename}")
        filenames.append(filename)
        
        # Load ABF file
        abf = pyabf.ABF(filename)
        
        # Calculate average traces across all sweeps
        avg_time, avg_current, avg_voltage = calculate_average_traces(abf, abf.sweepList)
        
        # Calculate input resistance
        current_diff = calculate_current_difference(avg_time, avg_current)
        input_resistance = round((-10 / current_diff) * 1000, 3)  # Convert to MOhm
        print(f"  Input resistance: {input_resistance} MOhm")
        
        # Calculate membrane capacitance
        capacitance = calculate_membrane_capacitance(avg_time, avg_current, avg_voltage, 
                                                   avg_time, avg_current)
        print(f"  Membrane capacitance: {capacitance[0]:.3f} pF")
        
        # Calculate spike amplitude
        spike_amplitude = calculate_spike_amplitude(avg_current)
        print(f"  Spike amplitude: {spike_amplitude:.3f} mV\n")
        
        # Store results
        input_resistances.append(input_resistance)
        capacitances.append(capacitance[0])
        spike_amplitudes.append(spike_amplitude)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'File': filenames,
        'Rinput': input_resistances,
        'Capacitance': capacitances,
        'Amplitude': spike_amplitudes
    })
    
    # Save to CSV file
    output_filename = 'output_Rinput.csv'
    results_df.to_csv(output_filename, 
                     columns=['File', 'Rinput', 'Capacitance', 'Amplitude'],
                     decimal=',', 
                     index=False)
    
    print(f"Results saved to: {output_filename}")
    print("\nSummary Statistics:")
    print(results_df.describe())


if __name__ == "__main__":
    main()
