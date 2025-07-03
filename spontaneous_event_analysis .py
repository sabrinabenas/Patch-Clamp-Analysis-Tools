"""
Spontaneous Event Analysis Tool for Patch Clamp Recordings

This script analyzes ABF (Axon Binary Format) files from patch clamp recordings
to detect and characterize spontaneous synaptic events (sEPSCs/sIPSCs), including:
- Event amplitude and frequency
- Event kinetics (rise time, decay tau, width)
- Area under the curve
- Artifact removal and signal cleaning

The script uses adaptive filtering and prominence-based peak detection
to automatically identify events in noisy recordings.

Author: Sabrina Benas
TODO: need validation with noisy data
"""


import pyabf
import pyabf.filter
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import glob


def apply_bessel_filters(current_data, low_cutoff=500, high_cutoff=1, sampling_rate=10000):
    """
    Apply band-pass Bessel filtering to remove noise.
    
    Args:
        current_data (array): Raw current trace
        low_cutoff (float): Low-pass filter cutoff frequency (Hz)
        high_cutoff (float): High-pass filter cutoff frequency (Hz)
        sampling_rate (float): Sampling frequency (Hz)
        
    Returns:
        tuple: (low_filtered, band_pass_filtered) current traces
    """
    # Subtract baseline (mean of full trace)
    current_adjusted = current_data - np.average(current_data)
    
    # Low-pass filter (removes high-frequency noise)
    b_low, a_low = signal.bessel(2, low_cutoff, 'low', 
                                analog=False, norm='phase', fs=sampling_rate)
    current_low_filtered = signal.lfilter(b_low, a_low, current_adjusted)
    
    # High-pass filter (removes low-frequency drift)
    b_high, a_high = signal.bessel(2, high_cutoff, 'high', 
                                  analog=False, norm='phase', fs=sampling_rate)
    current_band_filtered = signal.lfilter(b_high, a_high, current_low_filtered)
    
    return current_low_filtered, current_band_filtered


def determine_prominence_threshold(std_trace, filename=None):
    """
    Determine peak detection prominence based on trace noise level.
    
    Args:
        std_trace (float): Standard deviation of the filtered trace
        filename (str): Filename for manual adjustments
        
    Returns:
        float: Prominence threshold for peak detection
    """
    # Adaptive prominence based on noise level
    if std_trace < 0.75:
        prominence = 3
    elif 0.75 <= std_trace < 1.16:
        prominence = 4.5
    elif 1.16 <= std_trace < 1.8:
        prominence = 5
    elif 1.8 <= std_trace < 2.45:
        prominence = 7
    elif 2.45 <= std_trace < 3.2:
        prominence = 10
    else:  # std_trace >= 3.2
        prominence = 14
    
    # Manual adjustments for specific problematic files
    if filename in ['22d13090.abf', '22d12002.abf']:
        prominence = 3
        
    return prominence


def remove_artifact_segments(current_data, segment_size=20000, threshold=9):
    """
    Remove segments with large artifacts based on amplitude threshold.
    
    Args:
        current_data (array): Filtered current trace
        segment_size (int): Size of segments to analyze (samples)
        threshold (float): Amplitude threshold for artifact detection (pA)
        
    Returns:
        array: Current trace with artifacts removed
    """
    segments_to_remove = []
    
    # Check each segment for artifacts
    for i in range(int(len(current_data) / segment_size)):
        segment = current_data[i * segment_size:(i + 1) * segment_size]
        if (segment > threshold).any():
            segments_to_remove.append([i * segment_size, (i + 1) * segment_size])
    
    if not segments_to_remove:
        return current_data
    
    # Create list of segments to keep
    flat_indices = list(np.array(segments_to_remove).flatten())
    flat_indices.insert(0, 0)
    flat_indices.append(len(current_data))
    
    # Group indices into keep/remove pairs
    keep_segments = [flat_indices[i:i+2] for i in range(0, len(flat_indices), 2)]
    
    # Merge kept segments
    cleaned_segments = []
    for start, end in keep_segments[:-1]:  # Exclude the last placeholder
        if end == -1:
            cleaned_segments.append(list(current_data[start:]))
        else:
            cleaned_segments.append(list(current_data[start:end]))
    
    # Flatten the list
    cleaned_current = [value for segment in cleaned_segments for value in segment]
    
    return np.array(cleaned_current)


def remove_noisy_segments(current_data, segment_size=40000, noise_multiplier=2.5):
    """
    Remove segments with excessive noise based on rolling standard deviation.
    
    Args:
        current_data (array): Current trace
        segment_size (int): Size of segments for noise analysis (samples)
        noise_multiplier (float): Multiplier for median std threshold
        
    Returns:
        array: Current trace with noisy segments removed
    """
    # Calculate rolling standard deviation
    rolling_std = []
    for i in range(int(len(current_data) / segment_size)):
        segment = current_data[i * segment_size:(i + 1) * segment_size]
        rolling_std.append(np.std(segment))
    
    # Identify segments to remove
    threshold = noise_multiplier * np.median(rolling_std)
    segments_to_remove = []
    
    for idx, std_val in enumerate(rolling_std):
        if std_val > threshold:
            segments_to_remove.append([idx * segment_size, (idx + 1) * segment_size])
    
    # Remove segments (in reverse order to maintain indices)
    cleaned_current = list(current_data)
    for start, end in reversed(segments_to_remove):
        del cleaned_current[start:end]
    
    return np.array(cleaned_current)


def detect_spontaneous_events(current_data, sampling_rate=10000, prominence=5, 
                            height_min=3, height_max=120):
    """
    Detect spontaneous synaptic events using peak detection.
    
    Args:
        current_data (array): Filtered current trace
        sampling_rate (int): Sampling frequency (Hz)
        prominence (float): Minimum prominence for peak detection
        height_min (float): Minimum peak height (pA)
        height_max (float): Maximum peak height (pA)
        
    Returns:
        tuple: (peaks, peak_properties) from scipy.signal.find_peaks
    """
    peaks, peak_properties = find_peaks(
        -current_data,  # Negative for inward currents
        height=(height_min, height_max),
        threshold=None,
        distance=1,  # Minimum 0.1 ms between peaks
        prominence=prominence,
        width=0.2,  # Minimum width 0.02 ms
        wlen=500,  # Window length for prominence calculation
        rel_height=0.5,  # Relative height for width measurement
        plateau_size=None
    )
    
    return peaks, peak_properties


def analyze_event_kinetics(current_data, peaks, peak_properties, sampling_rate=10000):
    """
    Analyze kinetic properties of detected events.
    
    Args:
        current_data (array): Current trace
        peaks (array): Peak indices
        peak_properties (dict): Peak properties from find_peaks
        sampling_rate (int): Sampling frequency
        
    Returns:
        pandas.DataFrame: Table with event analysis results
    """
    if len(peaks) == 0:
        return pd.DataFrame()
    
    # Create results table
    table = pd.DataFrame({
        'event': np.arange(1, len(peaks) + 1),
        'peak_detection': peaks,
        'event_start': peak_properties['left_ips'] - (3 * sampling_rate / 1000),  # 3ms pre-trigger
        'event_end': peak_properties['right_ips'] + (6 * sampling_rate / 1000),   # 6ms post-trigger
        'Peak_Amp_pA': peak_properties['peak_heights'],
        'Width_ms': peak_properties['widths'] / (sampling_rate / 1000),  # Convert to ms
        'rise_half_amp_ms': (peaks - peak_properties['left_ips']) / (sampling_rate / 1000)
    })
    
    # Calculate instantaneous frequency
    freq_values = 1 / (np.diff(peaks) / sampling_rate)
    table['inst_freq'] = np.append(freq_values, np.nan)
    
    # Calculate inter-spike intervals
    table['isi_s'] = np.diff(peaks, prepend=peaks[0]) / sampling_rate
    
    # Calculate area under the curve for each event
    areas = []
    rise_times = []
    for i, (_, event) in enumerate(table.iterrows()):
        start_idx = int(event.event_start) + 20
        end_idx = int(event.event_end)
        
        # Area calculation
        individual_event = -current_data[start_idx:end_idx]
        area = np.round(individual_event.sum(), 1) / (sampling_rate / 1000)
        areas.append(area)
        
        # Rise time calculation (time from start to peak)
        time_ms = np.arange(len(current_data)) / (sampling_rate / 1000)
        rise_time = time_ms[peaks[i]] - start_idx / 10  # Convert to ms
        rise_times.append(rise_time)
    
    table['Area_pA/ms'] = areas
    table['rise_amp'] = rise_times
    
    # Calculate decay time constants
    log_decay_taus = []
    exp_decay_taus = []
    
    for i, (_, event) in enumerate(table.iterrows()):
        peak_idx = int(event.peak_detection)
        end_idx = int(event.event_end)
        
        # Logarithmic decay tau
        decay_trace = np.abs(current_data[peak_idx:end_idx])
        if len(decay_trace) > 1:
            log_decay = np.log(decay_trace + 1e-10)  # Add small value to avoid log(0)
            time_points = np.arange(len(decay_trace))
            
            try:
                slope, _ = np.polyfit(time_points, log_decay, 1)
                tau_log = -1 / slope / (sampling_rate / 1000)  # Convert to ms
            except:
                tau_log = np.nan
        else:
            tau_log = np.nan
        
        # Exponential decay tau
        try:
            decay_trace_exp = current_data[peak_idx:end_idx]
            time_points_exp = np.arange(len(decay_trace_exp))
            
            # Fit exponential: a * exp(b * t)
            popt, _ = curve_fit(
                lambda t, a, b: a * np.exp(b * t),
                time_points_exp, decay_trace_exp,
                p0=(200, 0.1), maxfev=5000
            )
            tau_exp = abs(1 / popt[1]) / (sampling_rate / 1000)  # Convert to ms
        except:
            tau_exp = np.nan
        
        log_decay_taus.append(tau_log)
        exp_decay_taus.append(tau_exp)
    
    table['log_decay'] = log_decay_taus
    table['tau_exp'] = exp_decay_taus
    
    return table


def plot_filtering_comparison(time, current_raw, current_low, current_band, 
                            filename, plot_enabled=True):
    """
    Plot comparison of filtering steps.
    
    Args:
        time (array): Time values
        current_raw (array): Raw current after baseline subtraction
        current_low (array): Low-pass filtered current
        current_band (array): Band-pass filtered current
        filename (str): Filename for plot title
        plot_enabled (bool): Whether to display the plot
    """
    if not plot_enabled:
        return
        
    plt.figure(figsize=(15, 6))
    plt.plot(time, current_raw, linewidth=0.5, label='Raw data', alpha=0.7)
    plt.plot(time, current_low, color='r', alpha=0.8, linewidth=0.5, label='Low-pass filter')
    plt.plot(time, current_band, color='g', alpha=0.8, linewidth=0.5, label='Band-pass filter')
    plt.legend(loc='upper right')
    plt.ylabel("Current (pA)")
    plt.xlabel("Time (s)")
    plt.title(f"Filtering comparison - {filename}")
    plt.show()


def plot_artifact_removal(original_current, cleaned_current, plot_enabled=True):
    """
    Plot before and after artifact removal.
    
    Args:
        original_current (array): Original current trace
        cleaned_current (array): Cleaned current trace
        plot_enabled (bool): Whether to display the plot
    """
    if not plot_enabled:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    ax1.plot(original_current)
    ax1.set_title('Original trace')
    ax1.set_ylabel('Current (pA)')
    
    ax2.plot(cleaned_current)
    ax2.set_title('After artifact removal')
    ax2.set_ylabel('Current (pA)')
    plt.show()


def plot_noise_analysis(rolling_std, threshold, cleaned_current, plot_enabled=True):
    """
    Plot noise analysis and final cleaned trace.
    
    Args:
        rolling_std (list): Rolling standard deviation values
        threshold (float): Noise threshold used
        cleaned_current (array): Final cleaned current trace
        plot_enabled (bool): Whether to display the plot
    """
    if not plot_enabled:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    
    ax1.plot(rolling_std)
    ax1.axhline(y=np.median(rolling_std), color='green', label='Median std')
    ax1.axhline(y=threshold, color='red', label='Threshold')
    ax1.set_title('Rolling standard deviation')
    ax1.set_ylabel('Standard deviation')
    ax1.legend()
    
    ax2.plot(cleaned_current)
    ax2.set_title('Final cleaned trace')
    ax2.set_ylabel('Current (pA)')
    plt.show()


def plot_event_detection(current_data, peaks, event_table, plot_enabled=True):
    """
    Plot detected events on the current trace.
    
    Args:
        current_data (array): Current trace
        peaks (array): Peak indices
        event_table (DataFrame): Event analysis results
        plot_enabled (bool): Whether to display the plot
    """
    if not plot_enabled:
        return
        
    fig = plt.figure(figsize=(18, 4))
    gridspec_layout = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[2, 1])
    ax1 = fig.add_subplot(gridspec_layout[0])
    
    ax1.set_title("Event Detection Results")
    ax1.plot(current_data, linewidth=0.8)
    ax1.plot(peaks, current_data[peaks], "r.", markersize=8)
    
    # Annotate events
    for i, event_num in enumerate(event_table['event']):
        ax1.annotate(str(event_num), (peaks[i], current_data[peaks[i]]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel("Sample number")
    ax1.set_ylabel("Current (pA)")
    plt.tight_layout()
    plt.show()


def plot_kinetics_histogram(event_table, column_name, plot_enabled=True):
    """
    Plot histogram of event kinetics parameter.
    
    Args:
        event_table (DataFrame): Event analysis results
        column_name (str): Column to plot histogram for
        plot_enabled (bool): Whether to display the plot
    """
    if not plot_enabled or event_table.empty:
        return
        
    plt.figure(figsize=(8, 6))
    plt.hist(event_table[column_name], bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {column_name}')
    plt.ylabel("Count")
    plt.xlabel(column_name)
    plt.grid(True, alpha=0.3)
    plt.show()


def analyze_single_file(filename, plot_enabled=True, height_min=3, noise_multiplier=2.5):
    """
    Analyze a single ABF file for spontaneous events.
    
    Args:
        filename (str): Path to ABF file
        plot_enabled (bool): Whether to display plots
        height_min (float): Minimum event height for detection
        noise_multiplier (float): Multiplier for noise threshold
        
    Returns:
        dict: Analysis results summary
    """
    print(f"Analyzing file: {filename}")
    
    # Load ABF file
    abf = pyabf.ABF(filename)
    time = abf.sweepX  # Time in seconds
    current = abf.sweepY  # Current in pA
    
    # Determine filter parameters based on filename
    file_prefix = filename.split('/')[-1][:2]
    low_filter = 250 if file_prefix == '23' else 500
    
    print(f"Using low-pass filter: {low_filter} Hz")
    
    # Apply filtering
    current_low_filtered, current_band_filtered = apply_bessel_filters(
        current, low_cutoff=low_filter
    )
    
    # Plot filtering comparison
    current_adjusted = current - np.average(current)
    plot_filtering_comparison(time, current_adjusted, current_low_filtered, 
                            current_band_filtered, filename, plot_enabled)
    
    # Determine prominence threshold
    std_trace = np.std(current_band_filtered)
    prominence = determine_prominence_threshold(std_trace, filename)
    print(f"Trace std: {std_trace:.3f}, Prominence threshold: {prominence}")
    
    # Remove artifact segments
    current_cleaned = remove_artifact_segments(current_band_filtered)
    plot_artifact_removal(current_band_filtered, current_cleaned, plot_enabled)
    
    # Remove noisy segments
    segment_size = 40000  # ~4 seconds at 10kHz
    rolling_std = []
    for i in range(int(len(current_cleaned) / segment_size)):
        segment = current_cleaned[i * segment_size:(i + 1) * segment_size]
        rolling_std.append(np.std(segment))
    
    threshold = noise_multiplier * np.median(rolling_std)
    final_current = remove_noisy_segments(current_cleaned, segment_size, noise_multiplier)
    
    plot_noise_analysis(rolling_std, threshold, final_current, plot_enabled)
    
    # Special handling for specific files
    if filename == '22413061.abf':
        final_current = final_current[450000:]  # Remove first ~45 seconds
    
    # Calculate analysis duration
    sampling_rate = int(len(current) / max(time))
    analysis_duration = len(final_current) / sampling_rate
    print(f"Analysis duration: {analysis_duration:.1f} seconds")
    
    # Detect events
    peaks, peak_properties = detect_spontaneous_events(
        final_current, sampling_rate, prominence, height_min
    )
    
    # Initialize results
    results = {
        'event_count': 0,
        'mean_amplitude': 0,
        'mean_area': 0,
        'mean_decay_tau': 0,
        'mean_rise_time': 0,
        'mean_width': 0,
        'analysis_duration': analysis_duration
    }
    
    if len(peaks) > 0:
        # Analyze event kinetics
        event_table = analyze_event_kinetics(final_current, peaks, peak_properties, sampling_rate)
        
        # Plot event detection
        plot_event_detection(final_current, peaks, event_table, plot_enabled)
        
        # Calculate summary statistics
        results.update({
            'event_count': len(event_table),
            'mean_amplitude': event_table['Peak_Amp_pA'].mean(),
            'mean_area': event_table['Area_pA/ms'].mean(),
            'mean_decay_tau': event_table['tau_exp'].mean(),
            'mean_rise_time': event_table['rise_amp'].mean(),
            'mean_width': event_table['Width_ms'].mean()
        })
        
        # Plot kinetics histogram
        plot_kinetics_histogram(event_table, 'rise_amp', plot_enabled)
        
        print(f"Detected {len(event_table)} events")
        print(f"Mean amplitude: {results['mean_amplitude']:.2f} pA")
        print(f"Event frequency: {len(event_table)/analysis_duration:.2f} Hz")
    else:
        print("No events detected")
    
    return results


def main(plot_enabled=True, height_min=3, noise_multiplier=2.5):
    """
    Main analysis function that processes all ABF files.
    
    Args:
        plot_enabled (bool): Whether to display plots during analysis
        height_min (float): Minimum event height for detection (pA)
        noise_multiplier (float): Multiplier for noise threshold
        
    Returns:
        pandas.DataFrame: Summary results for all files
    """
    # Find all ABF files
    abf_files = glob.glob('*.abf')
    print(f"Found {len(abf_files)} ABF files to analyze\n")
    
    # Initialize results storage
    results_data = {
        'event_count': [],
        'mean_amplitude': [],
        'mean_area': [],
        'mean_decay_tau': [],
        'mean_rise_time': [],
        'mean_width': [],
        'analysis_duration': []
    }
    filenames = []
    
    # Process each file
    for filename in abf_files:
        print(f"\n{'='*60}")
        
        try:
            file_results = analyze_single_file(filename, plot_enabled, height_min, noise_multiplier)
            
            # Store results
            filenames.append(filename.split('/')[-1])  # Just filename, not full path
            for key in results_data:
                results_data[key].append(file_results[key])
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            # Add zeros for failed files
            filenames.append(filename.split('/')[-1])
            for key in results_data:
                results_data[key].append(0)
        
        print(f"{'='*60}\n")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'File': filenames,
        'Event_Count': results_data['event_count'],
        'Mean_Amplitude_pA': results_data['mean_amplitude'],
        'Mean_Area_pA_ms': results_data['mean_area'],
        'Mean_Decay_Tau_ms': results_data['mean_decay_tau'],
        'Mean_Rise_Time_ms': results_data['mean_rise_time'],
        'Mean_Width_ms': results_data['mean_width'],
        'Analysis_Duration_s': results_data['analysis_duration']
    })
    
    # Round numerical values
    numeric_columns = [col for col in summary_df.columns if col != 'File']
    summary_df[numeric_columns] = summary_df[numeric_columns].round(3)
    
    # Save results
    output_filename = 'output_spontaneous_events.csv'
    summary_df.to_csv(output_filename, decimal=',', index=False)
    
    print(f"Results saved to: {output_filename}")
    print("\nSummary of all files:")
    print(summary_df)
    
    return summary_df


if __name__ == "__main__":
    # Run analysis with plots enabled (set to False to disable plotting)
    results = main(plot_enabled=True, height_min=3, noise_multiplier=2.5)
