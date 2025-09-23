import json
import os
import sys
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wasatch.WasatchBus import WasatchBus
from wasatch.WasatchDevice import WasatchDevice

import utils

######## Optimization mode ########
optimization_mode = False  # If True: 1s exposure, continuous, no dark/background, no saving
selected_peaks_cm = [] # For reporting, in cm^-1
optimization_integration_time = 1 # seconds
optimization_averages = 1 # spectra averaged per update (>=1)

######## Acquisition parameters ########
data_dir          = ""  # Directory for saving acquired data
prefix            = ""  # Prefix for file names
integration_time  = 1   # Seconds
laser_power       = 450 # mW
max_num_spectra   = 10  # Maximum number of spectra to acquire, set to None for unlimited

######## Spectrum correction parameters ########
use_dark          = False # True: Dark spectrum subtraction. False: No dark spectrum subtraction.
use_background    = False # True: Background + baseline correction. False: Just baseline correction.
background_file   = ""    # Path to background file (if use_background is True and you want to load from file)
poly_order        = 12    # Polynomial order for baseline fitting
max_iter          = 100   # Maximum number of iterations for background removal
crop_range        = None  # Crop range for the spectrum (in cm^-1). Set to None to disable cropping.

if optimization_mode:
    optimization_averages = max(1, int(optimization_averages))
    integration_time = optimization_integration_time
    max_num_spectra = None  # continuous until stopped
    use_background = False
    use_dark = False

# Initialize Wasatch spectrometer
bus = WasatchBus()
if not bus.device_ids:
    print("No spectrometers found")
    sys.exit(1)
device_id = bus.device_ids[0]
print("Found spectrometer with ID:", device_id)

spectrometer = WasatchDevice(device_id)
if not spectrometer.connect():
    print("Failed to connect to spectrometer")
    sys.exit(1)    

if spectrometer.settings.eeprom.model == None:
    print("Spectrometer information not available. Ensure the device is connected properly.")
    sys.exit(1)

print("Connected to %s %s" % (
    spectrometer.settings.eeprom.model,
    spectrometer.settings.eeprom.serial_number))

# Set spectrometer parameters
spectrometer.hardware.set_integration_time_ms(integration_time * 1000) # Convert seconds to milliseconds
print(f"Integration time set to {integration_time} seconds ({integration_time * 1000} ms)")
spectrometer.hardware.set_laser_power_mW(laser_power)
print(f"Laser power set to {laser_power} mW")

if spectrometer.settings.wavenumbers is None:
    print("Wavenumbers not available. Ensure the spectrometer is configured correctly.")
    sys.exit(1)
wavenumbers = np.asarray(spectrometer.settings.wavenumbers)

# Acquire dark spectrum
if use_dark:
    print("Acquiring dark spectrum...")
    spectrometer.hardware.set_laser_enable(False)
    dark_spectrum = np.array(spectrometer.hardware.get_line().data.spectrum)
else:
    print("Dark spectrum subtraction is disabled. Using zero dark spectrum.")
    dark_spectrum = np.zeros(spectrometer.settings.pixels())

# Turn on the laser
spectrometer.hardware.set_laser_enable(True)
print("WARNING: Laser is ON. Ensure safety precautions are followed.")

# Keep figure and event connection IDs accessible for cleanup
fig = None
keypress_cid = None

try:
    # Background spectrum acquisition
    if use_background and background_file:
        print(f"Loading background spectrum from file: {background_file}")
        background_df = pd.read_csv(background_file)
        background = background_df.drop(columns=['Wavenumbers', 'Background']).to_numpy()
        background = np.mean(background, axis=1)  # Average across all background spectra
    elif use_background:
        print("The script will now acquire 3 background spectra for averaging.")
        user_input = input("Press Enter to start background acquisition...")
        background = []
        for i in range(3): # Acquire 3 background spectra for averaging
            print(f"Acquiring background spectrum {i+1}/3...")
            spectrum = spectrometer.hardware.get_line().data.spectrum - dark_spectrum
            background.append(spectrum)
        background = np.mean(background, axis=0)
        print("Background spectrum acquired and averaged.")
    else:
        background = np.zeros(spectrometer.settings.pixels())
        print("Background subtraction is disabled. Using zero background.")

    # Active spectrum acquisition
    user_input = input("Press Enter to start active spectrum acquisition...")

    # Create a timestamp for creating unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.ion()
    if optimization_mode:
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.3, hspace=0.25)
        ax_corr = fig.add_subplot(gs[0, 0])
        ax_raw = fig.add_subplot(gs[1, 0], sharex=ax_corr)
        ax_time = fig.add_subplot(gs[:, 1])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        ax_raw, ax_corr = axes
        ax_time = None
    fig_manager = plt.get_current_fig_manager()
    fig_manager.full_screen_toggle()
    fig.tight_layout(pad=3)

    if ax_time is not None:
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Intensity (a.u.)')
        ax_time.set_title('Selected Peak Intensities Over Time')
        ax_time.grid(True, linestyle='--', alpha=0.3)

    # Raw plot (bottom-left in optimization mode)
    line_raw, = ax_raw.plot([], [], label='Raw Spectrum', color='blue')
    if use_background:
        line_baseline, = ax_raw.plot([], [], label='Baseline + Background', color='green')
        ax_raw.set_title('Raw, Background, Baseline')
    else:
        line_baseline, = ax_raw.plot([], [], label='Baseline', color='green')
        ax_raw.set_title('Raw, Baseline')
    ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax_raw.set_ylabel('Intensity (a.u.)')
    ax_raw.legend()

    # Corrected spectrum plot (top-left in optimization mode)
    line_corr, = ax_corr.plot([], [], label='Corrected Spectrum', color='red')
    ax_corr.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax_corr.set_ylabel('Intensity (a.u.)')
    ax_corr.legend()
    ax_corr.set_title('Corrected Spectrum')

    # Optional markers for selected peaks (only used in optimization mode)
    line_raw_peaks = None
    line_corr_peaks = None
    intensity_lines = {}
    intensity_history = {}
    time_history = []
    start_time = None
    if optimization_mode and selected_peaks_cm:
        line_raw_peaks, = ax_raw.plot([], [], linestyle='none', marker='o',
                                       markersize=6, color='magenta',
                                       label='Selected Peaks')
        line_corr_peaks, = ax_corr.plot([], [], linestyle='none', marker='o',
                                         markersize=6, color='magenta',
                                         label='Selected Peaks')
        ax_raw.legend()
        ax_corr.legend()

        # Prepare intensity tracking plot
        start_time = time.time()
        intensity_history = {peak_cm: [] for peak_cm in selected_peaks_cm}
        cmap = plt.cm.get_cmap('tab10')
        for idx, peak_cm in enumerate(selected_peaks_cm):
            color = cmap(idx % cmap.N)
            line, = ax_time.plot([], [], label=f'{peak_cm:.1f} cm$^{-1}$', color=color)
            intensity_lines[peak_cm] = line
        ax_time.legend(loc='upper right')

    stop_event = threading.Event()

    def wait_for_input(event):
        input("Press Enter in terminal to stop plotting...\n")
        event.set()

    # Start a thread to wait for user input to stop plotting
    input_thread = threading.Thread(target=wait_for_input, args=(stop_event,))
    input_thread.daemon = True
    input_thread.start()

    # Also allow stopping via keyboard ('q' or 'escape') in the figure window
    def on_key(event):
        if event.key in ('q', 'escape'):
            print("Stop requested via keyboard.")
            stop_event.set()

    keypress_cid = fig.canvas.mpl_connect('key_press_event', on_key)

    # Initialize data arrays
    raw_data = np.array([]).reshape(0, spectrometer.settings.pixels())
    corrected_data = np.array([]).reshape(0, spectrometer.settings.pixels())

    # Crop the wavenumbers and background
    if crop_range is not None:
        crop_mask = (wavenumbers >= crop_range[0]) & (wavenumbers <= crop_range[1])
        cropped_wavenumbers = wavenumbers[crop_mask]
        corrected_data = corrected_data[:, crop_mask]
        cropped_background = background[crop_mask]
    else:
        crop_mask = np.ones_like(wavenumbers, dtype=bool)
        cropped_wavenumbers = wavenumbers
        cropped_background = background

    # Acquire spectra
    counter = 0
    while not stop_event.is_set() and (counter < max_num_spectra if max_num_spectra is not None else True):
        # Acquire spectrum with optional averaging in optimization mode
        if optimization_mode and optimization_averages > 1:
            frames = []
            for _ in range(optimization_averages):
                frames.append(np.array(spectrometer.hardware.get_line().data.spectrum))
            spectrum = np.mean(frames, axis=0) - dark_spectrum
        else:
            spectrum = np.array(spectrometer.hardware.get_line().data.spectrum) - dark_spectrum
        if not optimization_mode:
            raw_data = np.vstack([raw_data, spectrum]) if raw_data.size else spectrum

        cropped_spectrum = spectrum[crop_mask]
        corrected_spectrum = utils.remove_background(
                cropped_wavenumbers, cropped_spectrum, cropped_background, poly_order, max_iter, eps=0.1
            )
        baseline = cropped_spectrum.flatten() - corrected_spectrum.flatten()
        if not optimization_mode:
            corrected_data = np.vstack([corrected_data, corrected_spectrum]) if corrected_data.size else corrected_spectrum

        # Update raw plot
        line_raw.set_data(wavenumbers, spectrum.flatten())
        line_baseline.set_data(cropped_wavenumbers, baseline)

        ax_raw.set_title(f'Raw Spectrum {counter + 1}')
        ax_raw.relim()
        ax_raw.autoscale_view()

        # Update corrected plot
        line_corr.set_data(cropped_wavenumbers, corrected_spectrum.flatten())
        ax_corr.set_title(f'Corrected Spectrum {counter + 1}')
        ax_corr.relim()
        ax_corr.autoscale_view()

        # Update selected peak markers if enabled
        if line_raw_peaks is not None and line_corr_peaks is not None:
            # Raw spectrum markers (left)
            raw_xs = []
            raw_ys = []
            for peak_cm in selected_peaks_cm:
                idx_raw = int(np.argmin(np.abs(wavenumbers - peak_cm)))
                raw_xs.append(float(wavenumbers[idx_raw]))
                raw_ys.append(float(spectrum.flatten()[idx_raw]))
            line_raw_peaks.set_data(raw_xs, raw_ys)

            # Corrected spectrum markers (right)
            corr_xs = []
            corr_ys = []
            corr_flat = corrected_spectrum.flatten()
            for peak_cm in selected_peaks_cm:
                idx_corr = int(np.argmin(np.abs(cropped_wavenumbers - peak_cm)))
                corr_xs.append(float(cropped_wavenumbers[idx_corr]))
                corr_ys.append(float(corr_flat[idx_corr]))
            line_corr_peaks.set_data(corr_xs, corr_ys)

        # If in optimization mode, report intensity at selected peaks (cm^-1)
        if optimization_mode and selected_peaks_cm:
            intensities = []
            corr_flat = corrected_spectrum.flatten()
            for peak_cm in selected_peaks_cm:
                # Find nearest index within the cropped wavenumbers
                idx = int(np.argmin(np.abs(cropped_wavenumbers - peak_cm)))
                nearest_cm = float(cropped_wavenumbers[idx])
                intensity = float(corr_flat[idx])
                intensities.append((peak_cm, nearest_cm, intensity))
            # Print a compact readout per iteration
            readout = ", ".join(
                [f"{p:.1f} cm^-1 (~{n:.1f}) = {i:.2f} a.u." for p, n, i in intensities]
            )
            print(f"[Optimization] Spectrum {counter + 1}: {readout}")

            # Update intensity history plot
            elapsed = time.time() - start_time if start_time is not None else 0.0
            time_history.append(elapsed)
            for peak_cm, _, intensity in intensities:
                intensity_history[peak_cm].append(intensity)
                intensity_lines[peak_cm].set_data(time_history, intensity_history[peak_cm])
            if ax_time is not None:
                ax_time.relim()
                ax_time.autoscale_view()

        plt.pause(0.05)
        counter += 1

    # Turn off interactive mode (window will be closed after saving)
    plt.ioff()

    if optimization_mode:
        print("Optimization mode run complete (no data saved).")
    else:
        # Directories for saving data
        os.makedirs(data_dir or ".", exist_ok=True)

        # Save the final plots and close the figure
        fig.savefig(os.path.join(data_dir or ".", f"{prefix}_{timestamp}_plots.png"))
        plt.close(fig)

        # Create dataframes
        raw_df = pd.DataFrame(raw_data.T)
        corrected_df = pd.DataFrame(corrected_data.T)
        raw_df.insert(0, 'Wavenumbers', wavenumbers)
        corrected_df.insert(0, 'Wavenumbers', cropped_wavenumbers)
        raw_df.insert(1, 'Background', background)
        corrected_df.insert(1, 'Background', cropped_background)

        # Save data to CSV files with timestamp and prefix
        raw_filename = os.path.join(data_dir or ".", f"{prefix}_{timestamp}_raw_data.csv")
        corrected_filename = os.path.join(data_dir or ".", f"{prefix}_{timestamp}_corrected_data.csv")

        raw_df.to_csv(raw_filename, index=False)
        corrected_df.to_csv(corrected_filename, index=False)

        # Save the parameters used for acquisition
        params = {
            'integration_time': integration_time,
            'laser_power': laser_power,
            'use_background': use_background,
            'background_file': background_file,
            'poly_order': poly_order,
            'max_iter': max_iter,
            'max_num_spectra': max_num_spectra,
            'timestamp': timestamp,
            'prefix': prefix
        }
        params_filename = os.path.join(data_dir or ".", f"{prefix}_{timestamp}_params.json")
        with open(params_filename, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"Data saved to {raw_filename} and {corrected_filename}")
        print(f"Acquisition parameters saved to {params_filename}")
        print("Acquisition complete.")
finally:
    # Ensure the laser is turned off even if errors occur
    spectrometer.hardware.set_laser_enable(False)
    print("Laser is OFF.")
    # Ensure figure window is closed and callbacks disconnected
    try:
        if fig is not None and keypress_cid is not None:
            fig.canvas.mpl_disconnect(keypress_cid)
    except Exception:
        pass
    try:
        if fig is not None:
            plt.close(fig)
    except Exception:
        pass
    spectrometer.disconnect()
