import json
import os
import sys
import threading
import time
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import ramanspy as rp
from wasatch.WasatchBus import WasatchBus
from wasatch.WasatchDevice import WasatchDevice

# PARAMETERS

# Acquisition / control
optimization_mode       = False   # If True: continuous, no saving
data_dir                = ""      # Directory to save data; "" means current dir
prefix                  = ""      # Filename prefix
integration_time        = 1.0     # seconds
laser_power             = 450.0   # mW
max_num_spectra         = 10      # int or None for unlimited

# Optional corrections BEFORE Ramanspy
use_dark                = False   # acquire a dark spectrum and subtract
use_background          = False   # subtract provided/acquired background BEFORE Ramanspy
background_file         = ""      # CSV file path (if use_background and load from file)

# Optimization extras
optimization_averages   = 1       # >=1 frames per displayed spectrum (only used in optimization_mode)
optimization_int_time   = 1.0     # seconds (overrides integration_time in optimization_mode)
selected_peaks_cm       = []      # e.g., [520, 1000, 1600]

# Ramanspy preprocessing parameters
crop_region             = (400, 1800)  # cm^-1
asls_lam                = 1e2
asls_p                  = 0.01
SG_window               = 7
SG_polyorder            = 3


# Ramanspy pipeline function (NO DESPIKING)
def rp_process_spectrum(raw_spectrum_1d: np.ndarray,
                        RS_1d_cm: np.ndarray,
                        crop_region=(400, 1800),
                        asls_lam=1e2, asls_p=0.01,
                        SG_window=7, SG_polyorder=3):
    """
    Returns:
        processed_RS       : cropped Raman shifts (cm^-1)
        processed_spectrum : ASLS baseline-corrected spectrum (cropped)
        smooth_spectrum    : SavGol + vector-normalized (cropped)
        baseline           : baseline estimate over cropped domain (raw_cropped - processed)
    """
    raw_spec = rp.Spectrum(np.asarray(raw_spectrum_1d, dtype=float),
                           np.asarray(RS_1d_cm, dtype=float))

    cropped   = rp.preprocessing.misc.Cropper(region=crop_region).apply(raw_spec)
    processed = rp.preprocessing.baseline.ASLS(lam=asls_lam, p=asls_p).apply(cropped)
    baseline  = cropped.spectral_data - processed.spectral_data

    smoothed  = rp.preprocessing.denoise.SavGol(window_length=SG_window,
                                                polyorder=SG_polyorder).apply(processed)
    smoothed  = rp.preprocessing.normalise.Vector(pixelwise=True).apply(smoothed)

    return (processed.spectral_axis,
            processed.spectral_data,
            smoothed.spectral_data,
            baseline)


def main():
    global integration_time, max_num_spectra, use_background, use_dark, optimization_averages
    if optimization_mode:
        integration_time = optimization_int_time
        max_num_spectra = None
        use_background = False
        use_dark = False
        optimization_averages = max(1, int(optimization_averages))

    # Initialize spectrometer
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

    if spectrometer.settings.eeprom.model is None:
        print("Spectrometer information not available. Ensure the device is connected properly.")
        sys.exit(1)

    print("Connected to %s %s" % (
        spectrometer.settings.eeprom.model,
        spectrometer.settings.eeprom.serial_number))

    # Configure spectrometer
    spectrometer.hardware.set_integration_time_ms(int(integration_time * 1000))
    print(f"Integration time set to {integration_time} seconds ({int(integration_time * 1000)} ms)")
    spectrometer.hardware.set_laser_power_mW(float(laser_power))
    print(f"Laser power set to {laser_power} mW")

    if spectrometer.settings.wavenumbers is None:
        print("Wavenumbers not available. Ensure the spectrometer is configured correctly.")
        sys.exit(1)
    wavenumbers = np.asarray(spectrometer.settings.wavenumbers)

    # ---------- Dark spectrum & Laser State Management ----------
    if use_dark:
        print("Acquiring dark spectrum with LASER OFF...")
        try:
            spectrometer.hardware.set_laser_enable(False)
        except Exception:
            pass
        dark_spectrum = np.array(spectrometer.hardware.get_line().data.spectrum)
        
        spectrometer.hardware.set_laser_enable(True)
        print("Dark acquired. Laser is NOW ON. Ensure safety precautions are followed.")
    else:
        print("Dark spectrum subtraction is disabled. Using zero dark spectrum.")
        dark_spectrum = np.zeros(spectrometer.settings.pixels())
        
        spectrometer.hardware.set_laser_enable(True)
        print("Laser is ON. Ensure safety precautions are followed.")
    # -----------------------------------------------------------

    # Background
    if use_background and background_file:
        print(f"Loading background spectrum from file: {background_file}")
        background_df = pd.read_csv(background_file)
        cols_to_drop = [c for c in ["Wavenumbers", "Background"] if c in background_df.columns]
        bg_block = background_df.drop(columns=cols_to_drop, errors="ignore").select_dtypes(include=[np.number])
        if bg_block.empty:
            raise ValueError("No numeric columns found in background file.")
        bg = bg_block.to_numpy()
        if bg.shape[0] == spectrometer.settings.pixels():
            background = np.mean(bg, axis=1)
        elif bg.shape[1] == spectrometer.settings.pixels():
            background = np.mean(bg, axis=0)
        else:
            raise ValueError("Background file dimensions don't match spectrometer pixel count.")
    elif use_background:
        print("The script will now acquire 3 background spectra for averaging.")
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to start background acquisition...")
        tmp = []
        for i in range(3):
            print(f"Acquiring background spectrum {i+1}/3...")
            spec = spectrometer.hardware.get_line().data.spectrum - dark_spectrum
            tmp.append(spec)
        background = np.mean(tmp, axis=0)
        print("Background spectrum acquired and averaged.")
    else:
        background = np.zeros(spectrometer.settings.pixels())
        print("Background subtraction is disabled. Using zero background.")

    # Plot/UI setup
    fig = None
    keypress_cid = None

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.ion()
        if optimization_mode:
            fig = plt.figure(figsize=(24, 9), constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.04)
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], wspace=0.02, hspace=0.05)
            ax_corr = fig.add_subplot(gs[0, 0])
            ax_raw = fig.add_subplot(gs[1, 0])
            ax_time = fig.add_subplot(gs[:, 1])
        else:
            fig, axes = plt.subplots(1, 2, figsize=(20, 5), constrained_layout=True)
            ax_raw, ax_corr = axes
            ax_time = None

        # Fullscreen
        try:
            fig_manager = plt.get_current_fig_manager()
            if hasattr(fig_manager, "full_screen_toggle"):
                fig_manager.full_screen_toggle()
        except Exception:
            pass

        if ax_time is not None:
            ax_time.set_xlabel('Time (s)')
            ax_time.set_ylabel('Intensity (a.u.)')
            ax_time.set_title('Selected Peak Intensities Over Time')
            ax_time.grid(True, linestyle='--', alpha=0.3)
            ax_time.set_anchor('W')

        # Lines
        line_raw, = ax_raw.plot([], [], label='Raw Spectrum')
        baseline_label = 'Baseline + Background' if use_background else 'Baseline'
        line_baseline, = ax_raw.plot([], [], label=baseline_label)
        ax_raw.set_title('Raw / Baseline')
        ax_raw.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax_raw.set_ylabel('Intensity (a.u.)')
        ax_raw.legend()

        line_corr, = ax_corr.plot([], [], label='Processed (SavGol + VectorNorm)')
        ax_corr.set_xlabel('Raman Shift (cm$^{-1}$)')
        ax_corr.set_ylabel('Intensity (a.u.)')
        ax_corr.legend()
        ax_corr.set_title('Corrected Spectrum')

        # Optional peaks
        line_raw_peaks = None
        line_corr_peaks = None
        intensity_lines = {}
        intensity_history = {}
        time_history = []
        start_time = None

        if optimization_mode and selected_peaks_cm:
            line_raw_peaks, = ax_raw.plot([], [], linestyle='none', marker='o', markersize=6, label='Selected Peaks')
            line_corr_peaks, = ax_corr.plot([], [], linestyle='none', marker='o', markersize=6, label='Selected Peaks')
            ax_raw.legend()
            ax_corr.legend()

            cmap = colormaps.get_cmap('tab10')
            intensity_history = {peak_cm: [] for peak_cm in selected_peaks_cm}
            for idx, peak_cm in enumerate(selected_peaks_cm):
                color = cmap(idx % cmap.N)
                line, = ax_time.plot([], [], label=f'{peak_cm:.1f} cm$^{-1}$', color=color)
                intensity_lines[peak_cm] = line
            ax_time.legend(loc='upper right')

        # Stop controls
        stop_event = threading.Event()

        def wait_for_stop(event):
            if sys.stdin and sys.stdin.isatty():
                input("Press Enter in terminal to STOP plotting...\n")
                event.set()

        def on_key(event):
            if event.key in ('q', 'escape'):
                print("Stop requested via keyboard.")
                stop_event.set()
        keypress_cid = fig.canvas.mpl_connect('key_press_event', on_key)

        # Buffers
        pixels = spectrometer.settings.pixels()
        raw_data = np.empty((0, pixels))
        corrected_data = None
        processed_RS = None

        # Start gate (only for acquisition/plotting, not laser)
        if sys.stdin and sys.stdin.isatty():
            input("Press Enter to START acquisition and plotting...")

        if optimization_mode and selected_peaks_cm:
            start_time = time.time()

        # Start stop-listener AFTER acquisition starts
        input_thread = threading.Thread(target=wait_for_stop, args=(stop_event,), daemon=True)
        input_thread.start()

        # Acquire
        counter = 0
        try:
            while not stop_event.is_set() and (max_num_spectra is None or counter < max_num_spectra):
                if optimization_mode and optimization_averages > 1:
                    frames = [np.array(spectrometer.hardware.get_line().data.spectrum)
                              for _ in range(optimization_averages)]
                    spectrum = np.mean(frames, axis=0)
                else:
                    spectrum = np.array(spectrometer.hardware.get_line().data.spectrum)

                spectrum = spectrum - dark_spectrum
                if use_background and background.size == spectrum.size:
                    spectrum = spectrum - background

                processed_RS, processed_spectrum, smooth_spectrum, baseline = rp_process_spectrum(
                    spectrum.flatten(), wavenumbers,
                    crop_region=crop_region,
                    asls_lam=asls_lam, asls_p=asls_p,
                    SG_window=SG_window, SG_polyorder=SG_polyorder
                )
                corrected_spectrum = smooth_spectrum

                if corrected_data is None:
                    corrected_data = np.empty((0, processed_RS.size))

                line_raw.set_data(wavenumbers, spectrum.flatten())
                line_baseline.set_data(processed_RS, baseline.flatten())
                ax_raw.set_title(f'Raw Spectrum {counter + 1}')
                ax_raw.relim(); ax_raw.autoscale_view()

                line_corr.set_data(processed_RS, corrected_spectrum.flatten())
                ax_corr.set_title(f'Corrected Spectrum {counter + 1}')
                ax_corr.relim(); ax_corr.autoscale_view()

                # Optional: peaks & time series (if configured)
                if line_raw_peaks is not None and line_corr_peaks is not None:
                    raw_xs, raw_ys = [], []
                    for peak_cm in selected_peaks_cm:
                        idx_raw = int(np.argmin(np.abs(wavenumbers - peak_cm)))
                        raw_xs.append(float(wavenumbers[idx_raw]))
                        raw_ys.append(float(spectrum.flatten()[idx_raw]))
                    line_raw_peaks.set_data(raw_xs, raw_ys)

                    corr_xs, corr_ys = [], []
                    corr_flat = corrected_spectrum.flatten()
                    for peak_cm in selected_peaks_cm:
                        idx_corr = int(np.argmin(np.abs(processed_RS - peak_cm)))
                        corr_xs.append(float(processed_RS[idx_corr]))
                        corr_ys.append(float(corr_flat[idx_corr]))
                    line_corr_peaks.set_data(corr_xs, corr_ys)

                if optimization_mode and selected_peaks_cm:
                    elapsed = time.time() - start_time if start_time is not None else 0.0
                    time_history.append(elapsed)
                    corr_flat = corrected_spectrum.flatten()
                    for peak_cm in selected_peaks_cm:
                        idx = int(np.argmin(np.abs(processed_RS - peak_cm)))
                        intensity = float(corr_flat[idx])
                        intensity_history[peak_cm].append(intensity)
                        intensity_lines[peak_cm].set_data(time_history, intensity_history[peak_cm])
                    if ax_time is not None:
                        ax_time.relim(); ax_time.autoscale_view()

                raw_data = np.vstack([raw_data, spectrum[np.newaxis, :]])
                corrected_data = np.vstack([corrected_data, corrected_spectrum[np.newaxis, :]])

                plt.pause(0.05)
                counter += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")

        plt.ioff()

        if not optimization_mode:
            os.makedirs(data_dir or ".", exist_ok=True)
            fig.savefig(os.path.join(data_dir or ".", f"{prefix}_{timestamp}_plots.png"))
            plt.close(fig)

            raw_df = pd.DataFrame(raw_data.T)
            raw_df.insert(0, 'Wavenumbers', wavenumbers)
            raw_df.insert(1, 'Background', background)

            corrected_df = pd.DataFrame(corrected_data.T)
            corrected_df.insert(0, 'Wavenumbers', processed_RS)
            corrected_df.insert(1, 'Background', np.zeros_like(processed_RS))

            raw_filename       = os.path.join(data_dir or ".", f"{prefix}_{timestamp}_raw_data.csv")
            corrected_filename = os.path.join(data_dir or ".", f"{prefix}_{timestamp}_corrected_data.csv")
            raw_df.to_csv(raw_filename, index=False)
            corrected_df.to_csv(corrected_filename, index=False)

            params = {
                'integration_time': integration_time,
                'laser_power': laser_power,
                'use_background': use_background,
                'background_file': background_file,
                'use_dark': use_dark,
                'max_num_spectra': max_num_spectra,
                'timestamp': timestamp,
                'prefix': prefix,
                'optimization_mode': optimization_mode,
                'optimization_averages': optimization_averages,
                'selected_peaks_cm': selected_peaks_cm,
                'ramanspy': {
                    'crop_region': crop_region,
                    'asls_lam': asls_lam,
                    'asls_p': asls_p,
                    'SG_window': SG_window,
                    'SG_polyorder': SG_polyorder,
                    'pipeline': ['Crop', 'ASLS', 'SavGol', 'VectorNorm(pixelwise)']
                }
            }
            with open(os.path.join(data_dir or ".", f"{prefix}_{timestamp}_params.json"), 'w') as f:
                json.dump(params, f, indent=4)

            print("Data saved successfully.")
    finally:
        try:
            spectrometer.hardware.set_laser_enable(False)
            print("Laser is OFF.")
        except Exception:
            pass
        try:
            if fig is not None and keypress_cid is not None:
                fig.canvas.mpl_disconnect(keypress_cid)
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass
        try:
            spectrometer.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()