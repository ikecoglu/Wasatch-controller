# Wasatch controller
Active Wasatch spectrometer acquisition script with live plotting and Ramanspy-based preprocessing.

## Overview

`Acquisition.py` connects to the first available Wasatch spectrometer, configures integration time and laser power, and streams spectra while plotting both the raw signal and a processed version. The preprocessing pipeline is built with Ramanspy:

1. Crop to a configurable Raman-shift window.
2. Apply ASLS baseline correction and capture the estimated baseline.
3. Despike (Whitaker-Hayes), Savitzky-Golay smooth, and pixelwise vector-normalize.

Both raw and corrected spectra are buffered and saved, together with the plots and acquisition parameters. Optional dark and background measurements can be subtracted before the Ramanspy pipeline. An optimization mode keeps the plots and time-trace running without writing output files, ideal for alignment/tuning.

## Setup

Create (or update) the conda environment declared in `environment.yml`:

```bash
conda env create -f environment.yml          # first time
# or
conda env update -f environment.yml --prune  # update an existing env

conda activate wasatch-controller
```

The environment installs Python 3.12, NumPy, Pandas, SciPy, Matplotlib, the `wasatch` SDK, and Ramanspy along with their dependencies.

## Running

After connecting the spectrometer and activating the environment, run:

```bash
python Acquisition.py
```

The script guides you through optional background/dark measurements and waits for confirmation before starting live acquisition. Observe standard laser safety practices whenever the laser is enabled.

While running:
- Press Enter in the terminal to stop (when stdin is interactive).
- Press `q` or `Esc` inside the Matplotlib window to stop.

The UI closes automatically when acquisition ends or an exception occurs.

## Configuration

Key parameters (all editable near the top of `Acquisition.py`):

- `optimization_mode`: `True` to keep the laser on and show plots/time traces without saving data.
- `data_dir`: Output directory (defaults to current folder).
- `prefix`: Filename prefix for saved artifacts.
- `integration_time`: Integration time in seconds.
- `laser_power`: Laser power setpoint in mW.
- `max_num_spectra`: Maximum spectra to acquire; `None` means run until stopped.
- `use_dark`: Acquire a dark spectrum (laser off) and subtract it from every frame.
- `use_background`: Subtract a provided/acquired background before Ramanspy processing.
- `background_file`: CSV with background spectra; if empty and `use_background=True`, three backgrounds are acquired automatically.
- `optimization_averages`: Number of frames to average per update when optimization mode is enabled.
- `selected_peaks_cm`: List of Raman shifts (cm⁻¹) to track over time in optimization mode.
- `crop_region`: `(min_cm1, max_cm1)` tuple used by the Ramanspy cropper.
- `asls_lam`, `asls_p`: Parameters for Ramanspy’s ASLS baseline correction.
- `SG_window`, `SG_polyorder`: Savitzky-Golay smoothing configuration.

## Outputs

When `optimization_mode` is `False`, the script saves to `data_dir`:

- `<prefix>_<timestamp>_plots.png`: Snapshot of the final Matplotlib window.
- `<prefix>_<timestamp>_raw_data.csv`: Raw spectra (columns per acquisition) with `Wavenumbers` and `Background` columns prepended.
- `<prefix>_<timestamp>_corrected_data.csv`: Processed spectra on the cropped axis with placeholder background column.
- `<prefix>_<timestamp>_params.json`: All acquisition and preprocessing parameters used for the run.

The `.gitignore` excludes these files by default; adjust if you need to version-control outputs.

## Background CSV format

If supplying `background_file`, use a CSV that contains numeric spectra shaped either `(pixels, samples)` or `(samples, pixels)`. Non-numeric columns (e.g., `Wavenumbers`, `Background`) are ignored automatically. The script averages the numeric columns/rows to derive a single background vector. When no file is provided, three background spectra are captured (after dark subtraction) and averaged.

## Optimization mode

Enable `optimization_mode=True` and optionally list `selected_peaks_cm` to monitor specific Raman shifts. The UI switches to a three-panel layout:

- Top-left: Corrected spectrum with optional peak markers.
- Bottom-left: Raw spectrum + baseline/background trace.
- Right: Time-trace of selected peak intensities.

Data are continually refreshed until stopped, but no files are written in this mode.

## Notes

- The script validates spectrometer connectivity, EEPROM info, and wavenumbers before acquisition.
- Live plotting uses Matplotlib’s interactive mode; `plt.pause(0.05)` keeps the UI responsive.
- The laser is always disabled and the spectrometer disconnected in a `finally` block to avoid leaving hardware in an unsafe state.
