# Wasatch controller
Wasatch spectrometer active acquisition script.

## Overview

This repository contains a Python script for controlling a Wasatch spectrometer and collecting Raman spectra. The main script, `Acquisition.py`, connects to the instrument, acquires spectra while displaying live plots, and removes the background using routines from `utils.py`. Both raw and corrected spectra are saved to CSV files together with the acquisition parameters and a PNG of the plots.

### Setup

Create the conda environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate wasatch-controller
```

The `environment.yml` installs NumPy, Pandas, SciPy, Matplotlib, and the `wasatch` Python package.

### Running

After activating the environment, execute:

```bash
python Acquisition.py
```

The script prompts for user input before capturing background and active spectra. Observe appropriate laser safety precautions while the instrument's laser is enabled.

While running:
- Press Enter in the terminal to stop acquisition.
- Or press `q` or `Esc` in the plot window to stop.

The plot window closes automatically when acquisition ends or if the program is terminated.

### Configuration

Edit the parameters at the top of `Acquisition.py` to suit your setup:

- `data_dir`: Directory for saving outputs (created if missing).
- `prefix`: Prefix added to all output filenames.
- `integration_time`: Integration time in seconds.
- `laser_power`: Laser power in mW.
- `max_num_spectra`: Maximum spectra to acquire; set to an integer or `None` for continuous until stopped.
- `use_dark`: If `True`, acquires a dark spectrum with laser off and subtracts it.
- `use_background`: If `True`, applies background plus baseline correction using `utils.remove_background`.
- `background_file`: Optional CSV to load a background; if empty and `use_background=True`, the script acquires and averages 3 background spectra.
- `poly_order`: Polynomial order used in baseline modeling during correction.
- `max_iter`: Maximum iterations for background removal.
- `crop_range`: Tuple `(min_cm1, max_cm1)` to crop the spectral axis; set to `None` to disable.

### Outputs

Files are saved to `data_dir` with a timestamp and the provided prefix:

- `<prefix>_<timestamp>_plots.png`: PNG snapshot of the final plots.
- `<prefix>_<timestamp>_raw_data.csv`: Raw spectra (columns per acquired spectrum) with `Wavenumbers` and `Background` columns prepended.
- `<prefix>_<timestamp>_corrected_data.csv`: Corrected spectra (columns per acquired spectrum) with `Wavenumbers` and `Background` columns prepended (cropped if `crop_range` is set).
- `<prefix>_<timestamp>_params.json`: Acquisition parameters used for the run.

Note: The repository `.gitignore` is configured to ignore these generated outputs. Remove the patterns in `.gitignore` if you prefer to commit them.

### Background CSV format (optional)

If `use_background=True` and `background_file` is provided, the file should be a CSV containing:

- A `Wavenumbers` column matching the instrument axis.
- A `Background` column (optional).
- One or more additional columns with background spectra to be averaged into a single background vector.

If no file is provided, the script acquires three background spectra (with dark subtraction) and averages them.

### Notes

- The script verifies spectrometer connectivity and wavenumber availability before starting.
- Live plotting uses `plt.pause` to keep the UI responsive; keyboard shortcuts are supported for stopping.
- The laser is automatically disabled and the device disconnected on exit.
