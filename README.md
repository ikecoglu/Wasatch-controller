# Wasatch controller
Wasatch spectrometer active acquisition script.

## Overview

This repository contains a Python script for controlling a Wasatch spectrometer and collecting Raman spectra. The main script, `Acquisition.py`, connects to the instrument, acquires spectra while displaying live plots, and removes the background using routines from `utils.py`. Both raw and corrected spectra are saved to CSV files together with the acquisition parameters.

To run the script, ensure Python and the required dependencies are installed and execute:

```bash
python Acquisition.py
```

The script prompts for user input before capturing background and active spectra. Observe appropriate laser safety precautions while the instrument's laser is enabled.

