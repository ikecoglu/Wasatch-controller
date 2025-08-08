import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from wasatch.WasatchBus import WasatchBus
from wasatch.WasatchDevice import WasatchDevice
import matplotlib.pyplot as plt
import utils
import threading
import json

##### Acquisition parameters #####
prefix              = "" # Prefix for file names
integration_time    = 1 # Seconds
laser_power         = 450 # mW
max_num_spectra     = 1000  # Maximum number of spectra to acquire
use_background      = False  # Use background subtraction.
background_file     = ""  # Path to background file (if use_background is True and you want to load from file)

##### Spectrum correction parameters #####
poly_order          = 5  # Polynomial order for baseline fitting
max_iter            = 100  # Maximum number of iterations for background removal
crop_range          = (300, 2300)  # Crop range for the spectrum (in cm^-1). Set to (0, 3500) to disable cropping.

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

wavenumbers = np.array(spectrometer.settings.wavenumbers)
if wavenumbers is None:
    print("Wavenumbers not available. Ensure the spectrometer is configured correctly.")
    sys.exit(1)

# Turn on the laser
spectrometer.hardware.set_laser_enable(True)
print("WARNING: Laser is ON. Ensure safety precautions are followed.")

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
        spectrum = spectrometer.hardware.get_line().data.spectrum
        background.append(spectrum)
    background = np.mean(background, axis=0)
    print("Background spectrum acquired and averaged.")
else:
    background = np.zeros(spectrometer.settings.pixels())
    print("Background subtraction is disabled. Using zero background.")

# Active spectrum acquisition
user_input = input("Press Enter to start active spectrum acquisition...")

plt.ion()
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 5))
fig_manager = plt.get_current_fig_manager()
fig_manager.full_screen_toggle()
plt.tight_layout(pad=3)

# Left plot: raw, baseline + background
line_raw, = ax_left.plot([], [], label='Raw Spectrum', color='blue')
if use_background:
    line_baseline, = ax_left.plot([], [], label='Baseline + Background', color='green')
    ax_left.set_title('Raw, Background, Baseline')
else:
    line_baseline, = ax_left.plot([], [], label='Baseline', color='green')
    ax_left.set_title('Raw, Baseline')
ax_left.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_left.set_ylabel('Intensity (a.u.)')
ax_left.legend()

# Right plot: corrected spectrum
line_corr, = ax_right.plot([], [], label='Corrected Spectrum', color='red')
ax_right.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_right.set_ylabel('Intensity (a.u.)')
ax_right.legend()
ax_right.set_title('Corrected Spectrum')

stop_plotting = False

def wait_for_input():
    global stop_plotting
    input("Press Enter in terminal to stop plotting...\n")
    stop_plotting = True

# Start a thread to wait for user input to stop plotting
input_thread = threading.Thread(target=wait_for_input)
input_thread.daemon = True
input_thread.start()

# Initialize data arrays
raw_data = np.array([]).reshape(0, spectrometer.settings.pixels())
corrected_data = np.array([]).reshape(0, spectrometer.settings.pixels())

# Crop the wavenumbers and background
crop_mask = (wavenumbers >= crop_range[0]) & (wavenumbers <= crop_range[1])
cropped_wavenumbers = wavenumbers[crop_mask]
corrected_data = corrected_data[:, crop_mask]
cropped_background = background[crop_mask]

# Acquire spectra
counter = 0
while not stop_plotting and counter < max_num_spectra:
    spectrum = np.array(spectrometer.hardware.get_line().data.spectrum)
    raw_data = np.vstack([raw_data, spectrum]) if raw_data.size else spectrum

    cropped_spectrum = spectrum[crop_mask]
    corrected_spectrum = utils.remove_background(cropped_wavenumbers, cropped_spectrum, cropped_background, poly_order, max_iter, eps=0.1)
    baseline = cropped_spectrum.flatten() - corrected_spectrum.flatten()
    corrected_data = np.vstack([corrected_data, corrected_spectrum]) if corrected_data.size else corrected_spectrum
    
    # Update left plot
    line_raw.set_data(wavenumbers, spectrum.flatten())
    line_baseline.set_data(cropped_wavenumbers, baseline)

    ax_left.set_title(f'Raw Spectrum {counter + 1}')
    ax_left.relim()
    ax_left.autoscale_view()

    # Update right plot
    line_corr.set_data(cropped_wavenumbers, corrected_spectrum.flatten())
    ax_right.set_title(f'Corrected Spectrum {counter + 1}')
    ax_right.relim()
    ax_right.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.05)
    counter += 1

plt.ioff()
plt.show()

# Turn off the laser
spectrometer.hardware.set_laser_enable(False)
print("Laser is OFF.")

# Create dataframes
raw_df = pd.DataFrame(raw_data.T)
corrected_df = pd.DataFrame(corrected_data.T)
raw_df.insert(0, 'Wavenumbers', wavenumbers)
corrected_df.insert(0, 'Wavenumbers', cropped_wavenumbers)
raw_df.insert(1, 'Background', background)
corrected_df.insert(1, 'Background', cropped_background)

# Save data to CSV files with timestamp and prefix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_filename = f"{prefix}_{timestamp}_raw_data.csv"
corrected_filename = f"{prefix}_{timestamp}_corrected_data.csv"

raw_df.to_csv(raw_filename, index=False)
corrected_df.to_csv(corrected_filename, index=False)

# Save the parameters used for acquisition
params = {
    'integration_time': integration_time,
    'laser_power': laser_power,
    'poly_order': poly_order,
    'max_iter': max_iter,
    'max_num_spectra': max_num_spectra,
    'timestamp': timestamp,
    'prefix': prefix
}
params_filename = f"{prefix}_{timestamp}_params.json"
with open(params_filename, 'w') as f:
    json.dump(params, f, indent=4)

print(f"Data saved to {raw_filename} and {corrected_filename}")
print(f"Acquisition parameters saved to {params_filename}")
print("Acquisition complete.")