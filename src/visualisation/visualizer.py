import pandas as pd
import numpy as np
import os
import mne
import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self, data, chart_type="psd", sampling_rate=512, save_path=None, no_of_plots=4, display=True):
        self.data = data
        self.channels = data.columns[:]
        self.chart_type = chart_type
        self.save_path = save_path
        self.no_of_plots = no_of_plots
        self.sampling_rate = sampling_rate
        self.display = display

    
    def plot(self):
        if self.chart_type == "psd":
            self.plot_psd()
        elif self.chart_type == "time":
            self.plot_time()
        elif self.chart_type == "topomap":
            self.plot_topomap()
        else:
            print("Invalid chart type")
        
        self.save_plot()

    
    def plot_psd(self):
        fft_output = np.fft.fft(self.data[self.channels[0]][:self.sampling_rate])  # Use one second of data
        frequencies = np.fft.fftfreq(self.sampling_rate, 1/self.sampling_rate)
        plt.figure(figsize=(10, 5))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_output)[:len(frequencies)//2])

        # Plot the frequency spectrum
        if self.display:
            plt.title('Frequency Spectrum of EEG')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.show()
        
        else:
            pass
        
    
    def plot_time(self):
        # Plot the first few seconds of EEG data for the first few channels
        plt.figure(figsize=(15, 5))
        for channel in self.channels[:5]:  # Adjust the slice for more/less channels
            plt.plot(self.data[channel][:self.sampling_rate * 2], label=channel)  # Plotting first 2 seconds

        if self.display:
            plt.title('EEG Time Series Plot')
            plt.xlabel('Time (milliseconds)')
            plt.ylabel('Amplitude (uV)')
            plt.legend()
            plt.show()

        else: 
            pass


    def psd_plot(self, channel_map):
        if isinstance(self.data, np.ndarray):
            data = pd.DataFrame(self.data)

        # Select only the columns that are actually mapped (ignores unmapped channels)
        channels_to_use = [channel for channel in channel_map.values() if channel in data.columns]

        # Create an MNE Info object with the channels that we have data for
        info = mne.create_info(ch_names=channels_to_use, sfreq=1000, ch_types='eeg')

        # Create the RawArray with the data from the channels we are using
        raw = mne.io.RawArray(data[channels_to_use].T.to_numpy(), info)

        # Define the montage (electrode positions)
        montage = mne.channels.make_standard_montage('standard_1020')

        # Set the montage, ignoring channels that are not present in the montage
        raw.set_montage(montage, on_missing='ignore')
        
        if self.display:
            # Visualize the data
            raw.plot_psd(fmax=50)  # Showing PSD up to 50 Hz
            # raw.plot_projs_topomap()  # Topomap