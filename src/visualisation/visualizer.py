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


    def plot_topomap(self):
        