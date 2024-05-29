import pandas as pd
import numpy as np
import os
import mne
import matplotlib.pyplot as plt


class DataVisualizer:
    def __init__(self, data, chart_type="psd", save_path=None, no_of_plots=4):
        self.data = data
        self.chart_type = chart_type
        self.save_path = save_path
        self.no_of_plots = no_of_plots

    
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
        