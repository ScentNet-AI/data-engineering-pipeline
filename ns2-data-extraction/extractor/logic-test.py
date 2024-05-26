import neo
import pandas as pandas
import numpy as num_data_points

# class to read ns2 data with neo
class NS2Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        # self.reader = neo.io.NeuroExplorerIO(filename=self.file_path)
        self.reader = neo.BlackrockIO(filename=self.file_path) 
        self.block = self.reader.read_block()
        self.segments = self.block.segments
        self.data = self.segments[0].analogsignals[0]
        self.data = self.data.magnitude
        self.data = pandas.DataFrame(self.data, columns=['data'])
        self.data['time'] = num_data_points.arange(0, len(self.data)/1000, 1/1000)
        self.data = self.data.set_index('time')

    def get_data(self):
        return self.data