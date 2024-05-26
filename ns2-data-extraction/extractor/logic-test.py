import os
import neo
import pandas as pd
import numpy as np

# class to read ns2 data with neo
class NS2Processor:
    def __init__(self, root_folder, output_folder):
        self.root_folder = root_folder
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_ns2_file(self, file_path):
        reader = neo.BlackrockIO(filename=file_path)
        blk = reader.read_block()

        data = []
        for seg in blk.segments:
            for asig in seg.analogsignals:
                data.append(asig.magnitude.flatten())

        # Assuming the number of channels is 64
        num_channels = 64
        num_data_points = len(data[0])  # Assuming all data arrays are the same length
        num_samples = num_data_points // num_channels

        # Reshape the data
        reshaped_data = np.array(data[0]).reshape((num_samples, num_channels))

        df = pd.DataFrame(reshaped_data)
        output_path = os.path.join(self.output_folder, os.path.basename(file_path).replace('.ns2', '.csv'))
        df.to_csv(output_path, index=False)

        print(f"Processed {file_path} and saved to {output_path}")

    def traverse_directories(self):
        for dirpath, dirnames, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith('.ns2'):
                    file_path = os.path.join(dirpath, filename)
                    self.process_ns2_file(file_path)
    
# Example usage:
root_folder = '../data/root-datafiles'
output_folder = '../data/output-datafiles'
processor = NS2Processor(root_folder, output_folder)
processor.traverse_directories()