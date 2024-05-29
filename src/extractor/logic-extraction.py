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

    def traverse_directories(self):
        print(f"Starting directory traversal in {self.root_folder}")
        for dirpath, dirnames, filenames in os.walk(self.root_folder):
            print(f"Visiting {dirpath}")
            for filename in filenames:
                if filename.endswith('.ns2'):
                    file_path = os.path.join(dirpath, filename)
                    print(f"Processing file: {file_path}")
                    self.process_ns2_file(file_path, dirpath)

    def process_ns2_file(self, file_path, subdir):
        print(f"Processing {file_path} in {subdir}")
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
        
        # Generate output file path based on subdir
        relative_path = os.path.relpath(subdir, self.root_folder)
        output_dir = os.path.join(self.output_folder, relative_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
        df.to_csv(output_file_path, index=False)

        print(f"Processed {file_path} and saved to {output_file_path}")


root_folder = '../../data/sample/'
output_folder = '../../data/output-sample'
processor = NS2Processor(root_folder, output_folder)
processor.traverse_directories()