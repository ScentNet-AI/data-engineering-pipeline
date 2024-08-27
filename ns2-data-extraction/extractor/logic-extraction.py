import os
import neo
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

class NS2Processor:
    def __init__(self, root_folder, output_folder):
        self.root_folder = root_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)
        print(f"Output folder created: {os.path.abspath(self.output_folder)}")  # Add this line
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename='ns2_processing.log',
                            filemode='w')
        self.logger = logging.getLogger()

    def traverse_directories(self):
        self.logger.info(f"Starting directory traversal in {self.root_folder}")
        total_ns2_files = 0
        for dirpath, _, filenames in os.walk(self.root_folder):
            self.logger.info(f"Visiting {dirpath}")
            ns2_files = [f for f in filenames if f.endswith('.ns2')]
            total_ns2_files += len(ns2_files)
            for filename in tqdm(ns2_files, desc=f"Processing files in {dirpath}"):
                file_path = os.path.join(dirpath, filename)
                self.process_ns2_file(file_path, dirpath)
        self.logger.info(f"Total .ns2 files processed: {total_ns2_files}")

    def process_ns2_file(self, file_path, subdir):
        self.logger.info(f"Processing {file_path} in {subdir}")
        try:
            reader = neo.BlackrockIO(filename=file_path)
            blk = reader.read_block()
            data = self.extract_data(blk)
            df = self.create_dataframe(data)
            self.save_dataframe(df, file_path, subdir)
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {type(e).__name__} - {str(e)}")
            # Log the full traceback for more detailed error information
            import traceback
            self.logger.error(traceback.format_exc())

    def extract_data(self, blk):
        data = [asig.magnitude[:, :32] for seg in blk.segments for asig in seg.analogsignals]
        return np.array(data)

    def create_dataframe(self, data_array):
        electrode_names = [
            'Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'Fz', 'C3', 'C4', 'Cz', 
            'P3', 'P4', 'Pz', 'O1', 'O2', 'T7', 'T8', 'P7', 'P8', 
            'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6', 'POz', 'Oz', 'AFz', 'M1', 'M2'
        ]
        num_samples, num_timepoints, num_channels = data_array.shape
        data_reshaped = data_array.reshape(num_samples * num_timepoints, num_channels)
        df = pd.DataFrame(data_reshaped, columns=electrode_names)
        df['Sample'] = np.repeat(np.arange(num_samples), num_timepoints)
        df['Time'] = np.tile(np.arange(num_timepoints) / 1000, num_samples)
        return df

    def save_dataframe(self, df, file_path, subdir):
        relative_path = os.path.relpath(subdir, self.root_folder)
        output_dir = os.path.join(self.output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.csv')
        df.to_csv(output_file_path, index=False)
        self.logger.info(f"Processed {file_path}")
        self.logger.info(f"Saved to {output_file_path}")
        self.logger.info(f"File exists: {os.path.exists(output_file_path)}")
        self.logger.info(f"File size: {os.path.getsize(output_file_path) if os.path.exists(output_file_path) else 'N/A'}")

if __name__ == "__main__":
    root_folder = os.path.abspath('Olfactory EEG data set induced by different odor types')
    output_folder = os.path.abspath('ExtractedData')
    print(f"Root folder: {root_folder}")
    print(f"Output folder: {output_folder}")
    processor = NS2Processor(root_folder, output_folder)
    processor.traverse_directories()