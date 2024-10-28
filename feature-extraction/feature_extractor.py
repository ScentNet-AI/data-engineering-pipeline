import os
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset
from scipy.signal import welch
from multiprocessing import Pool, Manager, cpu_count
import psutil
from tqdm import tqdm
import json
from datetime import datetime


class OlfactoryEEGFeatureDataset(Dataset):
    def __init__(
        self,
        root_dir,
        output_dir,
        sfreq=1000,
        window_size=1000,
        overlap=0.5,
        n_jobs=None,
        batch_size=100,
        memory_threshold=0.8,
        verbose=True
    ):
        """
        Initializes the OlfactoryEEGFeatureDataset.

        Args:
            root_dir (str): Path to the root directory containing raw EEG data.
            output_dir (str): Path to the directory where processed data will be saved.
            sfreq (int, optional): Sampling frequency. Defaults to 1000.
            window_size (int, optional): Size of the window for feature extraction. Defaults to 1000.
            overlap (float, optional): Overlap between windows (0 to 1). Defaults to 0.5.
            n_jobs (int, optional): Number of parallel processes. Defaults to number of CPU cores.
            batch_size (int, optional): Number of samples to process before flushing to disk. Defaults to 100.
            memory_threshold (float, optional): Memory usage threshold to trigger flushing (0 to 1). Defaults to 0.8.
            verbose (bool, optional): Whether to print detailed logs. Defaults to True.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.sfreq = sfreq
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        self.verbose = verbose

        self.n_jobs = n_jobs if n_jobs is not None else cpu_count()

        # Initialize shared lists for multiprocessing
        manager = Manager()
        self.samples = manager.list()
        self.labels = manager.list()
        self.subject_ids = manager.list()

        # Tracking variables
        self.file_count = 0
        self.progress_file = os.path.join(self.output_dir, 'progress.json')
        self.log_file = os.path.join(self.output_dir, 'processing_log.txt')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Start processing
        self._load_and_extract_features()

    def _load_and_extract_features(self):
        """
        Loads and processes EEG data from all subjects.
        """
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Root directory '{self.root_dir}' not found")

        subject_folders = [
            folder for folder in os.listdir(self.root_dir) if folder.startswith("Sub.")
        ]

        # Load progress to skip already processed subjects
        processed_subjects = self._load_progress()
        subjects_to_process = [
            subject for subject in subject_folders if subject not in processed_subjects
        ]

        if self.verbose:
            print(f"Starting processing with {self.n_jobs} parallel jobs.")
            print(f"Total subjects to process: {len(subjects_to_process)}")

        with Pool(processes=self.n_jobs) as pool:
            for _ in tqdm(
                pool.imap_unordered(self._process_subject, subjects_to_process),
                total=len(subjects_to_process),
                desc="Processing Subjects"
            ):
                pass
            
        # Final flush of any remaining data
        self._flush_data()

        if self.verbose:
            print("Processing complete.")

    def _process_subject(self, subject_folder):
        """
        Processes all EEG data files for a single subject.

        Args:
            subject_folder (str): Name of the subject folder.
        """
        subject_path = os.path.join(self.root_dir, subject_folder)
        if self.verbose:
            print(f"Processing subject: {subject_folder}")

        for smell_label in 'ABCDEFGHIJKLM':
            smell_path = os.path.join(subject_path, smell_label)
            if os.path.isdir(smell_path):
                if self.verbose:
                    print(f"  Processing smell: {smell_label}")
                for file_name in os.listdir(smell_path):
                    if file_name.startswith("datafile") and file_name.endswith(".csv"):
                        file_path = os.path.join(smell_path, file_name)
                        if self.verbose:
                            print(f"    Processing file: {file_name}")
                        try:
                            features, labels = self._process_file(
                                file_path, smell_label, subject_folder
                            )
                            if features is not None:
                                self.samples.extend(features)
                                self.labels.extend(labels)
                                self.subject_ids.extend([subject_folder] * len(features))

                                # Check memory usage and flush if necessary
                                if (
                                    self._check_memory_usage()
                                    or len(self.samples) >= self.batch_size
                                ):
                                    self._flush_data()

                        except Exception as e:
                            error_message = (
                                f"Error processing {file_path}: {str(e)}"
                            )
                            self._log_error(error_message)
                            if self.verbose:
                                print(error_message)

        # Update progress after processing the subject
        self._update_progress(subject_folder)

    def _process_file(self, file_path, smell_label, subject_folder):
        """
        Processes a single EEG data file.

        Args:
            file_path (str): Path to the EEG data file.
            smell_label (str): Label indicating the smell stimulus.
            subject_folder (str): Subject identifier.

        Returns:
            tuple: Features and labels extracted from the EEG data.
        """
        df = pd.read_csv(file_path)

        if df.empty:
            if self.verbose:
                print(f"    Empty dataframe for file: {file_path}")
            return None, None

        raw = self._preprocess_eeg(df)
        features = self._extract_features(raw)

        labels = [ord(smell_label) - ord('A')] * len(features)
        return features, labels

    def _preprocess_eeg(self, df):
        """
        Preprocesses raw EEG data.

        Args:
            df (pd.DataFrame): Raw EEG data.

        Returns:
            mne.io.Raw: Preprocessed EEG data.
        """
        ch_names = df.columns[:-4].tolist()  # Exclude non-EEG columns
        data = df[ch_names].T.values
        info = mne.create_info(
            ch_names=ch_names, sfreq=self.sfreq, ch_types='eeg'
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        # Bandpass filter
        raw.filter(l_freq=1, h_freq=50, method='iir', verbose=False)
        # Notch filter to remove powerline noise
        raw.notch_filter(freqs=60, method='iir', verbose=False)

        # Independent Component Analysis (ICA) for artifact removal
        ica = mne.preprocessing.ICA(
            n_components=20, random_state=97, max_iter=200, verbose=False
        )
        ica.fit(raw, verbose=False)
        raw = ica.apply(raw, verbose=False)

        # Set average reference
        raw.set_eeg_reference('average', projection=True, verbose=False)
        raw.apply_proj(verbose=False)

        return raw

    def _extract_features(self, raw):
        """
        Extracts statistical and frequency domain features from preprocessed EEG data.

        Args:
            raw (mne.io.Raw): Preprocessed EEG data.

        Returns:
            list: List of feature vectors extracted from EEG data.
        """
        data = raw.get_data()
        windows = self._create_windows(data)

        features = []
        for window in windows:
            # Time-domain features
            mean = np.mean(window, axis=1)
            std = np.std(window, axis=1)

            # Frequency-domain features using Welch's method
            freqs, psd = welch(
                window,
                fs=self.sfreq,
                nperseg=self.window_size,
                axis=1
            )

            # Band power features
            delta_band = self._band_power(psd, freqs, 1, 4)
            theta_band = self._band_power(psd, freqs, 4, 8)
            alpha_band = self._band_power(psd, freqs, 8, 13)
            beta_band = self._band_power(psd, freqs, 13, 30)
            gamma_band = self._band_power(psd, freqs, 30, 50)

            # Concatenate all features
            window_features = np.concatenate([
                mean,
                std,
                delta_band,
                theta_band,
                alpha_band,
                beta_band,
                gamma_band
            ])
            features.append(window_features)

        return features

    def _band_power(self, psd, freqs, fmin, fmax):
        """
        Calculates the average power in a specific frequency band.

        Args:
            psd (ndarray): Power spectral density.
            freqs (ndarray): Frequency values corresponding to PSD.
            fmin (float): Lower frequency bound.
            fmax (float): Upper frequency bound.

        Returns:
            ndarray: Average power in the specified frequency band for each channel.
        """
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.mean(psd[:, freq_mask], axis=1)
        return band_power

    def _create_windows(self, data):
        """
        Creates overlapping windows from EEG data.

        Args:
            data (ndarray): EEG data array of shape (n_channels, n_samples).

        Returns:
            list: List of windowed data arrays.
        """
        step = int(self.window_size * (1 - self.overlap))
        n_samples = data.shape[1]
        windows = [
            data[:, start:start + self.window_size]
            for start in range(0, n_samples - self.window_size + 1, step)
        ]
        return windows

    def _flush_data(self):
        """
        Saves accumulated data to disk and clears the buffers.
        """
        if len(self.samples) == 0:
            return

        samples_array = np.array(self.samples)
        labels_array = np.array(self.labels)
        subject_ids_array = np.array(self.subject_ids)

        # Create a DataFrame for saving
        df = pd.DataFrame(samples_array)
        df['Label'] = labels_array
        df['Subject_ID'] = subject_ids_array

        # Define output file path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(
            self.output_dir,
            f"processed_data_part_{self.file_count}_{timestamp}.csv"
        )

        # Save to CSV
        df.to_csv(output_file, index=False)
        if self.verbose:
            print(f"Saved {len(self.samples)} samples to {output_file}")

        # Clear buffers
        self.samples[:] = []
        self.labels[:] = []
        self.subject_ids[:] = []
        self.file_count += 1

    def _check_memory_usage(self):
        """
        Checks the current memory usage of the system.

        Returns:
            bool: True if memory usage exceeds the threshold, False otherwise.
        """
        memory_usage = psutil.virtual_memory().percent / 100
        return memory_usage >= self.memory_threshold

    def _load_progress(self):
        """
        Loads processing progress from a JSON file.

        Returns:
            list: List of subjects that have been processed.
        """
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                processed_subjects = json.load(f)
            return processed_subjects
        return []

    def _update_progress(self, subject):
        """
        Updates the progress file with a newly processed subject.

        Args:
            subject (str): Subject identifier to add to progress.
        """
        processed_subjects = self._load_progress()
        if subject not in processed_subjects:
            processed_subjects.append(subject)
            with open(self.progress_file, 'w') as f:
                json.dump(processed_subjects, f)

    def _log_error(self, message):
        """
        Logs error messages to a log file.

        Args:
            message (str): Error message to log.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

    def __len__(self):
        """
        Returns:
            int: Total number of samples processed and loaded in memory.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single sample by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (sample_features, label, subject_id)
        """
        sample = torch.FloatTensor(self.samples[idx])
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        return sample, label, subject_id


if __name__ == "__main__":
    # Example usage
    dataset = OlfactoryEEGFeatureDataset(
        root_dir='path/to/ExtractedData',
        output_dir='path/to/output',
        sfreq=1000,
        window_size=1000,
        overlap=0.5,
        n_jobs=4,
        batch_size=500,
        memory_threshold=0.75,
        verbose=True
    )
