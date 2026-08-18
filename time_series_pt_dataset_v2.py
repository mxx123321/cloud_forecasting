# 1. Standard library imports
import os
import re
from datetime import datetime, timedelta

# 2. Third-party library imports
import torch
from torch.utils.data import Dataset

def get_first_timestamp(filepath):
    # 1. Extract only the filename and ignore directory names such as /202403/.
    filename = os.path.basename(filepath)
    
    # 2. Search for the first sequence of 14 consecutive digits.
    # re.search scans from left to right and stops at the first match.
    match = re.search(r'(\d{14})', filename)
    
    if match:
        # group(1) returns the first matched timestamp string.
        return datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
    else:
        return datetime.min
    
def write_to_txt(data, output_file):
    # Open the output file in write mode.
    with open(output_file, 'w') as f:
        count = 0  # Count the number of written elements.
        
        # Iterate over each row in the 2D list.
        for row in data:
            for value in row:
                # Write the current value.
                f.write(f"{value}\n")
                count += 1
                
                # Insert an extra newline after every 10 values.
                if count % 10 == 0:
                    f.write("\n")

# Convert timestamps to datetime objects.
def convert_to_datetime(timestamp):
    timestamp_str = str(timestamp)
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M")

# Check whether adjacent timestamps are separated by 15 minutes.
def check_time_interval(time_list):
    for i in range(1, len(time_list)):
        # Get two adjacent timestamps.
        prev_time = convert_to_datetime(time_list[i - 1])
        curr_time = convert_to_datetime(time_list[i])

        # Compute the time difference.
        time_diff = curr_time - prev_time

        # Return False if the interval is not 15 minutes.
        if time_diff != timedelta(minutes=15):
            return False
    return True  # All adjacent intervals are 15 minutes.

class CloudMaskSequenceDataset(Dataset):
    def __init__(self, directory, num_input=5, num_output=5, time_difference=15, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15,\
        dataset_type='train',dataset_total=1):
        """
        :param directory: Directory containing the .pt files.
        :param num_input: Number of input timestamps (e.g., 5).
        :param num_output: Number of target timestamps to predict (e.g., 5).
        :param time_difference: Required temporal interval, defaulting to 15 minutes.
        """
        self.num_input = num_input
        self.num_output = num_output
        self.time_difference = time_difference  # Set the required temporal interval (15 minutes by default).
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.dataset_type = dataset_type
        self.dataset_total_size = dataset_total
        #self.valid_data_name = []
        
        self.valid_data = []
        self.file_paths = self.get_pt_files(directory)
        self.get_filtered_pt_files(directory)
        # Split the dataset into training, validation, and test subsets.
        self.valid_data_after1 = self.valid_data[:self.dataset_total(self.valid_data)]
        self.train_size, self.valid_size = self.split_data(self.valid_data_after1)
        
        
        # Select the requested dataset split.
        if self.dataset_type   == 'train':
            self.valid_data_after2 = self.valid_data_after1[:self.train_size]
        elif self.dataset_type == 'val':
            self.valid_data_after2 = self.valid_data_after1[self.train_size:self.train_size + self.valid_size]
        elif self.dataset_type == 'test':
            self.valid_data_after2 = self.valid_data_after1[self.train_size + self.valid_size:]
        
    def dataset_total(self, data):
        total_size = len(data)
        dataset_total_size_process = int(self.dataset_total_size * total_size)
        return dataset_total_size_process

    def split_data(self, data):
        """
        Split the dataset according to the specified ratios.
        """
        total_size = len(data)
        train_size = int(self.train_ratio * total_size)
        valid_size = int(self.valid_ratio * total_size)



        return train_size, valid_size
    def get_pt_files(self, directory):
        """
        Collect all .pt file paths and sort them by timestamp.
        """
        pt_files = []

        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.pt'):
                    file_path = os.path.join(root, filename)
                    pt_files.append(file_path)

        # Sort by the first timestamp found in each filename.
        pt_files.sort(key=get_first_timestamp)

        #"/root/autodl-tmp/cropped_images_128_zip/Dongjing/Dongjing/202206/cropped_FY4B-_AGRI--_N_DISK_1330E_L2-_CLM-_MULT_NOM_20220601000000_20220601001459_4000M_V0001.pt"
        return pt_files
    def get_filtered_pt_files(self, directory):
        """
        Filter out candidate sequences that do not satisfy the required temporal interval.
        """
        pt_files = self.get_pt_files(directory)

        # Retain only sequences with the required temporal continuity.
        for idx in range(len(pt_files) - self.num_input - self.num_output + 1):
            input_name = []
            output_name = []

            # Use num_input consecutive frames as the historical input sequence.
            for i in range(idx, idx + self.num_input):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    extracted_number = match.group(1)[:-2]  # Remove the last two digits to convert the timestamp to minute precision.
                    extracted_number = int(extracted_number)
                    #print(extracted_number)
                    input_name.append(extracted_number)

            # Use the following num_output frames as the prediction targets.
            for i in range(idx + self.num_input, idx + self.num_input + self.num_output):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    extracted_number = match.group(1)[:-2]
                    extracted_number = int(extracted_number)
                    output_name.append(extracted_number)

            # Convert timestamps to datetime objects.
            time_objects_input = [datetime.strptime(str(time), "%Y%m%d%H%M") for time in input_name]
            time_differences_input = [time_objects_input[i + 1] - time_objects_input[i] for i in range(len(time_objects_input) - 1)]

            time_objects_output = [datetime.strptime(str(time), "%Y%m%d%H%M") for time in output_name]
            #print(time_objects_output)
            list_all = time_objects_input + time_objects_output
            list_difference_all = [list_all[i + 1] - list_all[i] for i in range(len(list_all) - 1)]
            
            
            time_differences_output = [time_objects_output[i + 1] - time_objects_output[i] for i in range(len(time_objects_output) - 1)]
            #print(time_differences_output)
            # Verify that all adjacent frames are separated by the required 15-minute interval.
            
            if all(diff.total_seconds() == 900 for diff in list_difference_all):
                
                
                self.valid_data.append(pt_files[idx:idx+self.num_input + self.num_output])
                #print(len(pt_files[idx:idx+self.num_input + self.num_output]))
                #self.valid_data_name.append(pt_files[idx:idx+self.num_input + self.num_output])
            # self.no_valid_data.append(pt_files[idx:idx+self.num_input + self.num_output])
                # Optionally write the sorted file paths to a text file.
    def __len__(self):
        # Return the number of valid temporal sequences.
        #return len(self.file_paths) - self.num_input - self.num_output + 1
        return len(self.valid_data_after2)
    # Updated __getitem__ implementation for time_series_pt_dataset_v2.py.
    def __getitem__(self, idx):
        data_list = []
        name_list = []
        
        for file_path in self.valid_data_after2[idx]:
            # 1. Load the data.
            data = torch.load(file_path, weights_only=False)
            
            # 2. Convert to a tensor and handle invalid values.
            # Convert to float32 (or int16) first to avoid uint8 overflow.
            data = torch.as_tensor(data, dtype=torch.float32) 
            
            # Map all invalid values greater than 3 (e.g., values around 120) to -1.
            data[data > 3] = -1.0 
            
            data_list.append(data)
            
            # Preserve the original timestamp extraction for name_list.
            match = re.search(r'NOM_(\d+)_', file_path)
            if match:
                extracted_number = int(match.group(1)[:-2])
                name_list.append(extracted_number)
            else:
                name_list.append('None')
                
        input_data  = torch.stack(data_list[:self.num_input]) 
        output_data = torch.stack(data_list[self.num_input:])  
        
        return input_data, output_data, name_list


class CloudMaskSequenceDataset_Fixed_Month(Dataset):
    def __init__(self, base_directory, months=['202403', '202404'], num_input=5, num_output=5, 
                 time_difference=15, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15,
                 dataset_type='train', dataset_total=1.0):
        """
        :param base_directory: Base directory, e.g., '/.../ChangChun_Fixed_1024/'.
        :param months: List of month subdirectories to load, e.g., ['202403', '202404'].
        """
        self.num_input = num_input
        self.num_output = num_output
        self.time_difference = time_difference
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.dataset_type = dataset_type
        self.dataset_total_size = dataset_total
        
        self.valid_data = []
        # --- Update 1: Load data according to base_directory and the selected months. ---
        self.file_paths = self.get_pt_files_by_months(base_directory, months)
        
        # Filter sequences that satisfy the temporal-interval requirement.
        self.get_filtered_pt_files_from_list(self.file_paths)
        
        # Select the requested dataset split.集逻辑保持不变
        total_len = self.dataset_total_calc(self.valid_data)
        self.valid_data_after1 = self.valid_data[:total_len]
        self.train_size, self.valid_size = self.split_data(self.valid_data_after1)
        
        if self.dataset_type == 'train':
            self.valid_data_after2 = self.valid_data_after1[:self.train_size]
        elif self.dataset_type == 'val':
            self.valid_data_after2 = self.valid_data_after1[self.train_size:self.train_size + self.valid_size]
        elif self.dataset_type == 'test':
            self.valid_data_after2 = self.valid_data_after1[self.train_size + self.valid_size:]

    def get_pt_files_by_months(self, base_dir, months):
        """
        --- Update 2: Load files only from the specified month subdirectories. ---
        """
        pt_files = []
        for month in months:
            # Construct the month-specific path, e.g., /.../ChangChun_Fixed_1024/202403/.
            month_path = os.path.join(base_dir, month)
            
            if not os.path.exists(month_path):
                print(f"Warning: path does not exist: {month_path}")
                continue
                
            for root, _, files in os.walk(month_path):
                for filename in files:
                    if filename.endswith('.pt'):
                        pt_files.append(os.path.join(root, filename))

        # Sort files chronologically so that temporal-continuity checks are valid.
        # Assumes get_first_timestamp is defined above.
        pt_files.sort(key=get_first_timestamp) 
        return pt_files

    def get_filtered_pt_files_from_list(self, pt_files):
        """
        --- Update 3: Process the provided file list directly using the same filtering logic. ---
        """
        for idx in range(len(pt_files) - self.num_input - self.num_output + 1):
            input_name = []
            output_name = []

            # Extract timestamps for the input and target sequences.
            for i in range(idx, idx + self.num_input + self.num_output):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    # Convert timestamps to minute precision.
                    ts = int(match.group(1)[:-2])
                    if i < idx + self.num_input:
                        input_name.append(ts)
                    else:
                        output_name.append(ts)

            # Validate temporal continuity.
            all_timestamps = input_name + output_name
            time_objects = [datetime.strptime(str(ts), "%Y%m%d%H%M") for ts in all_timestamps]
            
            # Check whether every adjacent pair is separated by 15 minutes (900 seconds).
            is_continuous = True
            for i in range(len(time_objects) - 1):
                if (time_objects[i+1] - time_objects[i]).total_seconds() != 900:
                    is_continuous = False
                    break
            
            if is_continuous:
                self.valid_data.append(pt_files[idx : idx + self.num_input + self.num_output])

    def dataset_total_calc(self, data):
        return int(self.dataset_total_size * len(data))

    def split_data(self, data):
        total_size = len(data)
        return int(self.train_ratio * total_size), int(self.valid_ratio * total_size)

    def __len__(self):
        return len(self.valid_data_after2)

    def __getitem__(self, idx):
        data_list = []
        name_list = []
        
        for file_path in self.valid_data_after2[idx]:
            # 1. Load the data.
            data = torch.load(file_path, weights_only=False)
            
            # 2. Convert to a tensor and handle invalid values.
            # Convert to float32 (or int16) first to avoid uint8 overflow.
            data = torch.as_tensor(data, dtype=torch.float32) 
            
            # Map all invalid values greater than 3 (e.g., values around 120) to -1.
            data[data > 3] = -1.0 
            
            data_list.append(data)
            
            # Preserve the original timestamp extraction for name_list.
            match = re.search(r'NOM_(\d+)_', file_path)
            if match:
                extracted_number = int(match.group(1)[:-2])
                name_list.append(extracted_number)
            else:
                name_list.append('None')
                
        input_data  = torch.stack(data_list[:self.num_input]) 
        output_data = torch.stack(data_list[self.num_input:])  
        
        return input_data, output_data, name_list
#cities = ['Chengdu','Mohe','Huhehaote','Lanzhou','Kunming','Nanjing','Shanghai','Beijing','Changchun','Kuerle','Shenzhen', 'Xian']

