# 1. 标准库导入
import os
import re
from datetime import datetime, timedelta

# 2. 第三方库导入
import torch
from torch.utils.data import Dataset

def get_first_timestamp(filepath):
    # 1. 只取文件名，彻底忽略路径中的 /202403/ 文件夹
    filename = os.path.basename(filepath)
    
    # 2. 查找第一组连续的14位数字
    # re.search 的特性是从左往右找，找到第一个满足条件的就立刻停止
    match = re.search(r'(\d{14})', filename)
    
    if match:
        # group(1) 就是它找到的第一个串
        return datetime.strptime(match.group(1), '%Y%m%d%H%M%S')
    else:
        return datetime.min
    
def write_to_txt(data, output_file):
    # 打开文件以写入
    with open(output_file, 'w') as f:
        count = 0  # 用于计数每写入的元素个数
        
        # 遍历二维列表中的每一行
        for row in data:
            for value in row:
                # 写入当前值
                f.write(f"{value}\n")
                count += 1
                
                # 每写入 10 个数据后换行
                if count % 10 == 0:
                    f.write("\n")

# 将时间戳转换为 datetime 对象
def convert_to_datetime(timestamp):
    timestamp_str = str(timestamp)
    return datetime.strptime(timestamp_str, "%Y%m%d%H%M")

# 判断时间间隔是否是15分钟
def check_time_interval(time_list):
    for i in range(1, len(time_list)):
        # 获取相邻的两个时间戳
        prev_time = convert_to_datetime(time_list[i - 1])
        curr_time = convert_to_datetime(time_list[i])

        # 计算时间差
        time_diff = curr_time - prev_time

        # 如果差值不是15分钟，返回False
        if time_diff != timedelta(minutes=15):
            return False
    return True  # 所有时间间隔都是15分钟

class CloudMaskSequenceDataset(Dataset):
    def __init__(self, directory, num_input=5, num_output=5, time_difference=15, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15,\
        dataset_type='train',dataset_total=1):
        """
        :param directory: 包含 .pt 文件的文件夹路径
        :param num_input: 输入的时间戳数量（例如：5）
        :param num_output: 预测的时间戳数量（例如：5）
        :param time_difference: 时间间隔的要求，默认为15分钟
        """
        self.num_input = num_input
        self.num_output = num_output
        self.time_difference = time_difference  # 设定时间差为15分钟
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.dataset_type = dataset_type
        self.dataset_total_size = dataset_total
        #self.valid_data_name = []
        
        self.valid_data = []
        self.file_paths = self.get_pt_files(directory)
        self.get_filtered_pt_files(directory)
        # 划分训练集、验证集和测试集
        self.valid_data_after1 = self.valid_data[:self.dataset_total(self.valid_data)]
        self.train_size, self.valid_size = self.split_data(self.valid_data_after1)
        
        
        # 划分数据
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
        按照给定的比例划分数据集
        """
        total_size = len(data)
        train_size = int(self.train_ratio * total_size)
        valid_size = int(self.valid_ratio * total_size)



        return train_size, valid_size
    def get_pt_files(self, directory):
        """
        获取所有 .pt 文件的路径，并按时间戳排序
        """
        pt_files = []

        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.pt'):
                    file_path = os.path.join(root, filename)
                    pt_files.append(file_path)

        # 根据文件名中的时间戳排序，假设时间戳在文件名的特定位置
        pt_files.sort(key=get_first_timestamp)

        #"/root/autodl-tmp/cropped_images_128_zip/Dongjing/Dongjing/202206/cropped_FY4B-_AGRI--_N_DISK_1330E_L2-_CLM-_MULT_NOM_20220601000000_20220601001459_4000M_V0001.pt"
        return pt_files
    def get_filtered_pt_files(self, directory):
        """
        筛选掉那些时间间隔不符合条件的数据文件路径
        """
        pt_files = self.get_pt_files(directory)

        # 只保留符合时间间隔要求的数据
        for idx in range(len(pt_files) - self.num_input - self.num_output + 1):
            input_name = []
            output_name = []

            # 获取连续的 num_input 张图像作为输入
            for i in range(idx, idx + self.num_input):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    extracted_number = match.group(1)[:-2]  # 去掉最后两位
                    extracted_number = int(extracted_number)
                    #print(extracted_number)
                    input_name.append(extracted_number)

            # 获取后面的 num_output 张图像作为目标
            for i in range(idx + self.num_input, idx + self.num_input + self.num_output):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    extracted_number = match.group(1)[:-2]
                    extracted_number = int(extracted_number)
                    output_name.append(extracted_number)

            # 将时间戳转换为 datetime 对象
            time_objects_input = [datetime.strptime(str(time), "%Y%m%d%H%M") for time in input_name]
            time_differences_input = [time_objects_input[i + 1] - time_objects_input[i] for i in range(len(time_objects_input) - 1)]

            time_objects_output = [datetime.strptime(str(time), "%Y%m%d%H%M") for time in output_name]
            #print(time_objects_output)
            list_all = time_objects_input + time_objects_output
            list_difference_all = [list_all[i + 1] - list_all[i] for i in range(len(list_all) - 1)]
            
            
            time_differences_output = [time_objects_output[i + 1] - time_objects_output[i] for i in range(len(time_objects_output) - 1)]
            #print(time_differences_output)
            # 判断时间间隔是否为设定的15分钟
            
            if all(diff.total_seconds() == 900 for diff in list_difference_all):
                
                
                self.valid_data.append(pt_files[idx:idx+self.num_input + self.num_output])
                #print(len(pt_files[idx:idx+self.num_input + self.num_output]))
                #self.valid_data_name.append(pt_files[idx:idx+self.num_input + self.num_output])
            # self.no_valid_data.append(pt_files[idx:idx+self.num_input + self.num_output])
                # 将排序后的文件路径写入txt文件
    def __len__(self):
        # 返回数据集的长度（能够构建多少组时序数据）
        #return len(self.file_paths) - self.num_input - self.num_output + 1
        return len(self.valid_data_after2)
    # 在 time_series_pt_dataset_v2.py 中修改 __getitem__
    def __getitem__(self, idx):
        data_list = []
        name_list = []
        
        for file_path in self.valid_data_after2[idx]:
            # 1. 加载数据
            data = torch.load(file_path, weights_only=False)
            
            # 2. 转换为 Tensor 并处理异常值
            # 注意：这里建议先转为 float32 或 int16，避免 uint8 溢出
            data = torch.as_tensor(data, dtype=torch.float32) 
            
            # 将所有大于 3 的异常值（如 120 多）统一设为 -1
            data[data > 3] = -1.0 
            
            data_list.append(data)
            
            # ... 原有的 name_list 处理代码 ...
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
        :param base_directory: 基础路径，例如 '/.../ChangChun_Fixed_1024/'
        :param months: 需要加载的月份列表，例如 ['202403', '202404']
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
        # --- 修改点 1: 传入 base_directory 和 months ---
        self.file_paths = self.get_pt_files_by_months(base_directory, months)
        
        # 筛选符合时间间隔的序列
        self.get_filtered_pt_files_from_list(self.file_paths)
        
        # 划分数据集逻辑保持不变
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
        --- 修改点 2: 仅加载指定月份子目录下的文件 ---
        """
        pt_files = []
        for month in months:
            # 拼接月份路径: /.../ChangChun_Fixed_1024/202403/
            month_path = os.path.join(base_dir, month)
            
            if not os.path.exists(month_path):
                print(f"警告: 路径不存在 {month_path}")
                continue
                
            for root, _, files in os.walk(month_path):
                for filename in files:
                    if filename.endswith('.pt'):
                        pt_files.append(os.path.join(root, filename))

        # 排序确保时间连续性检查有效
        # 假设您的 get_first_timestamp 函数已在外部定义
        pt_files.sort(key=get_first_timestamp) 
        return pt_files

    def get_filtered_pt_files_from_list(self, pt_files):
        """
        --- 修改点 3: 直接处理传入的列表，逻辑与原 get_filtered_pt_files 一致 ---
        """
        for idx in range(len(pt_files) - self.num_input - self.num_output + 1):
            input_name = []
            output_name = []

            # 提取输入和输出的时间戳
            for i in range(idx, idx + self.num_input + self.num_output):
                match = re.search(r'NOM_(\d+)_', pt_files[i])
                if match:
                    # 提取到分钟级别
                    ts = int(match.group(1)[:-2])
                    if i < idx + self.num_input:
                        input_name.append(ts)
                    else:
                        output_name.append(ts)

            # 时间差验证
            all_timestamps = input_name + output_name
            time_objects = [datetime.strptime(str(ts), "%Y%m%d%H%M") for ts in all_timestamps]
            
            # 检查是否所有相邻时间差均为 15 分钟 (900秒)
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
            # 1. 加载数据
            data = torch.load(file_path, weights_only=False)
            
            # 2. 转换为 Tensor 并处理异常值
            # 注意：这里建议先转为 float32 或 int16，避免 uint8 溢出
            data = torch.as_tensor(data, dtype=torch.float32) 
            
            # 将所有大于 3 的异常值（如 120 多）统一设为 -1
            data[data > 3] = -1.0 
            
            data_list.append(data)
            
            # ... 原有的 name_list 处理代码 ...
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
# # cities = ['Chengdu']
# cities = ['Dongjing']
# url = '/root/autodl-tmp/cropped_images_128_zip/Dongjing/Dongjing/'
# # for name in cities: 
# #     # 使用示例def __init__(self, directory, num_input=5, num_output=5, time_difference=15, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15,dataset_type='train'):

# directory = "/root/autodl-tmp/cropped_images_128_zip/Dongjing/Dongjing/"
# dataset_train = CloudMaskSequenceDataset(directory, num_input=2, num_output=2,train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,\
#     dataset_type='train',dataset_total=0.1)
# dataset_val = CloudMaskSequenceDataset(directory, num_input=2, num_output=2,train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,\
#     dataset_type='val',dataset_total=0.1)
# dataset_test = CloudMaskSequenceDataset(directory, num_input=2, num_output=2,train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,\
#     dataset_type='test',dataset_total=0.1)

# # # 打印数据集长度

# print(f"数据集长度: {len(dataset_train)}")
# print(len(dataset_train.valid_data_after2),"len(dataset.valid_data)") 
# print(len(dataset_train),"len(dataset)")

#     print((dataset_train[0][0].shape),"(dataset)")
#     #print((dataset_train.valid_data_after2[-2:]),"len(dataset.valid_data)") 
    
#     print(f"数据集长度: {len(dataset_val)}")
#     print(len(dataset_val.valid_data_after2),"len(dataset.valid_data)") 
#     print(len(dataset_val),"len(dataset)")
#     #print((dataset_val.valid_data_after2[-2:]),"len(dataset.valid_data)") 
    
#     print(f"数据集长度: {len(dataset_test)}")
#     print(len(dataset_test.valid_data_after2),"len(dataset.valid_data)") 
#     print(len(dataset_test),"len(dataset)")
    #print((dataset_test.valid_data_after2[-2:]),"len(dataset.valid_data)") 
# 数据集长度: 81646 num_input = 2
# 数据集长度: 33638 num_input = 24 to 24  exited with code=0 in 30.365 seconds
# 数据集长度: 20395 num_input = 32 to 32  exited with code=0 in 39.675 seconds
# 数据集长度: 8981  num_input = 40 to 40  exited with code=0 in 48.618 seconds