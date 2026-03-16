import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 本地自定义模块导入
from return_models import return_models
from time_series_pt_dataset_v2 import CloudMaskSequenceDataset_Fixed_Month
# 导入新的通用工具模块
from utils_unified import MaskedMSELoss, train, save_to_csv, visualize_predictions_day_night

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Eff_Unet")
    
    # 必填：城市参数
    parser.add_argument('--city', type=str, required=True, choices=['Nanjing', 'Changchun', 'Zhongxin'],
                        help="指定城市，将自动决定数据路径和参数")
    parser.add_argument('--seq_len', type=int, default=72)
    parser.add_argument('--pred_len', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--output_folder', type=str, default='./predictions')
    parser.add_argument('--device', type=str, default='cuda:1', choices=['cuda:0', 'cuda:1', 'cpu'])
    parser.add_argument('--resolution', type=int, default=1024)
    
    parser.add_argument('--train_months', type=str, nargs='+', default=['202509', '202510'])
    parser.add_argument('--test_months', type=str, nargs='+', default=['202511'])
    parser.add_argument('--test_only', action='store_true', help="如果设置，则跳过训练，直接加载目标权重进行测试")
    
    return parser.parse_args()

args = parse_args()

if __name__ == '__main__':
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    train_months = args.train_months
    test_months  = args.test_months
    
    # --- 自动化路径配置逻辑 ---
    if args.city == 'Nanjing':
        url = "/data4/mxx_new_code/fengyun_mxx_code/cropped_images_fixed/Nanjing_Center_1024/"
        # 基础权重路径
        base_url = './pth/Nanjing/Input_32_Output_2_202403_202502.pth'
    elif args.city == 'Changchun' or args.city == 'Zhongxin':
        url = "/data4/mxx_new_code/fengyun_mxx_code/cropped_images_fixed/ChangChun_Fixed_1024/"
        # 基础权重路径
        base_url = './pth/Changchun/Input_32_Output_2_202403_202502.pth'
    else:
        raise ValueError(f"未定义的城市配置: {args.city}")

    print(f"当前城市: {args.city}")
    print(f"数据路径: {url}")
    
    # 定义数据集
    train_dataset = CloudMaskSequenceDataset_Fixed_Month(base_directory=url, months= train_months,\
                num_input=args.seq_len, num_output=args.pred_len,\
                train_ratio=1, valid_ratio=0.0, test_ratio=0.0,\
                dataset_type='train', dataset_total=1)
    val_dataset     = CloudMaskSequenceDataset_Fixed_Month(base_directory=url, months= test_months,\
                num_input=args.seq_len, num_output=args.pred_len,\
                train_ratio=0.0, valid_ratio=1, test_ratio=0.0,\
                dataset_type='val', dataset_total=1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    
    # 初始化模型
    model = return_models(args.model_name, args.seq_len, args.pred_len, args.resolution, device)
    
    # 路径定义
    test_months_str = "_".join(args.test_months)
    pth_save_path = './models_pth/{}/{}/{}/{}/Input_{}_Output_{}.pth'.format(
        args.model_name, 
        args.resolution, 
        args.city, 
        test_months_str,
        args.seq_len, 
        args.pred_len
    )
    
    # --- 逻辑分支 ---
    if args.test_only:
        print(f"--- 模式: 仅测试 (Test Only) ---")
        print(f"尝试加载目标权重: {pth_save_path}")
        
        if os.path.exists(pth_save_path):
            state_dict = torch.load(pth_save_path, map_location=device)
            model.load_state_dict(state_dict)
            print("权重加载成功。")
        else:
            raise FileNotFoundError(f"错误: 找不到权重文件 {pth_save_path}")
            
    else:
        print(f"--- 模式: 训练 + 测试 (Train & Test) ---")
        
        # 加载基础预训练权重 (从自动配置的 base_url 加载)
        if os.path.exists(base_url):
            state_dict = torch.load(base_url, map_location=device)
            model_state_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(filtered_dict)
            model.load_state_dict(model_state_dict)
            print(f"基础预训练权重装载完成: {base_url}")
        else:
            print(f"警告: 未找到基础预训练权重 {base_url}，将从头开始训练")
        
        criterion = MaskedMSELoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        os.makedirs(os.path.dirname(pth_save_path), exist_ok=True)
        print(f"开始训练，权重将保存至: {pth_save_path}")
        train(model, train_loader, val_loader, criterion, optimizer, num_epochs=args.num_epochs, save_path=pth_save_path)


    # --- 预测和可视化 ---
    args.output_folder = './output_predictions/{}/{}/{}/Input_{}_Output_{}/'.format(
        args.model_name, 
        args.city, 
        test_months_str, 
        args.seq_len, 
        args.pred_len
    )

    os.makedirs(args.output_folder, exist_ok=True)
    
    print(f"开始预测，结果输出至: {args.output_folder}")
    
    # 这里的 city 参数会传给 utils_unified，用于自动决定画图坐标
    visualize_predictions_day_night(model, val_loader, output_folder=args.output_folder, \
                                    model_name= args.model_name, seq_len = args.seq_len, pred_len = args.pred_len, \
                                    city=args.city, do_vis=1, device=device,test_months_str = test_months_str)