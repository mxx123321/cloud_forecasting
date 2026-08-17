import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import local custom modules
from return_models import return_models
from time_series_pt_dataset_v2 import CloudMaskSequenceDataset_Fixed_Month

# Import the unified utility module
from utils_unified import MaskedMSELoss, train, save_to_csv, visualize_predictions_day_night


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Eff_Unet")

    # Required argument: city
    parser.add_argument(
        '--city',
        type=str,
        required=True,
        choices=['Nanjing', 'Changchun', 'Zhongxin'],
        help="Specify the city; the data path and related settings will be configured automatically."
    )
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
    parser.add_argument(
        '--test_only',
        action='store_true',
        help="If enabled, skip training and directly load the target checkpoint for testing."
    )

    return parser.parse_args()


args = parse_args()


if __name__ == '__main__':
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    train_months = args.train_months
    test_months = args.test_months

    # --- Automatic path configuration ---
    if args.city == 'Nanjing':
        url = "/data4/mxx_new_code/fengyun_mxx_code/cropped_images_fixed/Nanjing_Center_1024/"

        # Base pretrained checkpoint
        base_url = './pth/Nanjing/Input_32_Output_2_202403_202502.pth'

    elif args.city == 'Changchun' or args.city == 'Zhongxin':
        url = "/data4/mxx_new_code/fengyun_mxx_code/cropped_images_fixed/ChangChun_Fixed_1024/"

        # Base pretrained checkpoint
        base_url = './pth/Changchun/Input_32_Output_2_202403_202502.pth'

    else:
        raise ValueError(f"Undefined city configuration: {args.city}")

    print(f"Current city: {args.city}")
    print(f"Data path: {url}")

    # Define training and validation datasets
    train_dataset = CloudMaskSequenceDataset_Fixed_Month(
        base_directory=url,
        months=train_months,
        num_input=args.seq_len,
        num_output=args.pred_len,
        train_ratio=1,
        valid_ratio=0.0,
        test_ratio=0.0,
        dataset_type='train',
        dataset_total=1
    )

    val_dataset = CloudMaskSequenceDataset_Fixed_Month(
        base_directory=url,
        months=test_months,
        num_input=args.seq_len,
        num_output=args.pred_len,
        train_ratio=0.0,
        valid_ratio=1,
        test_ratio=0.0,
        dataset_type='val',
        dataset_total=1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Initialize the model
    model = return_models(
        args.model_name,
        args.seq_len,
        args.pred_len,
        args.resolution,
        device
    )

    # Define the checkpoint path
    test_months_str = "_".join(args.test_months)
    pth_save_path = './models_pth/{}/{}/{}/{}/Input_{}_Output_{}.pth'.format(
        args.model_name,
        args.resolution,
        args.city,
        test_months_str,
        args.seq_len,
        args.pred_len
    )

    # --- Training/testing logic ---
    if args.test_only:
        print("--- Mode: Test Only ---")
        print(f"Attempting to load target checkpoint: {pth_save_path}")

        if os.path.exists(pth_save_path):
            state_dict = torch.load(pth_save_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully.")
        else:
            raise FileNotFoundError(f"Error: checkpoint file not found: {pth_save_path}")

    else:
        print("--- Mode: Train & Test ---")

        # Load the base pretrained checkpoint configured for the selected city
        if os.path.exists(base_url):
            state_dict = torch.load(base_url, map_location=device)
            model_state_dict = model.state_dict()

            # Load only parameters whose names and tensor shapes match the current model
            filtered_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state_dict and v.size() == model_state_dict[k].size()
            }

            model_state_dict.update(filtered_dict)
            model.load_state_dict(model_state_dict)
            print(f"Base pretrained checkpoint loaded: {base_url}")

        else:
            print(
                f"Warning: base pretrained checkpoint not found: {base_url}. "
                "Training will start from scratch."
            )

        criterion = MaskedMSELoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        os.makedirs(os.path.dirname(pth_save_path), exist_ok=True)
        print(f"Starting training. Checkpoint will be saved to: {pth_save_path}")

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=args.num_epochs,
            save_path=pth_save_path
        )

    # --- Prediction and visualization ---
    args.output_folder = './output_predictions/{}/{}/{}/Input_{}_Output_{}/'.format(
        args.model_name,
        args.city,
        test_months_str,
        args.seq_len,
        args.pred_len
    )

    os.makedirs(args.output_folder, exist_ok=True)

    print(f"Starting prediction. Results will be saved to: {args.output_folder}")

    # The city argument is passed to utils_unified so that the plotting coordinates
    # can be selected automatically for each study region.
    visualize_predictions_day_night(
        model,
        val_loader,
        output_folder=args.output_folder,
        model_name=args.model_name,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        city=args.city,
        do_vis=1,
        device=device,
        test_months_str=test_months_str
    )
