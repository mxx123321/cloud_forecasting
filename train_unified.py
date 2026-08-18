import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import local custom modules
from return_models import return_models
from time_series_pt_dataset_v2 import CloudMaskSequenceDataset_Fixed_Month
from utils_unified import (
    MaskedMSELoss,
    train,
    save_to_csv,
    visualize_predictions_day_night
)


def parse_args():
    parser = argparse.ArgumentParser()

    # ---------------------------------------------------------
    # Model configuration
    # ---------------------------------------------------------
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--pred_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--resolution', type=int, default=None)

    # ---------------------------------------------------------
    # Dataset configuration
    # ---------------------------------------------------------
    parser.add_argument(
        '--city',
        type=str,
        default=None,
        help='Name of the study region.'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the dataset.'
    )

    parser.add_argument(
        '--train_months',
        type=str,
        nargs='+',
        default=None,
        help='Months used for training.'
    )

    parser.add_argument(
        '--val_months',
        type=str,
        nargs='+',
        default=None,
        help='Months used for validation/testing.'
    )

    # ---------------------------------------------------------
    # Checkpoint configuration
    # ---------------------------------------------------------
    parser.add_argument(
        '--pretrained_checkpoint',
        type=str,
        default=None,
        help='Optional pretrained checkpoint.'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path for saving/loading the target checkpoint.'
    )

    # ---------------------------------------------------------
    # Runtime configuration
    # ---------------------------------------------------------
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device used for training and inference, e.g., cuda:0 or cpu.'
    )

    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Directory used to save prediction results.'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Skip training and directly load the target checkpoint.'
    )

    return parser.parse_args()


args = parse_args()


def check_required_arguments(args):
    """Check whether the required experiment settings are provided."""

    required_args = {
        'model_name': args.model_name,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'batch_size': args.batch_size,
        'resolution': args.resolution,
        'data_path': args.data_path,
        'val_months': args.val_months,
        'device': args.device,
        'checkpoint_path': args.checkpoint_path,
        'output_folder': args.output_folder,
    }

    if not args.test_only:
        required_args.update({
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'train_months': args.train_months,
        })

    missing_args = [
        name for name, value in required_args.items()
        if value is None
    ]

    if missing_args:
        raise ValueError(
            'The following experiment settings must be specified: '
            + ', '.join(missing_args)
        )


if __name__ == '__main__':

    check_required_arguments(args)

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu'
    )

    print(f'Device: {device}')
    print(f'Data path: {args.data_path}')

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    train_dataset = None

    if not args.test_only:
        train_dataset = CloudMaskSequenceDataset_Fixed_Month(
            base_directory=args.data_path,
            months=args.train_months,
            num_input=args.seq_len,
            num_output=args.pred_len,
            train_ratio=1.0,
            valid_ratio=0.0,
            test_ratio=0.0,
            dataset_type='train',
            dataset_total=1
        )

    val_dataset = CloudMaskSequenceDataset_Fixed_Month(
        base_directory=args.data_path,
        months=args.val_months,
        num_input=args.seq_len,
        num_output=args.pred_len,
        train_ratio=0.0,
        valid_ratio=1.0,
        test_ratio=0.0,
        dataset_type='val',
        dataset_total=1
    )

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
    else:
        train_loader = None

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = return_models(
        args.model_name,
        args.seq_len,
        args.pred_len,
        args.resolution,
        device
    )

    # ---------------------------------------------------------
    # Training / testing
    # ---------------------------------------------------------
    if args.test_only:

        print('--- Mode: Test Only ---')

        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint not found: {args.checkpoint_path}'
            )

        state_dict = torch.load(
            args.checkpoint_path,
            map_location=device
        )

        model.load_state_dict(state_dict)

        print('Checkpoint loaded successfully.')

    else:

        print('--- Mode: Train & Test ---')

        # Optional pretrained initialization
        if args.pretrained_checkpoint:

            if os.path.exists(args.pretrained_checkpoint):

                state_dict = torch.load(
                    args.pretrained_checkpoint,
                    map_location=device
                )

                model_state_dict = model.state_dict()

                filtered_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if (
                        k in model_state_dict
                        and v.size() == model_state_dict[k].size()
                    )
                }

                model_state_dict.update(filtered_dict)
                model.load_state_dict(model_state_dict)

                print(
                    'Pretrained checkpoint loaded: '
                    f'{args.pretrained_checkpoint}'
                )

            else:
                print(
                    'Warning: pretrained checkpoint was not found. '
                    'Training will start from scratch.'
                )

        # Loss and optimizer
        criterion = MaskedMSELoss(ignore_index=-1)

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate
        )

        # Create checkpoint directory
        checkpoint_dir = os.path.dirname(args.checkpoint_path)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        print(
            'Starting training. '
            f'Checkpoint will be saved to: {args.checkpoint_path}'
        )

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=args.num_epochs,
            save_path=args.checkpoint_path
        )

    # ---------------------------------------------------------
    # Prediction and visualization
    # ---------------------------------------------------------
    os.makedirs(args.output_folder, exist_ok=True)

    print(
        'Starting prediction. '
        f'Results will be saved to: {args.output_folder}'
    )

    val_period_str = '_'.join(args.val_months)

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
        test_months_str=val_period_str
    )
