import os
import time
import csv
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Loss function
# ============================================================

class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error computed only over valid pixels.

    Parameters
    ----------
    ignore_index : int or float
        Target value used to indicate invalid pixels.
    """

    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target):
        mask = (target != self.ignore_index).float()

        loss = self.mse(pred, target)

        masked_loss = (
            (loss * mask).sum()
            / (mask.sum() + 1e-8)
        )

        return masked_loss


# ============================================================
# Training
# ============================================================

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    save_path
):
    """
    Train the model and retain the checkpoint with the lowest
    validation loss.

    The best validation checkpoint is reloaded before returning,
    so subsequent evaluation uses the selected model rather than
    the model from the final training epoch.
    """

    device = next(model.parameters()).device

    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        start_time = time.time()

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        model.train()

        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as progress:

            progress.set_description(
                f"Epoch {epoch + 1}/{num_epochs}"
            )

            for inputs, labels, _ in progress:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                progress.set_postfix(
                    loss=loss.item()
                )

        epoch_train_loss = (
            running_loss / max(len(train_loader), 1)
        )

        train_losses.append(epoch_train_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Training Loss: {epoch_train_loss:.6f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for inputs, labels, _ in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

        epoch_val_loss = (
            val_loss / max(len(val_loader), 1)
        )

        val_losses.append(epoch_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Validation Loss: {epoch_val_loss:.6f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------
        if epoch_val_loss < best_val_loss:

            best_val_loss = epoch_val_loss

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                "Improved validation checkpoint detected "
                f"and saved to: {save_path}"
            )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"took {epoch_time:.2f} seconds."
        )

        # ----------------------------------------------------
        # Optional loss-curve visualization
        # ----------------------------------------------------
        if (epoch + 1) % 2 == 0:

            plt.figure()

            plt.plot(
                range(1, len(train_losses) + 1),
                train_losses,
                label="Train Loss"
            )

            plt.plot(
                range(1, len(val_losses) + 1),
                val_losses,
                label="Validation Loss"
            )

            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                "loss_curve_epoch.png"
            )

            plt.close()

    # --------------------------------------------------------
    # Reload best validation checkpoint
    # --------------------------------------------------------
    if os.path.exists(save_path):

        best_state_dict = torch.load(
            save_path,
            map_location=device
        )

        model.load_state_dict(
            best_state_dict
        )

        print(
            "Best validation checkpoint reloaded "
            "for subsequent evaluation."
        )

    return best_val_loss


# ============================================================
# Metric utilities
# ============================================================

def binary_metrics_masked(
    outputs,
    labels,
    mask,
    positive_label
):
    """
    Compute binary classification metrics over valid pixels.

    Parameters
    ----------
    outputs : torch.Tensor
        Discrete prediction labels.

    labels : torch.Tensor
        Discrete ground-truth labels.

    mask : torch.Tensor
        Boolean validity mask.

    positive_label : int
        Label treated as the positive class after binary merging.
    """

    outputs_pos = (
        (outputs == positive_label) & mask
    )

    labels_pos = (
        (labels == positive_label) & mask
    )

    outputs_neg = (
        (outputs != positive_label) & mask
    )

    labels_neg = (
        (labels != positive_label) & mask
    )

    tp = (
        outputs_pos & labels_pos
    ).sum().float()

    tn = (
        outputs_neg & labels_neg
    ).sum().float()

    fp = (
        outputs_pos & labels_neg
    ).sum().float()

    fn = (
        outputs_neg & labels_pos
    ).sum().float()

    epsilon = 1e-8

    accuracy = (
        (tp + tn)
        / (tp + tn + fp + fn + epsilon)
    )

    precision = (
        tp / (tp + fp + epsilon)
    )

    recall = (
        tp / (tp + fn + epsilon)
    )

    f1 = (
        2 * precision * recall
        / (precision + recall + epsilon)
    )

    iou = (
        tp / (tp + fp + fn + epsilon)
    )

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "iou": iou.item()
    }


# ============================================================
# Class conversion
# ============================================================

def discretize_prediction(
    outputs,
    min_class=0,
    max_class=3
):
    """
    Convert continuous model outputs into discrete cloud-mask
    categories by rounding to the nearest integer and clipping
    to the valid category range.
    """

    outputs = torch.round(outputs)

    outputs = torch.clamp(
        outputs,
        min=min_class,
        max=max_class
    )

    return outputs.to(torch.int64)


def merge_classes(
    tensor,
    class_mapping
):
    """
    Merge original categories according to a user-provided map.

    Example
    -------
    class_mapping = {
        0: 0,
        1: 0,
        2: 3,
        3: 3
    }
    """

    result = tensor.clone()

    for source_class, target_class in class_mapping.items():

        result = torch.where(
            tensor == source_class,
            torch.as_tensor(
                target_class,
                device=tensor.device,
                dtype=result.dtype
            ),
            result
        )

    return result


# ============================================================
# Evaluation
# ============================================================

def calculate_accu(
    label,
    outputs,
    roi_center_row=None,
    roi_center_col=None,
    roi_radius=None,
    ignore_index=-1,
    positive_label=3,
    class_mapping=None,
    min_class=0,
    max_class=3
):
    """
    Compute evaluation metrics for one sample.

    By default, evaluation is performed over the complete valid
    image. An optional ROI can be supplied explicitly.

    Parameters
    ----------
    roi_center_row : int or None
    roi_center_col : int or None
    roi_radius : int or None

        If all three values are specified, metrics are restricted
        to the corresponding ROI. Otherwise, the complete image is
        evaluated.

    class_mapping : dict or None

        Mapping used to merge original categories into binary
        categories.

        If None, the mapping used by the original implementation
        is retained:

            0 -> 0
            1 -> 0
            2 -> 3
            3 -> 3
    """

    if class_mapping is None:

        class_mapping = {
            0: 0,
            1: 0,
            2: 3,
            3: 3
        }

    # --------------------------------------------------------
    # Convert model outputs to discrete classes
    # --------------------------------------------------------
    outputs = discretize_prediction(
        outputs,
        min_class=min_class,
        max_class=max_class
    )

    label = label.to(torch.int64)

    H = label.shape[-2]
    W = label.shape[-1]

    # --------------------------------------------------------
    # Valid-pixel mask
    # --------------------------------------------------------
    valid_mask = (
        label != ignore_index
    )

    # --------------------------------------------------------
    # Optional ROI
    # --------------------------------------------------------
    if (
        roi_center_row is not None
        and roi_center_col is not None
        and roi_radius is not None
    ):

        roi_mask = torch.zeros_like(
            valid_mask,
            dtype=torch.bool
        )

        r_start = max(
            0,
            roi_center_row - roi_radius
        )

        r_end = min(
            H,
            roi_center_row + roi_radius
        )

        c_start = max(
            0,
            roi_center_col - roi_radius
        )

        c_end = min(
            W,
            roi_center_col + roi_radius
        )

        roi_mask[
            ...,
            r_start:r_end,
            c_start:c_end
        ] = True

        valid_mask = (
            valid_mask & roi_mask
        )

    # --------------------------------------------------------
    # Handle empty valid mask
    # --------------------------------------------------------
    if valid_mask.sum() == 0:

        empty_result = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0
        }

        return (
            0.0,
            0.0,
            empty_result
        )

    # --------------------------------------------------------
    # Original multi-class accuracy
    # --------------------------------------------------------
    correct_original = (
        (outputs == label)
        & valid_mask
    )

    original_accuracy = (
        correct_original.float().sum()
        / valid_mask.float().sum()
    )

    # --------------------------------------------------------
    # Binary class merging
    # --------------------------------------------------------
    outputs_binary = merge_classes(
        outputs,
        class_mapping
    )

    labels_binary = merge_classes(
        label,
        class_mapping
    )

    # --------------------------------------------------------
    # Binary metrics
    # --------------------------------------------------------
    result = binary_metrics_masked(
        outputs_binary,
        labels_binary,
        valid_mask,
        positive_label=positive_label
    )

    binary_correct = (
        (outputs_binary == labels_binary)
        & valid_mask
    )

    binary_accuracy = (
        binary_correct.float().sum()
        / valid_mask.float().sum()
    )

    return (
        binary_accuracy.item(),
        original_accuracy.item(),
        result
    )


# ============================================================
# Filename/time helpers
# ============================================================

def extract_sample_path(
    files_name,
    batch_index
):
    """
    Extract one sample identifier from common DataLoader
    collation formats.
    """

    try:

        if isinstance(
            files_name,
            (list, tuple)
        ):

            # Nested sequence format
            if (
                len(files_name) > 0
                and isinstance(
                    files_name[0],
                    (list, tuple)
                )
            ):
                return str(
                    files_name[0][batch_index]
                )

            return str(
                files_name[batch_index]
            )

        if torch.is_tensor(files_name):

            value = files_name[batch_index]

            if value.numel() == 1:
                return str(value.item())

            return str(
                value.detach().cpu().tolist()
            )

        return str(files_name)

    except Exception:

        return str(batch_index)


def parse_hour_from_filename(
    sample_path
):
    """
    Attempt to extract the observation hour from a filename.

    Returns None if no valid timestamp can be identified.
    """

    file_name = os.path.basename(
        str(sample_path)
    )

    file_name_pure = os.path.splitext(
        file_name
    )[0]

    try:

        if (
            file_name_pure.isdigit()
            and len(file_name_pure) >= 10
        ):

            return int(
                file_name_pure[8:10]
            )

        if "_" in file_name:

            parts = file_name.split("_")

            for part in parts:

                if (
                    part.isdigit()
                    and len(part) >= 10
                ):

                    return int(
                        part[8:10]
                    )

    except (IndexError, ValueError):

        pass

    return None


# ============================================================
# Visualization helpers
# ============================================================

def create_binary_visualization(
    class_map,
    valid_mask,
    foreground_classes=(0, 1)
):
    """
    Create a simple binary visualization image.
    """

    h, w = class_map.shape

    color_img = np.zeros(
        (h, w, 3),
        dtype=np.uint8
    )

    foreground_condition = (
        np.isin(
            class_map,
            foreground_classes
        )
        & valid_mask
    )

    color_img[
        foreground_condition
    ] = [255, 255, 255]

    return color_img


def draw_roi_box(
    image,
    center_row,
    center_col,
    radius,
    thickness=3
):
    """
    Draw an optional ROI box for visualization.
    """

    if (
        center_row is None
        or center_col is None
        or radius is None
    ):

        return image

    H, W, _ = image.shape

    r_start = max(
        0,
        center_row - radius
    )

    r_end = min(
        H,
        center_row + radius
    )

    c_start = max(
        0,
        center_col - radius
    )

    c_end = min(
        W,
        center_col + radius
    )

    image = image.copy()

    image[
        r_start:r_start + thickness,
        c_start:c_end
    ] = [255, 0, 0]

    image[
        r_end - thickness:r_end,
        c_start:c_end
    ] = [255, 0, 0]

    image[
        r_start:r_end,
        c_start:c_start + thickness
    ] = [255, 0, 0]

    image[
        r_start:r_end,
        c_end - thickness:c_end
    ] = [255, 0, 0]

    return image


# ============================================================
# Prediction / evaluation / visualization
# ============================================================

def visualize_predictions_day_night(
    model,
    test_loader,
    output_folder,
    model_name,
    seq_len,
    pred_len,
    city=None,
    do_vis=0,
    device=None,
    test_months_str="",
    roi_center_row=None,
    roi_center_col=None,
    roi_radius=None,
    day_start_hour=8,
    day_end_hour=20,
    eval_channel=0,
    ignore_index=-1,
    positive_label=3,
    class_mapping=None,
    min_class=0,
    max_class=3,
    csv_file=None
):
    """
    Evaluate model predictions and optionally separate samples
    into daytime and nighttime groups.

    All location-specific ROI parameters are supplied explicitly.
    No city-specific coordinates are hard-coded in this module.

    Parameters
    ----------
    eval_channel : int or None

        Output channel used for evaluation.

        If an integer is provided, that output channel is
        evaluated.

        If None, all output channels are evaluated jointly.

    day_start_hour / day_end_hour :

        Explicitly configurable day/night boundaries.
    """

    if device is None:
        device = next(
            model.parameters()
        ).device

    if class_mapping is None:

        class_mapping = {
            0: 0,
            1: 0,
            2: 3,
            3: 3
        }

    stats = {

        "Daytime": {
            "acc1": [],
            "acc2": [],
            "accuracy": [],
            "prec": [],
            "recall": [],
            "f1": [],
            "iou": []
        },

        "Nighttime": {
            "acc1": [],
            "acc2": [],
            "accuracy": [],
            "prec": [],
            "recall": [],
            "f1": [],
            "iou": []
        },

        "Total": {
            "acc1": [],
            "acc2": [],
            "accuracy": [],
            "prec": [],
            "recall": [],
            "f1": [],
            "iou": []
        }
    }

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    model.eval()

    with torch.no_grad():

        for batch_idx, batch_data in enumerate(
            tqdm(test_loader)
        ):

            inputs, label, files_name = batch_data

            inputs = inputs.to(device)
            label = label.to(device)

            outputs = model(inputs)

            batch_size = inputs.size(0)

            # ------------------------------------------------
            # Sample-wise metric computation
            # ------------------------------------------------
            for b in range(batch_size):

                sample_path = extract_sample_path(
                    files_name,
                    b
                )

                hour = parse_hour_from_filename(
                    sample_path
                )

                if hour is None:

                    period = None

                elif (
                    day_start_hour
                    <= hour
                    < day_end_hour
                ):

                    period = "Daytime"

                else:

                    period = "Nighttime"

                # --------------------------------------------
                # Select output channel
                # --------------------------------------------
                if eval_channel is None:

                    sample_label = (
                        label[b:b + 1]
                    )

                    sample_output = (
                        outputs[b:b + 1]
                    )

                else:

                    sample_label = label[
                        b:b + 1,
                        eval_channel:eval_channel + 1
                    ]

                    sample_output = outputs[
                        b:b + 1,
                        eval_channel:eval_channel + 1
                    ]

                acc_t, acc_t2, result = calculate_accu(
                    sample_label,
                    sample_output,
                    roi_center_row=roi_center_row,
                    roi_center_col=roi_center_col,
                    roi_radius=roi_radius,
                    ignore_index=ignore_index,
                    positive_label=positive_label,
                    class_mapping=class_mapping,
                    min_class=min_class,
                    max_class=max_class
                )

                targets = ["Total"]

                if period is not None:
                    targets.append(period)

                for target in targets:

                    stats[target][
                        "acc1"
                    ].append(acc_t)

                    stats[target][
                        "acc2"
                    ].append(acc_t2)

                    stats[target][
                        "accuracy"
                    ].append(
                        result["accuracy"]
                    )

                    stats[target][
                        "prec"
                    ].append(
                        result["precision"]
                    )

                    stats[target][
                        "recall"
                    ].append(
                        result["recall"]
                    )

                    stats[target][
                        "f1"
                    ].append(
                        result["f1"]
                    )

                    stats[target][
                        "iou"
                    ].append(
                        result["iou"]
                    )

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------
            if do_vis:

                pred_folder = os.path.join(
                    output_folder,
                    "pred"
                )

                gt_folder = os.path.join(
                    output_folder,
                    "gt"
                )

                os.makedirs(
                    pred_folder,
                    exist_ok=True
                )

                os.makedirs(
                    gt_folder,
                    exist_ok=True
                )

                for b in range(
                    outputs.size(0)
                ):

                    sample_path = extract_sample_path(
                        files_name,
                        b
                    )

                    fname = os.path.basename(
                        str(sample_path)
                    )

                    fname = os.path.splitext(
                        fname
                    )[0]

                    for c in range(
                        outputs.size(1)
                    ):

                        raw_pred = (
                            outputs[b, c]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        pred_idx = np.clip(
                            np.round(raw_pred),
                            min_class,
                            max_class
                        ).astype(int)

                        label_idx = (
                            label[b, c]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        valid_mask = (
                            label_idx
                            != ignore_index
                        )

                        pred_rgb = (
                            create_binary_visualization(
                                pred_idx,
                                valid_mask
                            )
                        )

                        label_rgb = (
                            create_binary_visualization(
                                label_idx,
                                valid_mask
                            )
                        )

                        pred_rgb = draw_roi_box(
                            pred_rgb,
                            roi_center_row,
                            roi_center_col,
                            roi_radius
                        )

                        label_rgb = draw_roi_box(
                            label_rgb,
                            roi_center_row,
                            roi_center_col,
                            roi_radius
                        )

                        pred_path = os.path.join(
                            pred_folder,
                            f"{fname}_frame_{c}.png"
                        )

                        gt_path = os.path.join(
                            gt_folder,
                            f"{fname}_frame_{c}.png"
                        )

                        if not os.path.exists(
                            pred_path
                        ):

                            plt.imsave(
                                pred_path,
                                pred_rgb
                            )

                        if not os.path.exists(
                            gt_path
                        ):

                            plt.imsave(
                                gt_path,
                                label_rgb
                            )

                        concat_path = os.path.join(
                            output_folder,
                            f"{fname}_frame_{c}_concat.png"
                        )

                        if not os.path.exists(
                            concat_path
                        ):

                            fig, axs = plt.subplots(
                                1,
                                2,
                                figsize=(8, 4)
                            )

                            axs[0].imshow(
                                pred_rgb
                            )

                            axs[0].set_title(
                                "Prediction"
                            )

                            axs[0].axis(
                                "off"
                            )

                            axs[1].imshow(
                                label_rgb
                            )

                            axs[1].set_title(
                                "Ground Truth"
                            )

                            axs[1].axis(
                                "off"
                            )

                            fig.savefig(
                                concat_path,
                                bbox_inches="tight",
                                pad_inches=0
                            )

                            plt.close(fig)

    # ========================================================
    # Macro-average metrics over individual samples
    # ========================================================

    final_return = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    )

    for period_name in [
        "Daytime",
        "Nighttime",
        "Total"
    ]:

        data = stats[period_name]

        if len(data["acc1"]) == 0:
            continue

        mean_acc1 = float(
            np.mean(data["acc1"])
        )

        mean_acc2 = float(
            np.mean(data["acc2"])
        )

        mean_accuracy = float(
            np.mean(data["accuracy"])
        )

        mean_precision = float(
            np.mean(data["prec"])
        )

        mean_recall = float(
            np.mean(data["recall"])
        )

        mean_f1 = float(
            np.mean(data["f1"])
        )

        mean_iou = float(
            np.mean(data["iou"])
        )

        save_name = (
            f"{model_name}_"
            f"{period_name}_"
            f"{test_months_str}"
        )

        save_to_csv(
            save_name,
            mean_acc1,
            mean_acc2,
            mean_accuracy,
            mean_precision,
            mean_recall,
            mean_f1,
            mean_iou,
            seq_len,
            pred_len,
            city=city,
            csv_file=csv_file
        )

        if period_name == "Total":

            final_return = (
                mean_acc1,
                mean_acc2,
                mean_accuracy,
                mean_precision,
                mean_recall,
                mean_f1,
                mean_iou
            )

    return final_return


# ============================================================
# CSV output
# ============================================================

def save_to_csv(
    model_name,
    mean_acc,
    mean_acc2,
    accuracy_avg,
    prec_avg,
    recall_avg,
    f1_avg,
    iou_avg,
    seq_len,
    pred_len,
    city=None,
    csv_file=None
):
    """
    Save evaluation results to a generic CSV file.

    No city name, image resolution, server path, or experiment
    date is hard-coded.
    """

    if csv_file is None:

        csv_file = os.path.join(
            "csv_results",
            "evaluation_results.csv"
        )

    csv_dir = os.path.dirname(
        csv_file
    )

    if csv_dir:

        os.makedirs(
            csv_dir,
            exist_ok=True
        )

    file_exists = os.path.exists(
        csv_file
    )

    with open(
        csv_file,
        mode="a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:

            writer.writerow([
                "model_name",
                "seq_len",
                "pred_len",
                "region",
                "mean_acc",
                "mean_acc_original",
                "accuracy_avg",
                "precision_avg",
                "recall_avg",
                "f1_avg",
                "iou_avg"
            ])

        writer.writerow([
            model_name,
            seq_len,
            pred_len,
            city if city is not None else "",
            mean_acc,
            mean_acc2,
            accuracy_avg,
            prec_avg,
            recall_avg,
            f1_avg,
            iou_avg
        ])

    print(
        f"Results saved to: {csv_file}"
    )
