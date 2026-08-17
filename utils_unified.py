import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configuration dictionaries (unchanged) ---
expansion_ratios_L = {
    '0': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

expansion_ratios_L_reverse = {
    '3': [4, 4, 4, 4, 4],
    '2': [4, 4, 4, 4, 4],
    '1': [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '0': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
}

# --- Loss function ---
class MaskedMSELoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(MaskedMSELoss, self).__init__()
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        mask = (target != self.ignore_index).float()
        loss = self.mse(pred, target)
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        return masked_loss

# --- Helper function: get city-specific parameters ---
def get_city_params(city):
    """
    Return city-specific parameters, such as the ROI center coordinates.
    """
    params = {}
    if city == 'Nanjing':
        # ROI center for Nanjing
        params['center_row'] = 512
        params['center_col'] = 512
        params['day_night_center_row'] = 512 # Day/night evaluation uses the same center row for Nanjing.
    elif city == 'Changchun':
        # ROI center for Changchun
        params['center_row'] = 339
        params['center_col'] = 512
        # Note: the original code used a different row setting for Changchun in visualize_predictions_day_night.
        params['day_night_center_row'] = 339 
    else:
        # Default values
        params['center_row'] = 339
        params['center_col'] = 512
        params['day_night_center_row'] = 339
        
    return params

# --- Training function ---
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, save_path='./model.pth'): 
    device = next(model.parameters()).device
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_val_loss = float('inf')
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            for i, (inputs, labels, _) in enumerate(tepoch):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}')

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Improved validation checkpoint detected and saved to: {save_path}")

        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] took {epoch_time:.2f} seconds.')

        if (epoch + 1) % 2 == 0:
            plt.figure()
            plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
            plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'loss_curve_epoch.png')
            plt.close()

# --- Evaluation metric helper ---
def binary_metrics_masked(outputs, labels, mask, positive_label=0):
    outputs_pos = (outputs == positive_label) & mask
    labels_pos = (labels == positive_label) & mask
    outputs_neg = (outputs != positive_label) & mask
    labels_neg = (labels != positive_label) & mask

    tp = (outputs_pos & labels_pos).sum().float()
    tn = (outputs_neg & labels_neg).sum().float()
    fp = (outputs_pos & labels_neg).sum().float()
    fn = (outputs_neg & labels_pos).sum().float()

    epsilon = 1e-8
    accuracy = (tp + tn) / (mask.sum() + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }

def calculate_accu(label, outputs, city='Nanjing'):
    device = label.device
    
    # --- 1. Get city-specific parameters ---
    city_params = get_city_params(city)
    center_row = city_params['center_row']
    center_col = city_params['center_col']
    radius = 32       

    # --- 2. Preprocess predictions and labels ---
    outputs = torch.clamp(torch.round(outputs), 0, 3).to(torch.int)
    label = label.to(torch.int)
    
    H, W = label.shape[-2], label.shape[-1]

    # --- 3. Create the ROI mask ---
    roi_mask = torch.zeros_like(label, dtype=torch.bool)
    r_start = max(0, center_row - radius)
    r_end = min(H, center_row + radius)
    c_start = max(0, center_col - radius)
    c_end = min(W, center_col + radius)
    roi_mask[..., r_start:r_end, c_start:c_end] = True

    # --- 4. Combine with the valid-pixel mask ---
    valid_mask = (label != -1) & roi_mask
    
    if valid_mask.sum() == 0: 
        return 0, 0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'kappa': 0}

    # --- 5. Compute the original four-class accuracy ---
    correct_mask_org = (outputs == label) & valid_mask
    accuracy2 = correct_mask_org.float().sum() / valid_mask.float().sum()
    
    # --- 6. Convert the four classes into the binary evaluation setting ---
    outputs_process = torch.where(outputs == 1, torch.tensor(0, device=device),
                        torch.where(outputs == 2, torch.tensor(3, device=device), outputs))
    label_process = torch.where(label == 1, torch.tensor(0, device=device),
                        torch.where(label == 2, torch.tensor(3, device=device), label))
    
    # --- 7. Compute binary evaluation metrics ---
    result = binary_metrics_masked(outputs_process, label_process, valid_mask, positive_label=3)
    
    correct_mask = (outputs_process == label_process) & valid_mask
    accuracy = correct_mask.float().sum() / valid_mask.float().sum()
    
    return accuracy.item(), accuracy2.item(), result

# --- Prediction and visualization function (day/night evaluation) ---
def visualize_predictions_day_night(model, test_loader, output_folder, model_name, seq_len, pred_len, city, do_vis, device,test_months_str=""):
    
    # --- Get city-specific parameters ---
    city_params = get_city_params(city)
    # Use the day/night-specific row coordinate here.
    center_row = city_params['day_night_center_row'] 
    center_col = city_params['center_col']
    radius = 32

    # 1. Initialize metric containers.
    stats = {
        "Daytime": {"acc1": [], "acc2": [], "accuracy": [], "prec": [], "recall": [], "f1": [], "iou": []},
        "Nighttime": {"acc1": [], "acc2": [], "accuracy": [], "prec": [], "recall": [], "f1": [], "iou": []},
        "Total": {"acc1": [], "acc2": [], "accuracy": [], "prec": [], "recall": [], "f1": [], "iou": []}
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, label, files_name) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)
            
            batch_size = inputs.size(0)
            
            for b in range(batch_size):
                # A. Extract the sample filename/path.
                if isinstance(files_name, (list, tuple)) and len(files_name) > 0:
                    sample_path = files_name[0][b] if isinstance(files_name[0], (list, tuple)) else files_name[b]
                else:
                    sample_path = files_name[b]
                
                if torch.is_tensor(sample_path):
                    sample_path = str(sample_path[0].item())

                # B. Parse the observation time.
                file_name = os.path.basename(sample_path)
                file_name_pure = os.path.splitext(file_name)[0] 
                
                try:
                    if file_name_pure.isdigit() and len(file_name_pure) >= 10:
                        hour = int(file_name_pure[8:10])
                    elif '_' in file_name:
                        time_str = file_name.split('_')[-4] 
                        hour = int(time_str[8:10])
                    else:
                        hour = 0 
                except (IndexError, ValueError) as e:
                    hour = 0
                
                period = "Daytime" if 8 <= hour < 20 else "Nighttime"

                # C. Compute metrics for this sample using the city-specific ROI.
                # acc_t, acc_t2, result = calculate_accu(label[b:b+1], outputs[b:b+1], city=city)
                ## C. Compute metrics using only the first channel (index 0).
                # label[b:b+1] has shape (1, C, H, W).
                # Slicing with 0:1 changes the shape to (1, 1, H, W), retaining only the first channel.
                acc_t, acc_t2, result = calculate_accu(label[b:b+1, 0:1], outputs[b:b+1, 0:1], city=city)
                
                # D. Store the metrics.
                for target in [period, "Total"]:
                    stats[target]["acc1"].append(acc_t)
                    stats[target]["acc2"].append(acc_t2)
                    stats[target]["accuracy"].append(result['accuracy'])
                    stats[target]["prec"].append(result['precision']) 
                    stats[target]["recall"].append(result['recall']) 
                    stats[target]["f1"].append(result['f1'])
                    stats[target]["iou"].append(result['iou'])

            # E. Save visualization results.
            if do_vis:
                for b in range(outputs.size(0)):  
                    for c in range(outputs.size(1)):
                        raw_pred = outputs[b, c].cpu().numpy()
                        pred_idx = np.clip(np.round(raw_pred), 0, 3).astype(int)
                        label_idx = label[b, c].cpu().numpy().astype(int)
                        valid_mask = (label_idx != -1)
                        
                        def create_colored_image(class_map, is_valid_mask):
                            h, w = class_map.shape
                            color_img = np.zeros((h, w, 3), dtype=np.uint8)
                            white_condition = ((class_map == 0) | (class_map == 1)) & is_valid_mask
                            color_img[white_condition] = [255, 255, 255]
                            return color_img

                        pred_rgb = create_colored_image(pred_idx, valid_mask)
                        label_rgb = create_colored_image(label_idx, valid_mask)

                        def draw_roi_box(img, r_center, c_center, r, thickness=3):
                            H, W, _ = img.shape
                            r_start, r_end = max(0, r_center - r), min(H, r_center + r)
                            c_start, c_end = max(0, c_center - r), min(W, c_center + r)
                            color = [255, 0, 0] 
                            img[r_start:r_start+thickness, c_start:c_end] = color
                            img[r_end-thickness:r_end, c_start:c_end] = color
                            img[r_start:r_end, c_start:c_start+thickness] = color
                            img[r_start:r_end, c_end-thickness:c_end] = color
                            return img

                        pred_rgb = draw_roi_box(pred_rgb, center_row, center_col, radius)
                        label_rgb = draw_roi_box(label_rgb, center_row, center_col, radius)

                        # Extract the output filename.
                        if isinstance(files_name, (list, tuple)) and isinstance(files_name[c], (list, tuple)):
                            fname = os.path.basename(str(files_name[c][b]))
                        else:
                            val = files_name[b]
                            if isinstance(val, torch.Tensor):
                                if val.numel() == 1:
                                    raw_name = str(val.item())
                                else:
                                    val_list = val.detach().cpu().tolist()
                                    def flatten(lst):
                                        out = []
                                        if isinstance(lst, list):
                                            for item in lst:
                                                out.extend(flatten(item) if isinstance(item, list) else [item])
                                        else:
                                            out.append(lst)
                                        return out
                                    flat_vals = flatten(val_list)
                                    raw_name = "_".join(map(str, flat_vals))
                            elif isinstance(val, (list, tuple)):
                                raw_name = "_".join(map(str, val))
                            else:
                                raw_name = str(val)
                            fname = os.path.basename(raw_name)
                        ##############################################
                        # ==========================================
                        # -------- Save prediction and ground-truth images --------
                        # ==========================================
                        # 1. Define and create the prediction and ground-truth folders.
                        pred_folder = os.path.join(output_folder, 'pred')
                        gt_folder = os.path.join(output_folder, 'gt')
                        os.makedirs(pred_folder, exist_ok=True)
                        os.makedirs(gt_folder, exist_ok=True)

                        # 2. Skip files that already exist to avoid redundant I/O.
                        pred_path = os.path.join(pred_folder, f'{fname}.png')
                        gt_path = os.path.join(gt_folder, f'{fname}.png')
                        if not os.path.exists(pred_path):
                            plt.imsave(pred_path, pred_rgb)
                        if not os.path.exists(gt_path):
                            plt.imsave(gt_path, label_rgb)

                        concat_path = os.path.join(output_folder, f'{fname}_concat.png')
                        if not os.path.exists(concat_path):
                            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                            axs[0].imshow(pred_rgb); axs[0].set_title('Prediction'); axs[0].axis('off')
                            axs[1].imshow(label_rgb); axs[1].set_title('Ground Truth'); axs[1].axis('off')
                            fig.savefig(concat_path, bbox_inches='tight', pad_inches=0)
                            plt.close(fig)

    # 2. Compute mean metrics and save the results.
    final_return = (0, 0, 0, 0, 0, 0, 0)
    for p_name in ["Daytime", "Nighttime", "Total"]:
        d = stats[p_name]
        if len(d["acc1"]) > 0:
            ############################
            m_acc1 = sum(d["acc1"]) / len(d["acc1"])
            m_acc2 = sum(d["acc2"]) / len(d["acc2"])
            m_accuracy = sum(d["accuracy"]) / len(d["accuracy"])
            m_prec = sum(d["prec"]) / len(d["prec"])
            m_recall = sum(d["recall"]) / len(d["recall"])
            m_f1 = sum(d["f1"]) / len(d["f1"])
            m_iou = sum(d["iou"]) / len(d["iou"])
            save_name = f"{model_name}_{p_name}_{test_months_str}"
            save_to_csv(save_name, m_acc1, m_acc2, m_accuracy, 
                        m_prec, m_recall, m_f1, m_iou, seq_len, pred_len, city)
            
            if p_name == "Total":
                final_return = (m_acc1, m_acc2, m_accuracy, m_prec, m_recall, m_f1, m_iou)

    return final_return

# --- CSV saving function ---
def save_to_csv(model_name, mean_acc, mean_acc2, accuracy_avg, prec_avg, recall_avg, f1_avg, iou_avg, seq_len, pred_len, city, csv_file=None):
    # If csv_file is not specified, select the default path according to the city.
    if csv_file is None:
        if city == 'Nanjing':
            csv_file = './csv_results/Nanjing_results_1024.csv'
        elif city == 'Changchun':
            csv_file = './csv_results/Changchun_results_1024.csv'
        else:
            csv_file = f'./csv_results/{city}_results_1024.csv'
            
    # Ensure that the output directory exists.
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['model_name','seq_len', 'pred_len', 'city', 'mean_acc', 'mean_acc2', 'accuracy_avg', 'prec_avg', 'recall_avg', 'f1_avg', 'iou_avg'])
        writer.writerow([model_name, seq_len, pred_len, city, mean_acc, mean_acc2, accuracy_avg, prec_avg, recall_avg, f1_avg, iou_avg])
        print(f"Results saved to {csv_file}")
