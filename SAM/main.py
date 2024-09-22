import torch
import argparse
import torch.optim as optim
import numpy as np
from ClipSamSegmenter import ClipSamIntegrator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Dataset import CustomDataset, load_data
from model import ImageTransformer 
from train import train
from base_model import device
from config import batch_size, num_classes, num_epochs, num_sites, learning_rate, sche_milestones, gamma, l2, embedding_dim
from config import full_train_data_path, full_val_data_path, full_test_data_path
from helpful.vis_metrics import plots, DoAna
from SamClipSegment import *
import tqdm

seed = 42  # You can choose any integer seed value
np.random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Model training parameters")

    # Add arguments for each hyperparameter
    parser.add_argument('--num_classes', type=int, default=num_classes, help="Number of classes")
    parser.add_argument('--num_sites', type=int, default=num_sites, help="Number of sites")
    parser.add_argument('--embedding_dim', type=int, default=embedding_dim, help="Embedding dimension")
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help="Learning rate")
    parser.add_argument('--shape', type=int, default=450, help="Learning rate")
    parser.add_argument('--n_heads', type=int, default=8, help="n_heads")
    parser.add_argument('--feedforward', type=int, default=512, help="feedforward")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--n_layers', type=int, default=8, help="n_layers")

    parser.add_argument('--num_epochs', type=int, default=num_epochs, help="Number of epochs")
    parser.add_argument('--l2', type=float, default=l2, help="L2 regularization")
    parser.add_argument('--batch_size', type=int, default=batch_size, help="Batch size")
    parser.add_argument('--gamma', type=float, default=gamma, help="Gamma")
    parser.add_argument('--optim', type=str, default="AdamW", help="Optimizer")

    parser.add_argument('--full_train_data_path', type=str, default=full_train_data_path, help="Full train data path")
    parser.add_argument('--full_val_data_path', type=str, default=full_val_data_path, help="Full validation data path")
    parser.add_argument('--full_test_data_path', type=str, default=full_test_data_path, help="Full test data path")
    parser.add_argument('--base', type=str, default='densenet', help="Base model")

    # Boolean flags
    parser.add_argument('--freeze', action='store_true', help="Freeze True")
    parser.add_argument('--to_freeze', type=int, default=0, help="parameters to freeze")
    parser.add_argument('--compile', action='store_true', help="Compile")

    return parser.parse_args()

args = parse_args()
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")
print(f"Used Device {device})")

""" --------------------------------------------------------------------"""

# Define the transformations based on the description provided
transform = transforms.Compose([
        transforms.Resize((args.shape, args.shape)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

test_transform = transforms.Compose([
    transforms.Resize((args.shape, args.shape)),
    transforms.ToTensor(),                                      # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

stra_train_data, idx_to_class, idx_to_site = load_data(args.full_train_data_path, False)
stra_test_data, _, _ = load_data(args.full_test_data_path, False)
stra_val_data, _, _ = load_data(args.full_val_data_path, False)
print(idx_to_site)

print(f"Used Device {device})")

train_set = CustomDataset(stra_train_data, transform, "train_distribution", oversample =False, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=False, ignore=False)
val_set = CustomDataset(stra_test_data, transform, "val_distribution", oversample = False, idx_to_class=idx_to_class, idx_to_site=idx_to_site, save_augmented=False, ignore=False)
test_set = CustomDataset(stra_test_data, test_transform, title = "test_distribution", oversample=False, ignore=False)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory =True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers = 4, pin_memory =True)

torch.cuda.empty_cache()

""" ----------------------------------------------------------------------------------------"""

def run_segmentation(loader, segmenter, prompt):
    segmenter.clip_model.eval()  # Set CLIP to evaluation mode
    for batch_idx, (images, labels, sites) in enumerate(loader):
        images = images.to(segmenter.device) 
        
        # Create a list of prompts, one for each image in the batch
        print(f"start segmentation...")
        batch_size = images.size(0)
        prompts = [prompt] * batch_size  # This should create a list of strings
        print(f"Prompts: {prompts}")
        # prompts_list = [[p] for p in prompts]
        
        # Get CLIP similarity scores for the batch
        similarities = segmenter.get_clip_embeddings(prompts, images)
        print(f"Batch {batch_idx + 1}: Similarities: {similarities}")
        
        first_image_np = images[0].cpu().numpy().transpose(1, 2, 0)  # Convert to NumPy (H, W, C) format
        
        input_point = np.array([[100, 100]])  
        input_label = np.array([1])  
        
        mask = segmenter.segment_image(first_image_np, input_point, input_label)
        segmenter.visualize_segmented_mask(mask)

if __name__ == '__main__':
    
    output_dir = "/Users/yomnaamuhammedd/C-D-G-Mouth_and_oral-disease/SAM/SegmentedImages"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch_idx, (images, labels, sites) in enumerate(train_loader):
        print(f"Processing batch {batch_idx + 1} of {len(train_loader)}")
        
        for img_idx, (image, label, site) in enumerate(zip(images, labels, sites)):
            print(f"  Processing image {img_idx + 1} of batch {batch_idx + 1}")

            risk_level = "high risk" if label == 2 else "low risk" if label == 1 else "normal"
        
            # Print the label and site
            label = labels[img_idx].item()  
            site = sites[img_idx]  
            print(f"    Label: {label}, Site: {site}")

            img_np = image.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy array (H, W, C)
            img_np = (img_np * 0.5 + 0.5) * 255
            img_np = img_np.astype(np.uint8)  # Convert to uint8 for proper image format
            print(f"Image shape after conversion: {img_np.shape}")

            prompt = f"Focus only on segmenting any abnormal tissue in the {site} area of the mouth for a {risk_level} patient."
            print(prompt)

            try:
                print("Segmenting the image...")
                segmented_image = segment(
                    predicted_iou_threshold=0.97, 
                    stability_score_threshold=0.9, 
                    clip_threshold=0.85, 
                    image=img_np, 
                    query=prompt
                )
                print("Segmentation done.")

                # Save both the original and segmented image with label and site in the filename
                original_image_path = os.path.join(output_dir, f"original_batch_{batch_idx + 1}_img_{img_idx + 1}.png")
                PIL.Image.fromarray(img_np).save(original_image_path)
                print(f"Original image saved at: {original_image_path}")

                output_path = os.path.join(output_dir, f"segmented_batch_{batch_idx + 1}_img_{img_idx + 1}_label_{label}_site_{site}.png")
                segmented_image.save(output_path)
                print(f"Segmented image saved at: {output_path}")

            except Exception as e:
                print(f"Error processing image {img_idx + 1} in batch {batch_idx + 1}: {e}")