import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import os
import random
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import csv
import random
import torch
from tqdm import tqdm
def main():
    transform = transforms.Compose([
            transforms.ToTensor()
            ])
    from datasets import PolygonDataset
    from models import LATENT_DIM, LinearClassifier
    from training.base import evaluate_model

    device = "cuda" 
    model_fixed_op = LinearClassifier(block_size=14, num_classes=9).to(device)
    mode="fixed_op"
    save_dir = f"checkpoints/mnist_yshiftt_new_arch_cls_{mode}"
    best_ckpt = torch.load(os.path.join  (save_dir, "best.pth"))
    model_fixed_op.load_state_dict(best_ckpt['model_state_dict'])

    trained_degrees = [24, 26, 0, 2, 4]
    val_set   = PolygonDataset("mnist_dataset_split.csv", split="val",   
                                transform=transform,
                                transform_type="shift_y",
                                keep_degrees=trained_degrees)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)


    def evaluate_with_refs(model, loader, ref_images, device, k):
        model.eval()
        
        # ref_images: (K, D) - Used only for finding the best rotation
        ref_images = ref_images.to(device)
        K = ref_images.size(0) 
        
        total = 0
        correct = 0
        correct_deg = 0

        with torch.no_grad():
            for img_anchor, d, cls, _, _ in tqdm(loader, leave=False):
                img_anchor = img_anchor.to(device)
                d = d.to(device)
                cls = cls.to(device)
                B = img_anchor.size(0)

                emb14, pred14 = model(img_anchor, skip_transform=False, degs=None) # shape: (B*14, D)
        
                dists = torch.cdist(emb14, ref_images.unsqueeze(0)) # (B, 14, 200)
                best_rot_indices = []
                for i in range(B):
                    # Find top 3 closest refs across all 14 rotations
                    top3_flat_idx = dists[i].view(-1).topk(k, largest=False).indices
                    rot_votes = top3_flat_idx // K 
                    best_rot_indices.append(rot_votes.mode().values.item())
                best_rot = torch.tensor(best_rot_indices, device=device)
                pred = pred14.reshape(B, 14, -1)[torch.arange(B), best_rot].argmax(-1)
                correct_deg += (best_rot==(14-d//2)%14).sum().item()
                correct += (pred == cls).sum().item()
                total += B
        return correct / total, correct_deg/total

    # 1. Collect all items into one flat list
    all_items = []
    for img_anchor, deg_anchor, _, _, _ in tqdm(val_loader):
        for i in range(img_anchor.size(0)):
            all_items.append((img_anchor[i], deg_anchor[i]))

    file_path_knn = "knn_experiment_results_yshift.csv"
    headers = ["transform", "model",  "N", "k", "seed", "degree", "accuracy1","accuracy2"]

    completed_tasks = set()

    # 1. Load existing results into a lookup set
    if os.path.exists(file_path_knn):
        with open(file_path_knn, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 8:
                    # Key: (Mode1, Mode2, N, k, seed, degree)
                    # indices: 0, 1, 2, 3, 4, 5
                    task_key = (row[0], row[1], row[2], row[3], row[4], row[5])
                    completed_tasks.add(task_key)

    if not os.path.exists(file_path_knn):
        with open(file_path_knn, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    num_workers = os.cpu_count() 
   
   
    ds = PolygonDataset(
        csv_file="mnist_dataset_split.csv",
        split="test",
        transform=transform,
        transform_type="shift_y",
        # keep_degrees=[d],
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4)

                    


    for N in [100, 200, 500, 1000, 2000, 5000]:
        for k in [1, 3, 10, 30, 100, 300]:
            if k>N:
                continue
            for seed in [0, 10, 20, 30, 42]:
                    
                    # 2. Sample 200 items once from the entire pool
                    random.seed(seed)
                    samples = random.sample(all_items, N)

                    # 3. Batch process
                    imgs = torch.stack([s[0] for s in samples]).to(device)
                    degs = torch.tensor([s[1] for s in samples]).to(device)

                    with torch.no_grad():
                        # Canonical rotation indexes and forward pass
                        emb, _ = model_fixed_op(imgs, skip_transform=False, degs=-(degs // 2))

                    # Store as a single embedding tensor
                    ref_embeddings = emb.cpu()
                    d=0
                    m1, m2 = "yshift", "fixed"
                
                    # Create a key using the exact strings you will write to the CSV
                    current_key = (m1, m2, str(N), str(k), str(seed), str(d))
                    
                    if current_key in completed_tasks:
                        continue  # Skip to next iteration
                    
                    acc1, acc2 = evaluate_with_refs(model_fixed_op, loader, ref_embeddings, device, k=k)
                    with open(file_path_knn, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["yshift", "fixed", N, k, seed, d, acc1, acc2])

    model_learned_op =LinearClassifier(block_size=LATENT_DIM, num_classes=9, learnable_P=True).to(device)

    mode="learned_op"
    save_dir = f"checkpoints/mnist_yshiftt_new_arch_cls_{mode}"
    best_ckpt = torch.load(os.path.join  (save_dir, "best.pth"))
    model_learned_op.load_state_dict(best_ckpt['model_state_dict'])




    for N in [100, 200, 500, 1000, 2000, 5000]:
        for k in [1, 3, 10, 30, 100, 300]:
            if k>N:
                continue
            for seed in [0, 10, 20, 30, 42]:
                    # 2. Sample 200 items once from the entire pool
                    random.seed(seed)
                    samples = random.sample(all_items, N)

                    # 3. Batch process
                    imgs = torch.stack([s[0] for s in samples]).to(device)
                    degs = torch.tensor([s[1] for s in samples]).to(device)

                    with torch.no_grad():
                        # Canonical rotation indexes and forward pass
                        emb, _ = model_learned_op(imgs, skip_transform=False, degs=-(degs // 2))

                    # Store as a single embedding tensor
                    ref_embeddings = emb.cpu()

                    d=0
                    m1, m2 = "yshift", "learned"
                
                    # Create a key using the exact strings you will write to the CSV
                    current_key = (m1, m2, str(N), str(k), str(seed), str(d))
                    
                    if current_key in completed_tasks:
                        continue  # Skip to next iteration
                    
                    acc1, acc2 = evaluate_with_refs(model_learned_op, loader, ref_embeddings, device, k=k)
                    with open(file_path_knn, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["yshift", "learned", N, k, seed, d, acc1, acc2])
if __name__ == '__main__':
    main()
