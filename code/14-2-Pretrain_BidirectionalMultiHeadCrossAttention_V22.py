# V22 two head
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import logging
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import argparse
from pytorch_toolbelt.losses import BinaryFocalLoss
import sys

sys.path.append("./../utilities")
from training_functions import *
from helper_functions import *
from thresholds import *

PARAMS_MAP = {
    "I2": I2_params,
    "C2": C2_params,
    "C1e": C1e_params,
    "C1f": C1e_params
}

current_dir = os.path.dirname(os.path.abspath(__file__))


class InteractionModel(nn.Module):
    def __init__(self, protein_dim, molecule_dim, embed_dim, num_heads=None):
        super(InteractionModel, self).__init__()

        self.cross_attention = NewTransformerXHCrossAttention(
            protein_dim,
            molecule_dim,
            embed_dim,
            num_heads
        )

        self.interaction_head = nn.Sequential(
            nn.Linear(2 * embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )

        self.subclass_head = nn.Sequential(
            nn.Linear(2 * embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1)
        )

    def forward(self, protein_emb, molecule_emb):
        protein_attn_out, molecule_attn_out = self.cross_attention(protein_emb, molecule_emb)
        protein_features = protein_attn_out.squeeze(1)
        molecule_features = molecule_attn_out.squeeze(1)
        combined_features = torch.cat([protein_features, molecule_features], dim=-1)
        interaction_logits = self.interaction_head(combined_features)
        subclass_logits = self.subclass_head(combined_features)

        return interaction_logits, subclass_logits


class InteractionDataset(Dataset):
    def __init__(self, df, protein_column, molecule_column):
        self.protein_embeddings = [torch.tensor(x).float() for x in df[protein_column]]
        self.molecule_embeddings = [torch.tensor(x).float() for x in df[molecule_column]]
        self.interaction_labels = torch.tensor(df["Mainclass"].values).long()
        self.subclass_labels = torch.tensor(df["Subclass"].values).long()

    def __len__(self):
        return len(self.interaction_labels)

    def __getitem__(self, idx):
        return (
            self.protein_embeddings[idx],
            self.molecule_embeddings[idx],
            self.interaction_labels[idx],
            self.subclass_labels[idx]
        )


def collate_fn(batch):
    protein = torch.stack([item[0] for item in batch])
    molecule = torch.stack([item[1] for item in batch])
    interaction_labels = torch.stack([item[2] for item in batch])
    subclass_labels = torch.stack([item[3] for item in batch])
    return protein, molecule, interaction_labels, subclass_labels


def multi_task_loss(interaction_logits, subclass_logits, interaction_labels, subclass_labels, split_tech):
    params = PARAMS_MAP[split_tech]
    interaction_gamma = params["loss"]["interaction_gamma"]
    subclass_gamma = params["loss"]["subclass_gamma"]
    interaction_alpha = params["loss"]["interaction_alpha"]  # For interaction head
    subclass_alpha = params["loss"]["subclass_alpha"]  # For subclass head
    reduction = params["loss"]["reduction"]
    subclass_weight = params["loss"]["subclass_weight"]

    # Reshape logits and labels to ensure proper dimensions
    interaction_logits = interaction_logits.view(-1)
    interaction_labels = interaction_labels.float().view(-1)

    # (1-2) Focal Loss for Interaction Head
    focal_loss_fn = BinaryFocalLoss(gamma=interaction_gamma, alpha=interaction_alpha, reduction=reduction)
    interaction_loss = focal_loss_fn(
        interaction_logits,
        interaction_labels,
    )

    # (2-2) Focal Loss for Subclass Head
    mask = (interaction_labels == 1)  # Only for interacting pairs
    if mask.sum() > 0:
        subclass_logits = subclass_logits[mask].view(-1)
        subclass_labels = subclass_labels[mask].float().view(-1)
        # Use FocalLoss with subclass_alpha for the subclass head
        subclass_focal_loss_fn = BinaryFocalLoss(gamma=subclass_gamma, alpha=subclass_alpha, reduction=reduction)
        subclass_loss = subclass_focal_loss_fn(
            subclass_logits,
            subclass_labels
        )
    else:
        subclass_loss = 0.0
    return interaction_loss + subclass_weight * subclass_loss


def training(model, dataloader, optimizer, device, split_tech):
    model.train()
    total_loss = 0

    for protein, molecule, interaction_labels, subclass_labels in dataloader:
        protein = protein.to(device)
        molecule = molecule.to(device)
        interaction_labels = interaction_labels.to(device)
        subclass_labels = subclass_labels.to(device)

        optimizer.zero_grad()
        interaction_logits, subclass_logits = model(protein, molecule)

        loss = multi_task_loss(interaction_logits, subclass_logits, interaction_labels, subclass_labels, split_tech)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device, split_tech):
    model.eval()
    total_loss = 0
    all_interaction_preds = []
    all_subclass_preds = []
    all_interaction_labels = []
    all_subclass_labels = []

    with torch.no_grad():
        for protein, molecule, interaction_labels, subclass_labels in dataloader:
            protein = protein.to(device)
            molecule = molecule.to(device)
            interaction_labels = interaction_labels.to(device)
            subclass_labels = subclass_labels.to(device)

            interaction_logits, subclass_logits = model(protein, molecule)
            loss = multi_task_loss(interaction_logits, subclass_logits,
                                 interaction_labels, subclass_labels, split_tech)
            total_loss += loss.item()

            # Store predictions and labels
            all_interaction_preds.append(torch.sigmoid(interaction_logits).cpu().numpy())
            all_subclass_preds.append(torch.sigmoid(subclass_logits).cpu().numpy())
            all_interaction_labels.append(interaction_labels.cpu().numpy())
            all_subclass_labels.append(subclass_labels.cpu().numpy())

    # Concatenate all batches
    all_interaction_preds = np.concatenate(all_interaction_preds)
    all_subclass_preds = np.concatenate(all_subclass_preds)
    all_interaction_labels = np.concatenate(all_interaction_labels)
    all_subclass_labels = np.concatenate(all_subclass_labels)

    return (
        total_loss / len(dataloader),
        all_interaction_preds,
        all_subclass_preds,
        all_interaction_labels,
        all_subclass_labels
    )


def main(args):
    split_tech = args.used_split_tech
    params = PARAMS_MAP[split_tech]
    batch_size = params["training"]["batch_size"]
    initial_lr = params["training"]["initial_lr"]
    weight_decay = params["training"]["weight_decay"]
    max_epochs = params["training"]["max_epochs"]
    wandb.init(project='SIP-MTL', entity='vahid-atabaigi',
               name=f"MTL_{args.protein_column_name}_{args.molecule_column_name}_{split_tech}")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"MTL_{split_tech}.log"),
            logging.StreamHandler()
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df_data = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"train_{split_tech}_2S.pkl"))
    df_train, df_val = train_test_split(df_data, test_size=0.2, random_state=42, stratify=df_data['Binding'])
    # train_path = os.path.join(current_dir, "..", "data", "splits", f"training_{split_tech}_2S.pkl")
    # val_path = os.path.join(current_dir, "..", "data", "splits", f"val_{split_tech}_2S.pkl")
    # if os.path.exists(train_path) and os.path.exists(val_path):
    #     df_train = pd.read_pickle(train_path)
    #     df_val = pd.read_pickle(val_path)
    # else:
    #     # For C1f epsilon=0.05, delta=0.05
    #     # For C1e epsilon=0.1, delta=0.05
    #     # For I2 epsilon=0.01, delta=0.01
    #     df_train, df_val = datasail_split(df_data, split_tech, split_size=[8, 2],
    #                                       stratification=True, epsilon=0.05, delta=0.05)
    #     # df_train, df_val = customized_data_split(df_data, train_size=0.8)
    #     # df_train, df_val = train_test_split(df_data, test_size=0.2, random_state=42, stratify=df_data['Binding'])
    #
    #     os.makedirs(os.path.dirname(train_path), exist_ok=True)
    #     df_train.to_pickle(train_path)
    #     df_val.to_pickle(val_path)
    #
    # df_train = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"train_{split_tech}_3S.pkl"))
    # df_val = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"val_{split_tech}_3S.pkl"))

    train_dataset = InteractionDataset(df_train, args.protein_column_name, args.molecule_column_name)
    val_dataset = InteractionDataset(df_val, args.protein_column_name, args.molecule_column_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = InteractionModel(
        protein_dim=640,  # ESM2 embedding size
        molecule_dim=768,  # ChemBERTa embedding size
        embed_dim=768,
        num_heads=32
    ).to(device)

    # Optimizer and scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    # optimizer = torch.optim.NAdam(model.parameters(), lr=initial_lr, weight_decay=weight_decay, momentum_decay=6e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    no_improvement = 0

    for epoch in range(max_epochs):
        train_loss = training(model, train_loader, optimizer, device, split_tech)
        val_loss, val_i_preds, val_s_preds, val_i_labels, val_s_labels = validate(
            model, val_loader, device, split_tech)
        scheduler.step(val_loss)

        # Calculate all metrics
        all_metrics = calculate_all_metrics(val_i_preds, val_s_preds,val_i_labels, val_s_labels)

        # Logging
        logging.info(f"\nEpoch {epoch + 1}")
        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Interaction head
        logging.info("\nInteraction Head Metrics:")
        logging.info(f"Accuracy: {all_metrics['interaction']['accuracy']:.4f}")
        logging.info(f"ROC-AUC: {all_metrics['interaction']['roc_auc']:.4f}")
        logging.info(f"F1: {all_metrics['interaction']['f1']:.4f}")
        logging.info(f"Confusion Matrix:\n{all_metrics['interaction']['confusion_matrix']}")

        # Subclass head
        logging.info("\nSubclass Head Metrics:")
        logging.info(f"Accuracy: {all_metrics['subclass']['accuracy']:.4f}")
        logging.info(f"ROC-AUC: {all_metrics['subclass']['roc_auc']:.4f}")
        logging.info(f"F1: {all_metrics['subclass']['f1']:.4f}")
        if all_metrics['subclass']['confusion_matrix'] is not None:
            logging.info(f"Confusion Matrix:\n{all_metrics['subclass']['confusion_matrix']}")

        # W&B logging
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            # Interaction metrics
            'interaction/accuracy': all_metrics['interaction']['accuracy'],
            'interaction/roc_auc': all_metrics['interaction']['roc_auc'],
            'interaction/f1': all_metrics['interaction']['f1'],
            # Subclass metrics
            'subclass/accuracy': all_metrics['subclass']['accuracy'],
            'subclass/roc_auc': all_metrics['subclass']['roc_auc'],
            'subclass/f1': all_metrics['subclass']['f1'],
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(current_dir, "..", "data","trained_models", f"best_model_MTL_{split_tech}.pth"))
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Final test evaluation
    df_test = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"test_{split_tech}_2S.pkl"))
    test_dataset = InteractionDataset(df_test, args.protein_column_name, args.molecule_column_name)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loss, test_i_preds, test_s_preds, test_i_labels, test_s_labels = validate(model, test_loader, device, split_tech)
    results_path = os.path.join(current_dir, "..", "data","training_results")
    np.save(os.path.join(results_path, f"interaction_y_test_pred_{split_tech}_2S.npy"), test_i_preds)
    np.save(os.path.join(results_path, f"interaction_y_test_true__{split_tech}_2S.npy"), test_i_labels)
    np.save(os.path.join(results_path, f"subclass_y_test_pred_{split_tech}_2S.npy"), test_s_preds)
    np.save(os.path.join(results_path, f"subclass_y_test_true__{split_tech}_2S.npy"), test_s_labels)

    # Calculate all metrics
    test_metrics = calculate_all_metrics(test_i_preds, test_s_preds, test_i_labels, test_s_labels)

    # Plot ROC curves
    os.makedirs("roc_plots", exist_ok=True)
    plot_path = f"auc_roc_curves_{split_tech}.png"
    plot_roc_curves(test_i_preds, test_s_preds, test_i_labels, test_s_labels, split_tech, save_path=plot_path)

    # Log the plot to wandb
    wandb.log({"ROC Curves": wandb.Image(plot_path)})

    # Logging
    logging.info("\n=== Final Test Results ===")
    logging.info(f"Test Loss: {test_loss:.4f}")

    # Interaction head
    logging.info("\nInteraction Head Metrics:")
    logging.info(f"Accuracy: {test_metrics['interaction']['accuracy']:.4f}")
    logging.info(f"ROC-AUC: {test_metrics['interaction']['roc_auc']:.4f}")
    logging.info(f"F1: {test_metrics['interaction']['f1']:.4f}")
    logging.info(f"Confusion Matrix:\n{test_metrics['interaction']['confusion_matrix']}")

    # Subclass head
    logging.info("\nSubclass Head Metrics:")
    logging.info(f"Accuracy: {test_metrics['subclass']['accuracy']:.4f}")
    logging.info(f"ROC-AUC: {test_metrics['subclass']['roc_auc']:.4f}")
    logging.info(f"F1: {test_metrics['subclass']['f1']:.4f}")
    if test_metrics['subclass']['confusion_matrix'] is not None:
        logging.info(f"Confusion Matrix:\n{test_metrics['subclass']['confusion_matrix']}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MTL Interaction Model')
    parser.add_argument('--used_split_tech', type=str, required=True)
    parser.add_argument('--molecule_column_name', type=str, required=True)
    parser.add_argument('--protein_column_name', type=str, required=True)
    args = parser.parse_args()
    main(args)
