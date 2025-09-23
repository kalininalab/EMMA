import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import logging
import wandb
from torch.utils.data import DataLoader, Dataset
from torch_optimizer import Lookahead
from sklearn.model_selection import train_test_split
import argparse
from pytorch_toolbelt.losses import BinaryFocalLoss
import sys

sys.path.append("./../utilities")
from training_functions import *
from helper_functions import *
from thresholds import *

PARAMS_MAP = {
    "C2": C2_params,
    "C1e": C1e_params,
    "C1f": C1e_params,
    "C1": C1_params
}

current_dir = os.path.dirname(os.path.abspath(__file__))


class EMMA(nn.Module):
    def __init__(self, protein_dim, molecule_dim, embed_dim, num_heads=None):
        super(EMMA, self).__init__()

        self.dual_stream_transformer = SelfCrossAttentionDualStreamTransformer(
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
            nn.Linear(1024, 1)
        )

        self.subclass_head = nn.Sequential(
            nn.Linear(2 * embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, protein_emb, molecule_emb):
        protein_attn_out, molecule_attn_out = self.dual_stream_transformer(protein_emb, molecule_emb)
        protein_features = protein_attn_out.squeeze(1)
        molecule_features = molecule_attn_out.squeeze(1)
        combined_features = torch.cat([protein_features, molecule_features], dim=-1)
        interaction_logits = self.interaction_head(combined_features)
        subclass_logits = self.subclass_head(combined_features)

        return interaction_logits, subclass_logits


class EnzMolDataset(Dataset):
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


def multi_task_loss(interaction_logits, subclass_logits, interaction_labels, subclass_labels, split_tech):
    params = PARAMS_MAP[split_tech]
    interaction_gamma = params["loss"]["interaction_gamma"]
    subclass_gamma = params["loss"]["subclass_gamma"]
    interaction_alpha = params["loss"]["interaction_alpha"]  # For interaction head
    subclass_alpha = params["loss"]["subclass_alpha"]  # For subclass head
    reduction = params["loss"]["reduction"]

    # Reshape logits and labels to ensure proper dimensions
    interaction_logits = interaction_logits.view(-1)
    interaction_labels = interaction_labels.float().view(-1)

    # Focal Loss for Interaction Head
    focal_loss_fn = BinaryFocalLoss(gamma=interaction_gamma, alpha=interaction_alpha, reduction=reduction)
    interaction_loss = focal_loss_fn(
        interaction_logits,
        interaction_labels,
    )

    # Focal Loss for Subclass Head
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
    return interaction_loss + subclass_loss


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


def validating(model, dataloader, device, split_tech):
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
    set_seed(args.current_seed)
    split_tech = args.used_split_tech
    params = PARAMS_MAP[split_tech]
    batch_size = params["training"]["batch_size"]
    initial_lr = params["training"]["initial_lr"]
    weight_decay = params["training"]["weight_decay"]
    max_epochs = params["training"]["max_epochs"]
    # All path
    models_path = os.path.join(current_dir, "..", "data", "trained_models")
    encoder_path = os.path.join(models_path, f"encoder_RS{args.current_seed}_{split_tech}.pth")
    inter_path = os.path.join(models_path, f"interHead_RS{args.current_seed}_{split_tech}.pth")
    sub_path = os.path.join(models_path, f"subHead_RS{args.current_seed}_{split_tech}.pth")
    results_path = os.path.join(current_dir, "..", "data", "training_test_results")

    wandb.init(project=f'EMMA', entity='vahid-atabaigi',
               name=f"{args.protein_column_name}_{args.molecule_column_name}_{split_tech}_RS{args.current_seed}")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_path,f"{split_tech}_RS{args.current_seed}.log")),
            logging.StreamHandler()
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    df_data = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"train_{split_tech}_2S.pkl"))
    df_train, df_val = train_test_split(df_data,
                                        test_size=0.2,
                                        random_state=args.current_seed,
                                        stratify=df_data['Binding'])
    train_dataset = EnzMolDataset(df_train, args.protein_column_name, args.molecule_column_name)
    val_dataset = EnzMolDataset(df_val, args.protein_column_name, args.molecule_column_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Initialize model
    model = EMMA(
        protein_dim=640,  # ESM2 embedding size
        molecule_dim=768,  # MolFormer embedding size
        embed_dim=768,
        num_heads=32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    best_val_loss = float('inf')
    patience = 3
    no_improvement = 0
    for epoch in range(max_epochs):
        train_loss = training(model, train_loader, optimizer, device, split_tech)
        val_loss, val_i_preds, val_s_preds, val_i_labels, val_s_labels = validating(
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
            torch.save(model.dual_stream_transformer.state_dict(), os.path.join(models_path,f"encoder_RS{args.current_seed}_{split_tech}.pth"))
            torch.save(model.interaction_head.state_dict(), os.path.join(models_path, f"interHead_RS{args.current_seed}_{split_tech}.pth"))
            torch.save(model.subclass_head.state_dict(), os.path.join(models_path, f"subHead_RS{args.current_seed}_{split_tech}.pth"))
            no_improvement = 0
        else:
            no_improvement += 1

        # Early stopping
        if no_improvement >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Final test evaluation
    df_test = pd.read_pickle(os.path.join(current_dir, "..", "data", "splits", f"test_{split_tech}_2S.pkl"))
    test_dataset = EnzMolDataset(df_test, args.protein_column_name, args.molecule_column_name)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # How to load model for validation
    loaded_model = EMMA(
        protein_dim=640,
        molecule_dim=768,
        embed_dim=768,
        num_heads=32
    ).to(device)

    # Load saved weights
    loaded_model.dual_stream_transformer.load_state_dict(torch.load(encoder_path, map_location=device))
    loaded_model.interaction_head.load_state_dict(torch.load(inter_path, map_location=device))
    loaded_model.subclass_head.load_state_dict(torch.load(sub_path, map_location=device))

    test_loss, test_i_preds, test_s_preds, test_i_labels, test_s_labels = validating(loaded_model,
                                                                                     test_loader,
                                                                                     device,
                                                                                     split_tech)
    np.save(os.path.join(results_path, f"interaction_y_test_pred_{split_tech}_RS{args.current_seed}.npy"), test_i_preds)
    np.save(os.path.join(results_path, f"interaction_y_test_true_{split_tech}_RS{args.current_seed}.npy"), test_i_labels)
    np.save(os.path.join(results_path, f"subclass_y_test_pred_{split_tech}_RS{args.current_seed}.npy"), test_s_preds)
    np.save(os.path.join(results_path, f"subclass_y_test_true_{split_tech}_RS{args.current_seed}.npy"), test_s_labels)

    # Calculate all metrics
    test_metrics = calculate_all_metrics(test_i_preds, test_s_preds, test_i_labels, test_s_labels)

    # Plot ROC curves
    os.makedirs(os.path.join(results_path, "roc_plots"), exist_ok=True)
    plot_path = os.path.join(results_path, "roc_plots", f"auc_roc_curves_{split_tech}_RS{args.current_seed}.png")
    plot_roc_curves(test_i_preds,
                    test_s_preds,
                    test_i_labels,
                    test_s_labels,
                    split_tech,
                    random_state=args.current_seed,
                    save_path=plot_path)

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
    parser.add_argument('--random_seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999],
                       help='List of random seeds (default: [42, 123, 456, 789, 999])')
    args = parser.parse_args()

    for seed in args.random_seeds:
        print(f"\n=== Running with seed {seed} ===")
        args.current_seed = seed
        main(args)
