import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import logging
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_auc_score, confusion_matrix, log_loss,matthews_corrcoef,precision_recall_curve, auc,
                             classification_report, accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import argparse
import sys
sys.path.append("./")
from helper_functions import *

from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite'.*")
warnings.filterwarnings("ignore", message=".*n_jobs value.*")
current_dir = os.path.dirname(os.path.abspath(__file__))


class SelfCrossAttentionDualStreamTransformer(nn.Module):
    def __init__(self, protein_dim, molecule_dim, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers with proper layer normalization
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        self.molecule_proj = nn.Sequential(
            nn.Linear(molecule_dim, embed_dim),
            nn.LayerNorm(embed_dim))

        # Self-attention for protein
        self.protein_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        # Self-attention for molecule
        self.molecule_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Cross-attention layers
        self.protein_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.molecule_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Layer norms
        self.protein_norm1 = nn.LayerNorm(embed_dim)
        self.molecule_norm1 = nn.LayerNorm(embed_dim)

        self.protein_norm2 = nn.LayerNorm(embed_dim)
        self.molecule_norm2 = nn.LayerNorm(embed_dim)

        self.protein_norm3 = nn.LayerNorm(embed_dim)
        self.molecule_norm3 = nn.LayerNorm(embed_dim)

        # FFNs
        self.protein_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.molecule_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, protein_emb, molecule_emb):
        # Project inputs
        protein = self.protein_proj(protein_emb)
        molecule = self.molecule_proj(molecule_emb)

        # Protein self-attention
        protein_self = self.protein_self_attn(protein, protein, protein, need_weights=False)[0]
        # Molecule self-attention
        molecule_self = self.molecule_self_attn(molecule, molecule, molecule, need_weights=False)[0]
        # Add residual connection and apply layer norm for protein
        protein1 = self.protein_norm1(protein + protein_self)
        # Add residual connection and apply layer norm for molecule
        molecule1 = self.molecule_norm1(molecule + molecule_self)

        # Protein Cross attention
        protein_cross = self.protein_cross_attn(protein1, molecule1, molecule1, need_weights=False)[0]
        # Molecule Cross attention
        molecule_cross = self.molecule_cross_attn(molecule1, protein1, protein1, need_weights=False)[0]
        # Add residual connection and apply layer norm for protein
        protein2 = self.protein_norm2(protein1 + protein_cross)
        # Add residual connection and apply layer norm for molecule
        molecule2 = self.molecule_norm2(molecule1 + molecule_cross)

        # FFNs
        protein3 = self.protein_norm3(protein2 + self.protein_ffn(protein2))
        molecule3 = self.molecule_norm3(molecule2 + self.molecule_ffn(molecule2))

        return protein3, molecule3


class LxMert(nn.Module):
    def __init__(self, protein_dim, molecule_dim, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers with proper layer normalization
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        self.molecule_proj = nn.Sequential(
            nn.Linear(molecule_dim, embed_dim),
            nn.LayerNorm(embed_dim))

        # Self-attention for protein
        self.protein_self_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Self-attention for molecule
        self.molecule_self_attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Self-attention for protein
        self.protein_self_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Self-attention for molecule
        self.molecule_self_attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Cross-attention layers
        self.protein_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)
        self.molecule_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Layer norms
        self.protein_norm1 = nn.LayerNorm(embed_dim)
        self.molecule_norm1 = nn.LayerNorm(embed_dim)

        self.protein_norm2 = nn.LayerNorm(embed_dim)
        self.molecule_norm2 = nn.LayerNorm(embed_dim)

        self.protein_norm3 = nn.LayerNorm(embed_dim)
        self.molecule_norm3 = nn.LayerNorm(embed_dim)

        self.protein_norm4 = nn.LayerNorm(embed_dim)
        self.molecule_norm4 = nn.LayerNorm(embed_dim)

        self.protein_norm5 = nn.LayerNorm(embed_dim)
        self.molecule_norm5 = nn.LayerNorm(embed_dim)

        # FFNs_1
        self.protein_ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.molecule_ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        # FFNs_2
        self.protein_ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

        self.molecule_ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, protein_emb, molecule_emb):
        # Project inputs
        protein = self.protein_proj(protein_emb)
        molecule = self.molecule_proj(molecule_emb)

        # Protein self-attention
        protein_self1 = self.protein_self_attn1(protein, protein, protein, need_weights=False)[0]
        # Molecule self-attention
        molecule_self1 = self.molecule_self_attn1(molecule, molecule, molecule, need_weights=False)[0]

        # Add residual connection and apply layer norm for protein
        protein1 = self.protein_norm1(protein + protein_self1)
        # Add residual connection and apply layer norm for molecule
        molecule1 = self.molecule_norm1(molecule + molecule_self1)

        # FFNs_1 followed by residual connection and apply layer norm for molecule
        protein2 = self.protein_norm2(protein1 + self.protein_ffn1(protein1))
        molecule2 = self.molecule_norm2(molecule1 + self.molecule_ffn1(molecule1))

        # Protein Cross attention
        protein_cross = self.protein_cross_attn(protein2, molecule2, molecule2, need_weights=False)[0]
        # Molecule Cross attention
        molecule_cross = self.molecule_cross_attn(molecule2, protein2, protein2, need_weights=False)[0]

        # Add residual connection and apply layer norm for protein
        protein3 = self.protein_norm3(protein2 + protein_cross)
        # Add residual connection and apply layer norm for molecule
        molecule3 = self.molecule_norm3(molecule2 + molecule_cross)

        # Protein self-attention
        protein_self2 = self.protein_self_attn2(protein3, protein3, protein3, need_weights=False)[0]
        # Molecule self-attention
        molecule_self2 = self.molecule_self_attn2(molecule3, molecule3, molecule3, need_weights=False)[0]

        # Add residual connection and apply layer norm for protein
        protein4 = self.protein_norm4(protein3 + protein_self2)
        # Add residual connection and apply layer norm for molecule
        molecule4 = self.molecule_norm4(molecule3 + molecule_self2)

        # FFNs_2 followed by residual connection and apply layer norm
        protein5 = self.protein_norm5(protein4 + self.protein_ffn2(protein4))
        molecule5 = self.molecule_norm5(molecule4 + self.molecule_ffn2(molecule4))

        return protein5, molecule5


def visualize_embeddings(embeddings, labels, epoch=None, title_suffix=""):
    """
    t-SNE plot without numeric axis ticks (cleaner visualization).
    Retains axis labels ('t-SNE 1' and 't-SNE 2') for orientation.
    """
    # Convert to numpy if tensors
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    embed_2d = tsne.fit_transform(embeddings)

    # Create figure
    plt.figure(figsize=(10, 8), dpi=300)

    # Class styling (same as before)
    class_info = {
        0: {'name': 'enzyme-inhibitor', 'color': '#FF6B00', 'marker': 'o'},
        1: {'name': 'enzyme-substrate', 'color': '#AA00FF', 'marker': 's'},
        2: {'name': 'enzyme-non-substrate', 'color': '#00CC66', 'marker': '^'}
    }

    # Plot each class
    for class_id, style in class_info.items():
        mask = labels == class_id
        plt.scatter(
            embed_2d[mask, 0],  # t-SNE 1 (x-axis)
            embed_2d[mask, 1],  # t-SNE 2 (y-axis)
            c=style['color'],
            marker=style['marker'],
            s=40,
            label=style['name'],
            alpha=1.0,
            edgecolors='k',
            linewidths=0.5,

        )

    # --- Key Change: Remove tick numbers ---
    plt.xticks([])  # Remove x-axis tick labels
    plt.yticks([])  # Remove y-axis tick labels

    # Axis titles (kept for orientation)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)

    # Legend and title
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        edgecolor='black'
    )
    title = f"Embeddings at Epoch {epoch}" if epoch else "Embeddings"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    # Log to WandB
    wandb.log({"embedding_visualization": wandb.Image(plt)})
    plt.close()


# def visualize_embeddings(embeddings, labels, epoch=None, title_suffix=""):
#     """
#     UMAP plot without numeric axis ticks (cleaner visualization).
#     Retains axis labels ('UMAP 1' and 'UMAP 2') for orientation.
#     """
#     # Convert to numpy if tensors
#     if torch.is_tensor(embeddings):
#         embeddings = embeddings.cpu().numpy()
#     if torch.is_tensor(labels):
#         labels = labels.cpu().numpy()
#
#     # Compute UMAP
#     reducer = umap.UMAP(random_state=42, n_jobs=-1)
#     embed_2d = reducer.fit_transform(embeddings)
#
#     # Create figure
#     plt.figure(figsize=(10, 8), dpi=300)
#
#     # Class styling (same as before)
#     class_info = {
#         0: {'name': 'enzyme-inhibitor', 'color': '#FF6B00', 'marker': 'o'},
#         1: {'name': 'enzyme-substrate', 'color': '#AA00FF', 'marker': 's'},
#         2: {'name': 'enzyme-non-substrate', 'color': '#00CC66', 'marker': '^'}
#     }
#
#     # Plot each class
#     for class_id, style in class_info.items():
#         mask = labels == class_id
#         plt.scatter(
#             embed_2d[mask, 0],  # UMAP 1 (x-axis)
#             embed_2d[mask, 1],  # UMAP 2 (y-axis)
#             c=style['color'],
#             marker=style['marker'],
#             s=40,
#             label=style['name'],
#             alpha=1.0,
#             edgecolors='k',
#             linewidths=0.5,
#         )
#
#     # --- Key Change: Remove tick numbers ---
#     plt.xticks([])  # Remove x-axis tick labels
#     plt.yticks([])  # Remove y-axis tick labels
#
#     # Axis titles (kept for orientation)
#     plt.xlabel('UMAP 1', fontsize=12)
#     plt.ylabel('UMAP 2', fontsize=12)
#
#     # Legend and title
#     plt.legend(
#         bbox_to_anchor=(1.02, 1),
#         loc='upper left',
#         frameon=True,
#         edgecolor='black'
#     )
#     title = f"Embeddings at Epoch {epoch}" if epoch else "Embeddings"
#     if title_suffix:
#         title += f" {title_suffix}"
#     plt.title(title, fontsize=14, pad=20)
#     plt.grid(True, alpha=0.2)
#     plt.tight_layout()
#
#     # Log to WandB
#     wandb.log({"embedding_visualization": wandb.Image(plt)})
#     plt.close()


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for protein, molecule, labels in dataloader:
            protein, molecule, labels = protein.to(device), molecule.to(device), labels.to(device)
            # Forward pass to get embeddings
            _, embeddings = model(protein, molecule)
            # Save embeddings and labels
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    # Concatenate all embeddings and labels
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_embeddings, all_labels


def get_initial_embeddings(dataset, device='cpu'):
    """Extracts and concatenates raw embeddings without model processing"""
    chemberta_embs = []
    esm2t30_embs = []
    labels = []

    for i in range(len(dataset)):
        if len(dataset[i]) == 3:  # With labels
            chem, esm, lbl = dataset[i]
            labels.append(lbl)
        else:  # Without labels
            chem, esm = dataset[i]

        chemberta_embs.append(chem)
        esm2t30_embs.append(esm)

    # Stack and concatenate
    chem_tensor = torch.stack(chemberta_embs).to(device)
    esm_tensor = torch.stack(esm2t30_embs).to(device)
    combined = torch.cat([chem_tensor, esm_tensor], dim=-1)  # Concatenate along feature dimension

    if labels:
        return combined, torch.stack(labels).to(device)
    return combined, None


def plot_roc_curves(interaction_preds,
                    subclass_preds,
                    interaction_labels,
                    subclass_labels,
                    split_tech,
                    random_state=None,
                    save_path=None):
    plt.figure(figsize=(10, 8))
    # Ensure predictions are 1D arrays
    interaction_preds = interaction_preds.squeeze()
    subclass_preds = subclass_preds.squeeze()
    # 1. Interaction Head ROC Curve (all samples)
    fpr_i, tpr_i, _ = roc_curve(interaction_labels, interaction_preds)
    roc_auc_i = roc_auc_score(interaction_labels, interaction_preds)
    plt.plot(fpr_i, tpr_i, color='blue', lw=2,
             label=f'Interaction Head (AUC = {roc_auc_i:.2f})')
    # 2. Subclass Head ROC Curve (only interacting pairs)
    interacting_mask = (interaction_labels == 1)
    if interacting_mask.sum() > 0:
        fpr_s, tpr_s, _ = roc_curve(subclass_labels[interacting_mask],
                                    subclass_preds[interacting_mask])
        roc_auc_s = roc_auc_score(subclass_labels[interacting_mask],
                                  subclass_preds[interacting_mask])
        plt.plot(fpr_s, tpr_s, color='green', lw=2,
                 label=f'Subclass Head (AUC = {roc_auc_s:.2f})')
    # Plot formatting
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curves for {split_tech} split with random seed:{random_state}')
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def calculate_all_metrics(interaction_preds, subclass_preds,
                          interaction_labels, subclass_labels):

    # 1. Calculate interaction head metrics
    binary_interaction = (interaction_preds >= 0.5).astype(int)
    interaction_cm = confusion_matrix(interaction_labels, binary_interaction)
    interaction_metrics = {
        'accuracy': accuracy_score(interaction_labels, binary_interaction),
        'roc_auc': roc_auc_score(interaction_labels, interaction_preds),
        'f1': f1_score(interaction_labels, binary_interaction),
        'confusion_matrix': interaction_cm
    }

    # 2. Calculate subclass head metrics
    interacting_mask = (interaction_labels == 1)
    subclass_cm = np.zeros((2, 2))  # Initialize empty matrix
    subclass_preds_binary=None
    if interacting_mask.sum() > 0:
        subclass_preds_binary = (subclass_preds[interacting_mask] >= 0.5).astype(int)
        subclass_cm = confusion_matrix(subclass_labels[interacting_mask],
                                       subclass_preds_binary)

    subclass_metrics = {
        'accuracy': accuracy_score(subclass_labels[interacting_mask],
                                   subclass_preds_binary) if interacting_mask.sum() > 0 else 0,
        'roc_auc': roc_auc_score(subclass_labels[interacting_mask],
                                 subclass_preds[interacting_mask]) if interacting_mask.sum() > 0 else 0,
        'f1': f1_score(subclass_labels[interacting_mask],
                       subclass_preds_binary) if interacting_mask.sum() > 0 else 0,
        'confusion_matrix': subclass_cm
    }

    # 3. Calculate combined overall metrics by element-wise sum
    overall_cm = interaction_cm + subclass_cm

    # Calculate combined accuracy
    combined_accuracy = np.trace(overall_cm) / np.sum(overall_cm)

    combined_metrics = {
        'confusion_matrix': overall_cm,
        'accuracy': combined_accuracy,
        'roc_auc': (interaction_metrics['roc_auc'] + subclass_metrics['roc_auc']) / 2,
        'f1': (interaction_metrics['f1'] + subclass_metrics['f1']) / 2
    }

    return {
        'interaction': interaction_metrics,
        'subclass': subclass_metrics,
        'overall': combined_metrics
    }


def set_seed(seed=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False






