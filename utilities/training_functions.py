import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import logging
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, auc, accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
sys.path.append("./")
from helper_functions import *
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






