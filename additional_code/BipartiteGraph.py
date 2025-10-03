import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import time
import matplotlib
from os.path import join
from matplotlib.gridspec import GridSpec
import os
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr


# ==============================================
# Graph Construction
# ==============================================
def load_data_and_build_graph(path):
    CURRENT_DIR = Path(os.getcwd())
    OUTPUT_DIR = join(CURRENT_DIR, "..", "data", "final_dataset_analysis")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data and constructing graph...")
    data = pd.read_pickle(path)
    df = data[["Uni_SwissProt", "molecule_ID", "Binding"]]

    B = nx.Graph()
    enzyme_nodes = [(node, {'bipartite': 0}) for node in df["Uni_SwissProt"].unique()]
    molecule_nodes = [(node, {'bipartite': 1}) for node in df["molecule_ID"].unique()]
    B.add_nodes_from(enzyme_nodes + molecule_nodes)

    edges = [(row["Uni_SwissProt"], row["molecule_ID"], {'binding': row["Binding"]})
             for _, row in df.iterrows()]
    B.add_edges_from(edges)

    assert nx.is_bipartite(B), "Graph is not bipartite!"
    print(f"Network contains {B.number_of_nodes()} nodes and {B.number_of_edges()} edges")

    return B, OUTPUT_DIR


# ==============================================
# Exact Betweenness Centrality
# ==============================================
def calculate_exact_betweenness(graph):
    """Calculate exact betweenness centrality using NetworkX"""
    print("\n" + "=" * 60)
    print("CALCULATING EXACT BETWEENNESS CENTRALITY")
    print("=" * 60)
    print(f"Graph size: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print("Approximate time for each dataframe: 0.30 to 1.26 hours in Mac M1")
    print("=" * 60)
    start_time = time.time()
    betweenness = nx.betweenness_centrality(graph, normalized=False)
    duration = time.time() - start_time
    hours = duration / 3600
    print(f"\nExact betweenness calculation completed!")
    print(f"          : {hours:.2f} hours")
    return betweenness, duration


# ==============================================
#  degree
# ==============================================
def calculate_nodes_degrees(graph):
    """
    Calculate the degree of each node in the graph.

    Parameters:
    graph (nx.Graph): NetworkX graph object

    Returns:
    dict: Dictionary with node IDs as keys and their degrees as values
    """
    print("\n" + "=" * 60)
    print("CALCULATING NODE DEGREES")
    print("=" * 60)

    start_time = time.time()

    # Calculate degrees for all nodes
    degrees = dict(graph.degree())

    duration = time.time() - start_time
    print(f"Degree calculation completed in {duration:.4f} seconds!")
    print(f"Total nodes: {len(degrees)}")

    return degrees, duration

# ==============================================
# Analysis
# ==============================================


def analyze_betweenness(B, betweenness, OUTPUT_DIR,df_name):
    enzymes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    molecules = set(B.nodes()) - enzymes

    enzyme_betweenness = {k: v for k, v in betweenness.items() if k in enzymes}
    molecule_betweenness = {k: v for k, v in betweenness.items() if k in molecules}

    # Statistical Analysis
    def print_stats(b_dict, name):
        values = np.array(list(b_dict.values()))
        stats = {
            'Count': len(values),
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Q1': np.percentile(values, 25),
            'Q3': np.percentile(values, 75)
        }
        print(f"\n{name} Betweenness Statistics:")
        for k, v in stats.items():
            print(f"- {k}: {v:.6f}")

    print_stats(enzyme_betweenness, "Enzyme")
    print_stats(molecule_betweenness, "Molecule")

    # Save Results
    results_df = pd.DataFrame({
        'Node': list(enzyme_betweenness.keys()) + list(molecule_betweenness.keys()),
        'Betweenness': list(enzyme_betweenness.values()) + list(molecule_betweenness.values()),
        'Type': ['Enzyme'] * len(enzyme_betweenness) + ['Molecule'] * len(molecule_betweenness)
    })
    results_df.to_csv(join(OUTPUT_DIR, f'Betweenness_values_{df_name}.csv'), index=False)

    print(f"\nAnalysis complete! Files saved to: {OUTPUT_DIR}")


def analyze_degree(B, degrees, OUTPUT_DIR, df_name):
    # Separate nodes by type
    enzymes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
    molecules = set(B.nodes()) - enzymes

    enzyme_degrees = {k: v for k, v in degrees.items() if k in enzymes}
    molecule_degrees = {k: v for k, v in degrees.items() if k in molecules}

    # Statistical Analysis
    def print_stats(degree_dict, name):
        values = np.array(list(degree_dict.values()))
        stats = {
            'Count': len(values),
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Q1': np.percentile(values, 25),
            'Q3': np.percentile(values, 75)
        }
        print(f"\n{name} Degree Statistics:")
        for k, v in stats.items():
            print(f"- {k}: {v:.6f}")

    print_stats(enzyme_degrees, "Enzyme")
    print_stats(molecule_degrees, "Molecule")

    # Save Results in same format
    results_df = pd.DataFrame({
        'Node': list(enzyme_degrees.keys()) + list(molecule_degrees.keys()),
        'Degree': list(enzyme_degrees.values()) + list(molecule_degrees.values()),
        'Type': ['Enzyme'] * len(enzyme_degrees) + ['Molecule'] * len(molecule_degrees)
    })
    results_df.to_csv(join(OUTPUT_DIR, f'Degree_values_{df_name}.csv'), index=False)

    # print(f"\nDegree analysis complete! Files saved to: {OUTPUT_DIR}")

    return enzyme_degrees, molecule_degrees


# ==============================================
#Visualization
# ==============================================

def plot_betweenness_degrees(path, metric=None):
    # plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    OUTPUT_DIR = path

    # Define splits - now with 4 splits
    splits = [("R", "Random"), ("C1f", "enzyme-based"), ("C1e", "small molecule-based"), ("C1", "Label-based"),
              ("C2", "two-dimensional")]
    # splits = [("R", "Random"), ("C1f", "enzyme-based"), ("C1e", "small molecule-based"), ("C1", "Label-based")]
    colors = {'Enzyme': '#fc8d62', 'Small molecule': '#8da0cb'}

    # Create a single comprehensive figure with 4 rows and 4 columns
    fig = plt.figure(figsize=(10, 12))  # Increased height to accommodate 4 rows
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1], hspace=0.4, wspace=0.35)

    # Store all data for consistent axis limits
    all_values = []
    all_kde_max_values = []

    # First, collect all data to set consistent axis limits
    for s in splits:
        test = pd.read_csv(join(OUTPUT_DIR, f"{metric}_values_test_{s[0]}_2S.csv"))
        train = pd.read_csv(join(OUTPUT_DIR, f"{metric}_values_train_{s[0]}_2S.csv"))
        # train_set = pd.read_pickle(join("..", "data", "splits", f"train_{s[0]}_2S.pkl"))
        # test_set = pd.read_pickle(join("..", "data", "splits", f"test_{s[0]}_2S.pkl"))

        if metric == 'Betweenness':
            test[metric] = np.log(test[metric] + 1)
            train[metric] = np.log(train[metric] + 1)
        else:
            test[metric] = np.log(test[metric])
            train[metric] = np.log(train[metric])
        test['Type'] = test['Type'].replace('Molecule', 'Small molecule')
        train['Type'] = train['Type'].replace('Molecule', 'Small molecule')
        all_values.extend(train[metric].tolist())
        all_values.extend(test[metric].tolist())

        # Calculate KDE max for this split
        def calculate_kde_max(data):
            from scipy.stats import gaussian_kde
            if len(data) > 1:
                kde = gaussian_kde(data)
                x = np.linspace(data.min(), data.max(), 100)
                y = kde(x)
                return y.max()
            return 0

        for dataset in [train, test]:
            for node_type in ['Enzyme', 'Small molecule']:
                data_subset = dataset[dataset['Type'] == node_type][metric]
                if len(data_subset) > 0:
                    kde_max = calculate_kde_max(data_subset)
                    all_kde_max_values.append(kde_max)

    # Calculate global limits
    global_kde_max = max(all_kde_max_values) * 1.2
    global_box_min = min(all_values)
    global_box_max = max(all_values) * 1.05

    # Calculate minimum x limit - ensure no negative values
    global_x_min = max(0, min(all_values) * 0.9)  # Start at 0 or slightly below min

    # Now create the actual plots
    row = 0
    for s in splits:
        test = pd.read_csv(join(OUTPUT_DIR, f"{metric}_values_test_{s[0]}_2S.csv"))
        train = pd.read_csv(join(OUTPUT_DIR, f"{metric}_values_train_{s[0]}_2S.csv"))
        test[metric] = np.log(test[metric] + 1)
        train[metric] = np.log(train[metric] + 1)
        test['Type'] = test['Type'].replace('Molecule', 'Small molecule')
        train['Type'] = train['Type'].replace('Molecule', 'Small molecule')

        # Add Split identifier to datasets
        train['Dataset'] = 'Train'
        test['Dataset'] = 'Test'

        # KDE Plot for Train set - Column 0
        ax_kde_train = fig.add_subplot(gs[row, 0])
        sns.kdeplot(data=train, x=metric, hue='Type', fill=True,
                    palette=colors, alpha=0.6, linewidth=1,
                    ax=ax_kde_train, common_norm=False)

        ax_kde_train.set_title(f'{s[1].title()} - Train', fontsize=8, pad=4)

        # Set x-axis limits to prevent negative values
        ax_kde_train.set_xlim(global_x_min, None)

        # Only show x label for bottom row (row 3)
        if row == 4:
            if metric == 'Betweenness':
                ax_kde_train.set_xlabel(f'BC score (log + 1 scaled)')
            else:
                ax_kde_train.set_xlabel(f'{metric} score (log scaled)')
        else:
            ax_kde_train.set_xlabel('')

        # Show y label for all rows in column 0
        ax_kde_train.set_ylabel('Density')

        ax_kde_train.set_ylim(0, global_kde_max)
        ax_kde_train.grid(True, linestyle=':', alpha=0.3)

        # KDE Plot for Test set - Column 1
        ax_kde_test = fig.add_subplot(gs[row, 1])
        sns.kdeplot(data=test, x=metric, hue='Type', fill=True,
                    palette=colors, alpha=0.6, linewidth=1,
                    ax=ax_kde_test, common_norm=False)

        ax_kde_test.set_title(f'{s[1].title()} - Test', fontsize=8, pad=4)

        # Set x-axis limits to prevent negative values
        ax_kde_test.set_xlim(global_x_min, None)

        # Only show x label for bottom row (row 3)
        if row == 4:
            if metric == 'Betweenness':
                ax_kde_test.set_xlabel(f'BC score (log + 1 scaled)')
            else:
                ax_kde_test.set_xlabel(f'{metric} score (log scaled)')
        else:
            ax_kde_test.set_xlabel('')

        # Don't show y label for column 1
        ax_kde_test.set_ylabel('')

        ax_kde_test.set_ylim(0, global_kde_max)
        ax_kde_test.grid(True, linestyle=':', alpha=0.3)

        # Add legend only to the first KDE plot
        if row == 0:
            legend_elements = [
                Patch(facecolor=colors['Enzyme'], label='Enzyme'),
                Patch(facecolor=colors['Small molecule'], label='Small molecule')
            ]
            ax_kde_train.legend(handles=legend_elements, title='Type', loc='upper right')
        else:
            ax_kde_train.legend_.remove()
            ax_kde_test.legend_.remove()

        # Boxplot for Train set - Column 2
        ax_box_train = fig.add_subplot(gs[row, 2])
        sns.boxplot(data=train, x='Type', y=metric, hue='Type',
                    palette=colors, showfliers=False, width=0.4,
                    ax=ax_box_train, linewidth=0.5, dodge=False, medianprops=dict(color='red', linewidth=1.5))

        ax_box_train.set_title(f'{s[1].title()} - Train', fontsize=8, pad=4)

        # Show y label for all rows in column 2
        if metric == 'Betweenness':
            ax_box_train.set_ylabel(f'BC score (log + 1 scaled)')
        else:
            ax_box_train.set_ylabel(f'{metric} score (log scaled)')


        ax_box_train.set_xlabel('')
        ax_box_train.set_ylim(global_box_min, global_box_max)
        ax_box_train.grid(True, linestyle=':', alpha=0.3)

        # Remove legend from boxplot
        if ax_box_train.get_legend():
            ax_box_train.get_legend().remove()

        # Boxplot for Test set - Column 3
        ax_box_test = fig.add_subplot(gs[row, 3])
        sns.boxplot(data=test, x='Type', y=metric, hue='Type',
                    palette=colors, showfliers=False, width=0.4,
                    ax=ax_box_test, linewidth=0.5, dodge=False, medianprops=dict(color='red', linewidth=1.5))

        ax_box_test.set_title(f'{s[1].title()} - Test', fontsize=8, pad=4)

        # Don't show y label for column 3
        ax_box_test.set_ylabel('')

        ax_box_test.set_xlabel('')
        ax_box_test.set_ylim(global_box_min, global_box_max)
        ax_box_test.grid(True, linestyle=':', alpha=0.3)

        # Remove legend from boxplot
        if ax_box_test.get_legend():
            ax_box_test.get_legend().remove()

        row += 1

    # Add panel labels - now with 16 panels (4 rows × 4 columns)
    # Add panel labels - now with 20 panels (5 rows × 4 columns)
    panel_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
    axes = fig.get_axes()
    for i, ax in enumerate(axes):
        # Position labels differently for KDE and box plots
        if i % 4 < 2:  # KDE plots (columns 0 and 1)
            ax.text(-0.15, 1.05, panel_labels[i], transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top')
        else:  # Box plots (columns 2 and 3)
            ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top')

    plt.tight_layout()
    output_path = join(OUTPUT_DIR, f'{metric}_analysis_combined.pdf')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Combined analysis figure saved to: {output_path}")


def binning_betweenness(heads, splits, bins):
    results = {}
    for s in splits:
        for head in heads:
            print(f"\n{'=' * 60}")
            print(f"Analyzing split: {s[0]}, head: {head} (Using {bins} bins)")
            print(f"{'=' * 60}")

            try:
                # Load data
                test_btwn = pd.read_csv(
                    os.path.join("..", "data", "final_dataset_analysis", f"Betweenness_values_test_{s[0]}_2S.csv"))
                test_set = pd.read_pickle(os.path.join("..", "data", "splits", f"test_{s[0]}_2S.pkl"))
                test_true = np.load(
                    os.path.join("..", "data", "training_test_results", f"{head}_y_test_true_{s[0]}_RS42.npy"))
                test_pred = np.load(
                    os.path.join("..", "data", "training_test_results", f"{head}_y_test_pred_{s[0]}_RS42.npy"))

                # Process betweenness
                test_btwn["Betweenness"] = np.log(test_btwn["Betweenness"] + 1)
                betweenness_map = dict(zip(test_btwn["Node"], test_btwn["Betweenness"]))
                test_set["Betweenness_enzyme"] = test_set["Uni_SwissProt"].map(betweenness_map)
                test_set["Betweenness_molecule"] = test_set["molecule_ID"].map(betweenness_map)

                # Add predictions and filter out -1 labels
                test_set["true_label"] = test_true
                test_set["pred_prob"] = test_pred
                test_set = test_set[test_set["true_label"] != -1].reset_index(drop=True)

                # Analyze for enzyme and molecule betweenness
                split_results = {}

                for betweenness_type in ['Betweenness_enzyme', 'Betweenness_molecule']:
                    print(f"\n--- {betweenness_type} ---")

                    # Create bins using global parameter
                    test_set['bin'] = pd.cut(test_set[betweenness_type], bins=bins, labels=range(1, bins + 1))

                    bin_results = []
                    # Check all possible bins from 1 to bins
                    for bin in range(1, bins + 1):
                        bin_data = test_set[test_set['bin'] == bin]

                        if len(bin_data) > 0:
                            if len(np.unique(bin_data['true_label'])) > 1:
                                auroc = roc_auc_score(bin_data['true_label'], bin_data['pred_prob'])
                                n_samples = len(bin_data)
                                betweenness_mean = bin_data[betweenness_type].mean()
                                bin_results.append({
                                    'bin': bin,
                                    'n_samples': n_samples,
                                    'betweenness_mean': betweenness_mean,
                                    'auroc': auroc
                                })

                                print(
                                    f"  BIN {bin}: AUROC = {auroc:.3f}, Samples = {n_samples}, Mean Betweenness = {betweenness_mean:.3f}")
                            else:
                                print(f"  BIN {bin}: SKIPPED - Only one class present, Samples = {len(bin_data)}")
                        else:
                            print(f"  BIN {bin}: No samples")
                    split_results[betweenness_type] = pd.DataFrame(bin_results)

                # Store results with the full tuple as key
                if s not in results:
                    results[s] = {}
                results[s][head] = split_results

            except Exception as e:
                print(f"Error processing split {s[0]}, head {head}: {e}")
                continue

    return results


# ============================================================================
# COMBINED PLOTS WITH CORRELATION TABLES (r and p-value in rows, splits in columns)
# ============================================================================


def plot_binned_btweenness(results, splits, heads, bins,OUTPUT_DIR):
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    # Create a 2x2 grid: interaction (enzyme, molecule) and subclass (enzyme, molecule)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # Define colors for different splits
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Calculate correlation data for tables
    correlation_data = {}
    for head in heads:
        correlation_data[head] = {}
        for betweenness_type in ['Betweenness_enzyme', 'Betweenness_molecule']:
            correlation_data[head][betweenness_type] = {}
            for s in splits:
                if s in results and head in results[s] and betweenness_type in results[s][head]:
                    df = results[s][head][betweenness_type]
                    if len(df) > 1:
                        correlation, _ = pearsonr(df['bin'], df['auroc'])
                        correlation_data[head][betweenness_type][s[1]] = correlation
                    else:
                        correlation_data[head][betweenness_type][s[1]] = np.nan

    # Plot 1: Interaction - Enzyme Betweenness
    ax1 = axes[0, 0]
    for i, s in enumerate(splits):
        if s in results and 'interaction' in results[s] and 'Betweenness_enzyme' in results[s]['interaction']:
            df = results[s]['interaction']['Betweenness_enzyme']
            ax1.plot(df['bin'], df['auroc'], 'o-', label=s[1], linewidth=2.5, markersize=6, alpha=0.8, color=colors[i])

    ax1.set_xlabel('Binned betweenness', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax1.set_title('a) Interaction head - Enzyme Betweenness', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.set_xticks(range(1, bins + 1))
    ax1.set_ylim(0.4, 1.0)

    # Add correlation table for Interaction - Enzyme (r only, splits in columns)
    split_names = [s[1] for s in splits]
    table_data = []
    row_data = []
    for split_name in split_names:
        if split_name in correlation_data['interaction']['Betweenness_enzyme']:
            r = correlation_data['interaction']['Betweenness_enzyme'][split_name]
            row_data.append(f"{r:.3f}")
        else:
            row_data.append("NaN")
    table_data.append(row_data)

    table_ax1 = ax1.inset_axes([0.09, -0.35, 0.85, 0.25])
    table_ax1.axis('off')
    table1 = table_ax1.table(cellText=table_data,
                             rowLabels=[' r '],
                             colLabels=split_names,
                             loc='center',
                             cellLoc='center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1.15, 1.5)

    # Plot 2: Interaction - Molecule Betweenness
    ax2 = axes[0, 1]
    for i, s in enumerate(splits):
        if s in results and 'interaction' in results[s] and 'Betweenness_molecule' in results[s]['interaction']:
            df = results[s]['interaction']['Betweenness_molecule']
            ax2.plot(df['bin'], df['auroc'], 'o-', label=s[1], linewidth=2.5, markersize=6, alpha=0.8, color=colors[i])

    ax2.set_xlabel('Binned betweenness', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax2.set_title('b) Interaction head - Molecule Betweenness', fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.set_xticks(range(1, bins + 1))
    ax2.set_ylim(0.4, 1.0)

    # Add correlation table for Interaction - Molecule (r only, splits in columns)
    table_data = []
    row_data = []
    for split_name in split_names:
        if split_name in correlation_data['interaction']['Betweenness_molecule']:
            r = correlation_data['interaction']['Betweenness_molecule'][split_name]
            row_data.append(f"{r:.3f}")
        else:
            row_data.append("NaN")
    table_data.append(row_data)

    table_ax2 = ax2.inset_axes([0.09, -0.35, 0.85, 0.25])
    table_ax2.axis('off')
    table2 = table_ax2.table(cellText=table_data,
                             rowLabels=[' r '],
                             colLabels=split_names,
                             loc='center',
                             cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1.15, 1.5)

    # Plot 3: Subclass - Enzyme Betweenness
    ax3 = axes[1, 0]
    for i, s in enumerate(splits):
        if s in results and 'subclass' in results[s] and 'Betweenness_enzyme' in results[s]['subclass']:
            df = results[s]['subclass']['Betweenness_enzyme']
            ax3.plot(df['bin'], df['auroc'], 'o-', label=s[1], linewidth=2.5, markersize=6, alpha=0.8, color=colors[i])

    ax3.set_xlabel('Binned betweenness', fontsize=12, fontweight='bold')
    ax3.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax3.set_title('c) Subclass head - Enzyme Betweenness', fontsize=14, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.3)
    ax3.set_xticks(range(1, bins + 1))
    ax3.set_ylim(0.85, 1.02)

    # Add correlation table for Subclass - Enzyme (r only, splits in columns)
    table_data = []
    row_data = []
    for split_name in split_names:
        if split_name in correlation_data['subclass']['Betweenness_enzyme']:
            r = correlation_data['subclass']['Betweenness_enzyme'][split_name]
            row_data.append(f"{r:.3f}")
        else:
            row_data.append("NaN")
    table_data.append(row_data)

    table_ax3 = ax3.inset_axes([0.09, -0.35, 0.85, 0.25])
    table_ax3.axis('off')
    table3 = table_ax3.table(cellText=table_data,
                             rowLabels=[' r '],
                             colLabels=split_names,
                             loc='center',
                             cellLoc='center')
    table3.auto_set_font_size(False)
    table3.set_fontsize(12)
    table3.scale(1.15, 1.5)

    # Plot 4: Subclass - Molecule Betweenness
    ax4 = axes[1, 1]
    for i, s in enumerate(splits):
        if s in results and 'subclass' in results[s] and 'Betweenness_molecule' in results[s]['subclass']:
            df = results[s]['subclass']['Betweenness_molecule']
            ax4.plot(df['bin'], df['auroc'], 'o-', label=s[1], linewidth=2.5, markersize=6, alpha=0.8, color=colors[i])

    ax4.set_xlabel('Binned betweenness', fontsize=12, fontweight='bold')
    ax4.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax4.set_title('d) Subclass head - Molecule Betweenness', fontsize=14, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.3)
    ax4.set_xticks(range(1, bins + 1))
    ax4.set_ylim(0.85, 1.02)

    # Add correlation table for Subclass - Molecule (r only, splits in columns)
    table_data = []
    row_data = []
    for split_name in split_names:
        if split_name in correlation_data['subclass']['Betweenness_molecule']:
            r = correlation_data['subclass']['Betweenness_molecule'][split_name]
            row_data.append(f"{r:.3f}")
        else:
            row_data.append("NaN")
    table_data.append(row_data)

    table_ax4 = ax4.inset_axes([0.09, -0.35, 0.85, 0.25])
    table_ax4.axis('off')
    table4 = table_ax4.table(cellText=table_data,
                             rowLabels=[' r '],
                             colLabels=split_names,
                             loc='center',
                             cellLoc='center')
    table4.auto_set_font_size(False)
    table4.set_fontsize(12)
    table4.scale(1.15, 1.5)

    plt.tight_layout()
    fig.legend(handles=[plt.Line2D([0], [0], color=colors[i], lw=2) for i, s in enumerate(splits)],
               labels=[s[1] for s in splits],
               loc='lower center',
               bbox_to_anchor=(0.5, - 0.025),
               ncol=len(splits),
               frameon=True,
               fancybox=True,
               shadow=True,
               fontsize=12)
    output_path = join(OUTPUT_DIR, f'Betweenness_bin_analysis.pdf')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
