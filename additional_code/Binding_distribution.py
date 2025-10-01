import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_binding_distribution(final_dataset, final_dataset_BC, degrees, output_path):
    """
    Plot binding distribution with median degree (MD) and median betweenness centrality (MBC)

    Parameters:
    final_dataset: DataFrame with binding information
    final_dataset_BC: DataFrame with betweenness centrality values
    degrees: Dictionary of node degrees from calculate_nodes_degrees
    output_path: Path to save the output plot
    """

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 14,
        'axes.titlesize': 8,
        'axes.labelsize': 14,
        'legend.fontsize': 14,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    final_dataset_BC['Betweenness'] = np.log(final_dataset_BC['Betweenness'] + 1)

    df = final_dataset[["Uni_SwissProt", "molecule_ID", "Binding"]]
    df_betwns = final_dataset_BC[["Node", "Betweenness", "Type"]]

    # Ensure Node column is string type for consistent comparison
    df_betwns["Node"] = df_betwns["Node"].astype(str)

    # ==============================================

    # For enzyme diagram - convert to strings
    enzyme_non_substrate = set(df[df["Binding"] == 2]["Uni_SwissProt"].astype(str).unique())
    enzyme_substrate = set(df[df["Binding"] == 1]["Uni_SwissProt"].astype(str).unique())
    enzyme_inhibitor = set(df[df["Binding"] == 0]["Uni_SwissProt"].astype(str).unique())

    # For molecule diagram - convert to strings
    mol_non_substrate = set(df[df["Binding"] == 2]["molecule_ID"].astype(str).unique())
    mol_substrate = set(df[df["Binding"] == 1]["molecule_ID"].astype(str).unique())
    mol_inhibitor = set(df[df["Binding"] == 0]["molecule_ID"].astype(str).unique())

    # ==============================================

    def calculate_median_metrics(nodes_set, df_betwns, node_type, degrees_dict):
        """
        Calculate median betweenness and median degree for a set of nodes
        """
        # Filter by node type (case-insensitive)
        type_filtered = df_betwns[df_betwns["Type"].str.lower() == node_type.lower()]

        if type_filtered.empty:
            return 0, 0, 0, 0

        # Use merge for faster matching
        nodes_df = pd.DataFrame({'Node': list(nodes_set)})
        merged = pd.merge(nodes_df, type_filtered, on='Node', how='inner')

        if not merged.empty:
            # Calculate median betweenness
            betweenness_values = merged['Betweenness'].tolist()
            median_bet = np.median(betweenness_values)

            # Calculate median degree
            degree_values = []
            for node in nodes_set:
                if node in degrees_dict:
                    degree_values.append(degrees_dict[node])

            median_deg = np.median(degree_values) if degree_values else 0

            return median_bet, median_deg, len(betweenness_values), len(degree_values)
        else:
            return 0, 0, 0, 0

    # ==============================================

    # Calculate median metrics for all enzyme subsets
    enzyme_subsets = {
        'non-interacting': enzyme_non_substrate - enzyme_substrate - enzyme_inhibitor,
        'substrate': enzyme_substrate - enzyme_non_substrate - enzyme_inhibitor,
        'inhibitor': enzyme_inhibitor - enzyme_non_substrate - enzyme_substrate,
        'substrate_inhibitor': (enzyme_substrate & enzyme_inhibitor) - enzyme_non_substrate,
        'non-interacting_substrate': (enzyme_non_substrate & enzyme_substrate) - enzyme_inhibitor,
        'non-interacting_inhibitor': (enzyme_non_substrate & enzyme_inhibitor) - enzyme_substrate,
        'all_three': enzyme_non_substrate & enzyme_substrate & enzyme_inhibitor
    }

    enzyme_metrics = {}
    for subset_name, subset_nodes in enzyme_subsets.items():
        if subset_nodes:  # Only calculate if there are nodes
            median_bet, median_deg, bet_count, deg_count = calculate_median_metrics(
                subset_nodes, df_betwns, 'enzyme', degrees)
            enzyme_metrics[subset_name] = (median_bet, median_deg, bet_count, deg_count, len(subset_nodes))

    # Calculate median metrics for all molecule subsets
    molecule_subsets = {
        'non-interacting': mol_non_substrate - mol_substrate - mol_inhibitor,
        'substrate': mol_substrate - mol_non_substrate - mol_inhibitor,
        'inhibitor': mol_inhibitor - mol_non_substrate - mol_substrate,
        'substrate_inhibitor': (mol_substrate & mol_inhibitor) - mol_non_substrate,
        'non-interacting_substrate': (mol_non_substrate & mol_substrate) - mol_inhibitor,
        'non-interacting_inhibitor': (mol_non_substrate & mol_inhibitor) - mol_substrate,
        'all_three': mol_non_substrate & mol_substrate & mol_inhibitor
    }

    molecule_metrics = {}
    for subset_name, subset_nodes in molecule_subsets.items():
        if subset_nodes:  # Only calculate if there are nodes
            median_bet, median_deg, bet_count, deg_count = calculate_median_metrics(
                subset_nodes, df_betwns, 'molecule', degrees)
            molecule_metrics[subset_name] = (median_bet, median_deg, bet_count, deg_count, len(subset_nodes))

    # ==============================================

    # Set up the figure with larger size to accommodate annotations
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

    # Common style parameters
    venn_params = {
        'alpha': 0.7,
        'set_colors': ('#66c2a5', '#fc8d62', '#8da0cb')
    }

    # ========== Enzyme Plot ==========
    ax1 = fig.add_subplot(gs[0, 0])
    venn_enzyme = venn3(
        [enzyme_non_substrate, enzyme_substrate, enzyme_inhibitor],
        set_labels=('Non-interacting enzyme', 'Substrate-binding enzyme', 'Inhibitor-binding enzyme'),
        ax=ax1,
        **venn_params
    )
    venn3_circles(
        [enzyme_non_substrate, enzyme_substrate, enzyme_inhibitor],
        linestyle="dashed",
        linewidth=1.5,
        color="gray",
        ax=ax1
    )
    ax1.set_title('a', fontsize=20, pad=70, fontweight='bold')

    # Add annotations for MBC and MD
    for label in venn_enzyme.subset_labels:
        if label is not None:
            # Get the position of the count label
            pos = label.get_position()
            count_text = label.get_text()

            # Find which subset this label corresponds to
            subset_name = None
            if count_text == str(len(enzyme_non_substrate - enzyme_substrate - enzyme_inhibitor)):
                subset_name = 'non-interacting'
            elif count_text == str(len(enzyme_substrate - enzyme_non_substrate - enzyme_inhibitor)):
                subset_name = 'substrate'
            elif count_text == str(len(enzyme_inhibitor - enzyme_non_substrate - enzyme_substrate)):
                subset_name = 'inhibitor'
            elif count_text == str(len((enzyme_substrate & enzyme_inhibitor) - enzyme_non_substrate)):
                subset_name = 'substrate_inhibitor'
            elif count_text == str(len((enzyme_non_substrate & enzyme_substrate) - enzyme_inhibitor)):
                subset_name = 'non-interacting_substrate'
            elif count_text == str(len((enzyme_non_substrate & enzyme_inhibitor) - enzyme_substrate)):
                subset_name = 'non-interacting_inhibitor'
            elif count_text == str(len(enzyme_non_substrate & enzyme_substrate & enzyme_inhibitor)):
                subset_name = 'all_three'

            # Add MBC and MD annotations below the count
            if subset_name and subset_name in enzyme_metrics and enzyme_metrics[subset_name][2] > 0:
                median_bet, median_deg, bet_count, deg_count, total_count = enzyme_metrics[subset_name]
                annotation_text = f"MBC: {median_bet:.1f}\nMD: {median_deg:.1f}"
                ax1.annotate(annotation_text,
                             xy=(pos[0], pos[1] - 0.03),  # Position slightly below the count
                             xycoords='data',
                             ha='center', va='top', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # ========== Molecule Plot ==========
    ax2 = fig.add_subplot(gs[0, 1])
    venn_molecule = venn3(
        [mol_non_substrate, mol_substrate, mol_inhibitor],
        set_labels=('Non-interacting small molecule', 'Substrate', 'Inhibitor'),
        ax=ax2,
        **venn_params
    )
    venn3_circles(
        [mol_non_substrate, mol_substrate, mol_inhibitor],
        linestyle="dashed",
        linewidth=1.5,
        color="gray",
        ax=ax2
    )
    ax2.set_title('b', fontsize=20, pad=20, fontweight='bold')

    # Add MBC and MD annotations for molecule plot
    for label in venn_molecule.subset_labels:
        if label is not None:
            # Get the position of the count label
            pos = label.get_position()
            count_text = label.get_text()

            # Find which subset this label corresponds to
            subset_name = None
            if count_text == str(len(mol_non_substrate - mol_substrate - mol_inhibitor)):
                subset_name = 'non-interacting'
            elif count_text == str(len(mol_substrate - mol_non_substrate - mol_inhibitor)):
                subset_name = 'substrate'
            elif count_text == str(len(mol_inhibitor - mol_non_substrate - mol_substrate)):
                subset_name = 'inhibitor'
            elif count_text == str(len((mol_substrate & mol_inhibitor) - mol_non_substrate)):
                subset_name = 'substrate_inhibitor'
            elif count_text == str(len((mol_non_substrate & mol_substrate) - mol_inhibitor)):
                subset_name = 'non-interacting_substrate'
            elif count_text == str(len((mol_non_substrate & mol_inhibitor) - mol_substrate)):
                subset_name = 'non-interacting_inhibitor'
            elif count_text == str(len(mol_non_substrate & mol_substrate & mol_inhibitor)):
                subset_name = 'all_three'

            # Add MBC and MD annotations below the count
            if subset_name and subset_name in molecule_metrics and molecule_metrics[subset_name][2] > 0:
                median_bet, median_deg, bet_count, deg_count, total_count = molecule_metrics[subset_name]
                annotation_text = f"MBC: {median_bet:.1f}\nMD: {median_deg:.1f}"
                ax2.annotate(annotation_text,
                             xy=(pos[0], pos[1] - 0.03),  # Position slightly below the count
                             xycoords='data',
                             ha='center', va='top', fontsize=12,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Combined Venn diagram with MBC and MD saved to: {output_path}")
