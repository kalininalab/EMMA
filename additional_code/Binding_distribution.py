import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import os
from pathlib import Path
from matplotlib.gridspec import GridSpec
import numpy as np


def plot_binding_distribution(final_dataset, final_dataset_BC, output_path):

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

    # For enzyme diagram - convert to strings (FIXED THE SYNTAX ERROR HERE)
    enzyme_non_substrate = set(df[df["Binding"] == 2]["Uni_SwissProt"].astype(str).unique())
    enzyme_substrate = set(df[df["Binding"] == 1]["Uni_SwissProt"].astype(str).unique())
    enzyme_inhibitor = set(df[df["Binding"] == 0]["Uni_SwissProt"].astype(str).unique())

    # For molecule diagram - convert to strings
    mol_non_substrate = set(df[df["Binding"] == 2]["molecule_ID"].astype(str).unique())
    mol_substrate = set(df[df["Binding"] == 1]["molecule_ID"].astype(str).unique())
    mol_inhibitor = set(df[df["Binding"] == 0]["molecule_ID"].astype(str).unique())

    # ==============================================

    def calculate_average_betweenness(nodes_set, df_betwns, node_type):
        """
        Calculate average betweenness for a set of nodes
        """
        # Filter by node type (case-insensitive)
        type_filtered = df_betwns[df_betwns["Type"].str.lower() == node_type.lower()]

        if type_filtered.empty:
            return 0, 0

        # Use merge for faster matching
        nodes_df = pd.DataFrame({'Node': list(nodes_set)})
        merged = pd.merge(nodes_df, type_filtered, on='Node', how='inner')

        if not merged.empty:
            betweenness_values = merged['Betweenness'].tolist()
            avg_bet = np.mean(betweenness_values)
            return avg_bet, len(betweenness_values)
        else:
            return 0, 0

    # ==============================================

    # Calculate average betweenness for all enzyme subsets
    enzyme_subsets = {
        'non_substrate_only': enzyme_non_substrate - enzyme_substrate - enzyme_inhibitor,
        'substrate_only': enzyme_substrate - enzyme_non_substrate - enzyme_inhibitor,
        'inhibitor_only': enzyme_inhibitor - enzyme_non_substrate - enzyme_substrate,
        'substrate_inhibitor': (enzyme_substrate & enzyme_inhibitor) - enzyme_non_substrate,
        'non_substrate_substrate': (enzyme_non_substrate & enzyme_substrate) - enzyme_inhibitor,
        'non_substrate_inhibitor': (enzyme_non_substrate & enzyme_inhibitor) - enzyme_substrate,
        'all_three': enzyme_non_substrate & enzyme_substrate & enzyme_inhibitor
    }

    enzyme_betweenness = {}
    for subset_name, subset_nodes in enzyme_subsets.items():
        if subset_nodes:  # Only calculate if there are nodes
            avg_bet, count = calculate_average_betweenness(subset_nodes, df_betwns, 'enzyme')
            enzyme_betweenness[subset_name] = (avg_bet, count, len(subset_nodes))

    # Calculate average betweenness for all molecule subsets
    molecule_subsets = {
        'non_substrate_only': mol_non_substrate - mol_substrate - mol_inhibitor,
        'substrate_only': mol_substrate - mol_non_substrate - mol_inhibitor,
        'inhibitor_only': mol_inhibitor - mol_non_substrate - mol_substrate,
        'substrate_inhibitor': (mol_substrate & mol_inhibitor) - mol_non_substrate,
        'non_substrate_substrate': (mol_non_substrate & mol_substrate) - mol_inhibitor,
        'non_substrate_inhibitor': (mol_non_substrate & mol_inhibitor) - mol_substrate,
        'all_three': mol_non_substrate & mol_substrate & mol_inhibitor
    }

    molecule_betweenness = {}
    for subset_name, subset_nodes in molecule_subsets.items():
        if subset_nodes:  # Only calculate if there are nodes
            avg_bet, count = calculate_average_betweenness(subset_nodes, df_betwns, 'molecule')
            molecule_betweenness[subset_name] = (avg_bet, count, len(subset_nodes))

    # # Print results for verification
    # print("Enzyme betweenness results:", enzyme_betweenness)
    # print("Molecule betweenness results:", molecule_betweenness)
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

    # Get the positions of the existing count labels and add betweenness below them
    for label in venn_enzyme.subset_labels:
        if label is not None:
            # Get the position of the count label
            pos = label.get_position()
            count_text = label.get_text()

            # Find which subset this label corresponds to
            subset_name = None
            if count_text == str(len(enzyme_non_substrate - enzyme_substrate - enzyme_inhibitor)):
                subset_name = 'non_substrate_only'
            elif count_text == str(len(enzyme_substrate - enzyme_non_substrate - enzyme_inhibitor)):
                subset_name = 'substrate_only'
            elif count_text == str(len(enzyme_inhibitor - enzyme_non_substrate - enzyme_substrate)):
                subset_name = 'inhibitor_only'
            elif count_text == str(len((enzyme_substrate & enzyme_inhibitor) - enzyme_non_substrate)):
                subset_name = 'substrate_inhibitor'
            elif count_text == str(len((enzyme_non_substrate & enzyme_substrate) - enzyme_inhibitor)):
                subset_name = 'non_substrate_substrate'
            elif count_text == str(len((enzyme_non_substrate & enzyme_inhibitor) - enzyme_substrate)):
                subset_name = 'non_substrate_inhibitor'
            elif count_text == str(len(enzyme_non_substrate & enzyme_substrate & enzyme_inhibitor)):
                subset_name = 'all_three'

            # Add betweenness annotation below the count
            if subset_name and subset_name in enzyme_betweenness and enzyme_betweenness[subset_name][1] > 0:
                avg_bet, count, total_count = enzyme_betweenness[subset_name]
                annotation_text = f"ABC: {avg_bet:.3f}"
                ax1.annotate(annotation_text,
                             xy=(pos[0], pos[1] - 0.02),  # Position slightly below the count
                             xycoords='data',
                             ha='center', va='top', fontsize=11,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # # Add total count
    # total_enzymes = len(enzyme_non_substrate | enzyme_substrate | enzyme_inhibitor)
    # ax1.text(0.02, 0.02, f'Total: {total_enzymes}', transform=ax1.transAxes, fontsize=12,
    #          bbox=dict(boxstyle="round", fc="white", alpha=0.8))

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

    # Add betweenness annotations for molecule plot - positioned below the count numbers
    for label in venn_molecule.subset_labels:
        if label is not None:
            # Get the position of the count label
            pos = label.get_position()
            count_text = label.get_text()

            # Find which subset this label corresponds to
            subset_name = None
            if count_text == str(len(mol_non_substrate - mol_substrate - mol_inhibitor)):
                subset_name = 'non_substrate_only'
            elif count_text == str(len(mol_substrate - mol_non_substrate - mol_inhibitor)):
                subset_name = 'substrate_only'
            elif count_text == str(len(mol_inhibitor - mol_non_substrate - mol_substrate)):
                subset_name = 'inhibitor_only'
            elif count_text == str(len((mol_substrate & mol_inhibitor) - mol_non_substrate)):
                subset_name = 'substrate_inhibitor'
            elif count_text == str(len((mol_non_substrate & mol_substrate) - mol_inhibitor)):
                subset_name = 'non_substrate_substrate'
            elif count_text == str(len((mol_non_substrate & mol_inhibitor) - mol_substrate)):
                subset_name = 'non_substrate_inhibitor'
            elif count_text == str(len(mol_non_substrate & mol_substrate & mol_inhibitor)):
                subset_name = 'all_three'

            # Add betweenness annotation below the count
            if subset_name and subset_name in molecule_betweenness and molecule_betweenness[subset_name][1] > 0:
                avg_bet, count, total_count = molecule_betweenness[subset_name]
                annotation_text = f"ABC: {avg_bet:.3f}"
                ax2.annotate(annotation_text,
                             xy=(pos[0], pos[1] - 0.02),  # Position slightly below the count
                             xycoords='data',
                             ha='center', va='top', fontsize=11,
                             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # Add total count
    # total_molecules = len(mol_non_substrate | mol_substrate | mol_inhibitor)
    # ax2.text(0.02, 0.02, f'Total: {total_molecules}', transform=ax2.transAxes, fontsize=12,
    #          bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()

    print(f"Combined Venn diagram with betweenness saved to: {output_path}")

    # Create a summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Subset':<25} {'Avg BC':<15} {'Nodes with Data':<15} {'Total Nodes':<15}")
    print("-" * 70)

    print("\nENZYMES:")
    for subset, (avg_bet, count, total) in enzyme_betweenness.items():
        print(f"{subset:<25} {avg_bet:<15.3f} {count:<15} {total:<15}")

    print("\nMOLECULES:")
    for subset, (avg_bet, count, total) in molecule_betweenness.items():
        print(f"{subset:<25} {avg_bet:<15.3f} {count:<15} {total:<15}")
