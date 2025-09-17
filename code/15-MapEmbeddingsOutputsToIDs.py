import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from os.path import join
sys.path.append("./../utilities/")
from helper_functions import *


def main(args):
    current_dir = os.getcwd()
    split_folder = args.path_split_folder
    embedding_smiles = args.path_embedding_smiles
    embedding_sequence = args.path_embedded_sequence
    model_type = embedding_sequence.split("/")[-1].split("_")[-1]
    split_method = split_folder.split("/")[-1].split("_")[0]
    split_scenario = split_folder.split("/")[-1].split("_")[-1]

    # Load datasets
    data = pd.read_pickle(join(current_dir, "..", "data", "processed_data", "Final_Dataset2.pkl"))
    train = pd.read_pickle(join(current_dir, "..", "data", "splits",f"{split_folder}", f"train_{split_method}_{split_scenario}.pkl"))
    test = pd.read_pickle(join(current_dir, "..", "data", "splits", f"{split_folder}", f"test_{split_method}_{split_scenario}.pkl"))
    val=None
    splits_df = [train, test]
    splits_name = ["train", "test"]
    if "3S"==split_scenario:
        val = pd.read_pickle(
            join(current_dir, "..", "data", "splits", f"{split_folder}", f"val_{split_method}_{split_scenario}.pkl"))
        splits_df.append(val)
        splits_name.append("val")

    pro_dict = map_embeddings(data, embedding_sequence, 'Protein_embeddings_V{}.pt', 'Uni_SwissProt', torch.load)
    smiles_dict = map_embeddings(data, embedding_smiles, 'SMILES_repr_{}.pkl', 'molecule_ID', pd.read_pickle)
    for df in splits_df:
        df[f"ESM2{model_type}"] = df["Uni_SwissProt"].map(pro_dict)
        df[f"MolFormer"] = df["molecule_ID"].map(smiles_dict)

    if "3S" == split_scenario:
        result, total_samples, test_ratio, val_ratio = three_split_report(train, test, val)
        print(f"Data report after adding embedding data and checking for NaN or null cells:\n{result.to_string()}")
        print(f"Total number of samples: {total_samples}")
        print(f"Ratio of test set to dataset: {test_ratio}")
        print(f"Ratio of val set to dataset: {val_ratio}")

    elif "2S" == split_scenario:
        result, total_samples, test_ratio = two_split_report(train, test)
        print(f"Data report after adding embedding data and checking for NaN or null cells:\n{result.to_string()}")
        print(f"Total number of samples: {total_samples}")
        print(f"Ratio of test set to dataset: {test_ratio}")

    for df, name in zip(splits_df, splits_name):
        df.to_pickle(join(current_dir, "..", "data", "splits", f"{name}_{split_method}_{split_scenario}.pkl"))
    print("Finished successfully")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map all embeddings to SMILES IDs and UniProt IDs")
    parser.add_argument("--path-split-folder", type=str, required=True, help="Path to the folder where the specific split is located.")
    parser.add_argument("--path-embedding-smiles", type=str, required=True, help="Path to the folder where the smiles embedding results are located.")
    parser.add_argument("--path-embedded-sequence", type=str, required=True, help="Path to the folder where the sequence embedding results are located.")
    args = parser.parse_args()
    main(args)