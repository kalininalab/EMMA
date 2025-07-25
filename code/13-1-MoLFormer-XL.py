import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import argparse
import os
from os.path import join

sys.path.append("./../utilities/")
from helper_functions import *

def main(args):
    input_path = args.input_path
    output_path = args.output_path
    SMILES_BERT = "ibm-research/MoLFormer-XL-both-10pct"
    MAX_LENGTH = 202
    BATCH_SIZE = 3000

    # Load tokenizer and model with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(SMILES_BERT, trust_remote_code=True)
    model = AutoModel.from_pretrained(SMILES_BERT, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model max position embeddings: {model.config.max_position_embeddings}")

    # Load input data
    data = pd.read_pickle(input_path)
    unique_data = data.drop_duplicates(subset=["molecule_ID"])

    batch = 1
    total_batches = int(np.ceil(len(unique_data) / BATCH_SIZE))

    for part, smiles_batch in enumerate(np.array_split(unique_data[["molecule_ID", "SMILES"]], total_batches)):
        smiles_reprs = {}

        for _, row in smiles_batch.iterrows():
            smiles = row["SMILES"]

            tokens = tokenizer(smiles, max_length=MAX_LENGTH, padding=True, truncation=True,
                             return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**tokens)

            # Extract mean representation from the last hidden state
            mean_repr = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            smiles_reprs[row["molecule_ID"]] = mean_repr

        # Save batch embeddings
        os.makedirs(output_path, exist_ok=True)
        output_file = join(output_path, f"SMILES_repr_{part}.pkl")
        with open(output_file, 'wb') as handle:
            pickle.dump(smiles_reprs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"{batch} batches out of {total_batches} processed")
        batch += 1

    print("**Finished without error**")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SMILES embeddings using MoLFormer-XL.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data (pickle file).")
    parser.add_argument("--output-path", type=str, required=True, help="Directory where the embeddings will be saved.")
    args = parser.parse_args()
    main(args)