import os
import torch
from os.path import join
import shutil
import argparse
import pandas as pd
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from Bio import SeqIO
import sys
from torch.cuda.amp import autocast
import gc  # Added to force garbage collection

sys.path.append("./../utilities/")
from helper_functions import *
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


CURRENT_DIR = os.getcwd()


def main(args):
    input_path = args.input_path
    output_path = args.output_path
    prot_emb_no = 1000
    max_seq_len = 1022
    data = pd.read_pickle(input_path)
    data_df = data.drop_duplicates(subset=["Uni_SwissProt"])
    os.makedirs(output_path, exist_ok=True)
    fasta_file = join(output_path, "all_sequences.fasta")
    with open(fasta_file, "w") as seq_file:
        for _, row in data_df.iterrows():
            uni_id, seq = row['Uni_SwissProt'], row['Protein_Sequence']
            seq_file.write(f">{uni_id}\n{seq[:max_seq_len]}\n")
            # esm2_t33_650M_UR50D
            # esm2_t30_150M_UR50D
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t30_150M_UR50D")
    model.eval()
    print(model)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(128, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches,
                                              num_workers=4,persistent_workers=True)

    temp_output_dir = join(output_path, "temp")
    os.makedirs(temp_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            toks = toks[:, :max_seq_len]
            toks = toks.to(device="cuda", non_blocking=True)
            failed_InFreeSASA = []
            with open(join(output_path, "failed_InFreeSASA.txt"), 'w') as file:
                for uniprot_id in failed_InFreeSASA:
                    file.write(f'{uniprot_id}\n')
            with autocast():
                out = model(toks, repr_layers=[30], return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            for i, label in enumerate(labels):
                output_file = join(temp_output_dir, f"{label}.pt")
                result = {"label": label}
                result["mean_representations"] = {layer: t[i, 1: len(strs[i]) + 1].mean(0).clone() for layer, t in
                                                  representations.items()}
                torch.save(result, output_file)
            del out, representations, toks,strs,labels,batch_idx, result
        torch.cuda.empty_cache()
        gc.collect()
    new_dict = {}
    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
    for k, fasta in enumerate(fasta_sequences):
        if k % prot_emb_no == 0 and k > 0:
            torch.save(new_dict, join(output_path, f"Protein_embeddings_V{version}.pt"))
            new_dict = {}
            version += 1
        name = fasta.id
        rep_dict = torch.load(join(temp_output_dir, f"{name}.pt"), weights_only=True)
        new_dict[name] = rep_dict["mean_representations"][30].numpy()
    torch.save(new_dict, join(output_path, f"Protein_embeddings_V{version}.pt"))
    shutil.rmtree(temp_output_dir)
    torch.cuda.empty_cache()
    gc.collect()

    print("Finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein embeddings using a pre-trained BERT model.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data (pickle file).")
    parser.add_argument("--output-path", type=str, required=True, help="Directory where the embeddings will be saved.")
    args = parser.parse_args()
    main(args)
