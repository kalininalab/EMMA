import pandas as pd
import numpy as np
import inspect
import re
import sys
import time
from time import sleep
import collections
from collections import defaultdict, namedtuple, Counter
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from Bio import PDB
import os
from os.path import join
import torch
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit import Chem
from rdkit import DataStructs
import urllib.parse
import shutil
from libchebipy import ChebiEntity
import pickle
from datasail.sail import datasail
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, accuracy_score, f1_score, roc_auc_score, roc_curve, auc,
                             pairwise_distances)
from sklearn.cluster import KMeans
import warnings
import logging

rdBase.DisableLog('rdApp.*')
sys.path.append("./")
from utilities.thresholds import *
warnings.filterwarnings("ignore")


session = requests.Session()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)


def split_on_empty_lines(s):
    # greedily match 2 or more new-lines
    blank_line_regex = r"(?:\r?\n){2,}"
    return re.split(blank_line_regex, s.strip())


def create_empty_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def remove_whitespace_end(s):
    # Remove occurrences of \n, \t, or space, or a combination of them from the end of the string
    return re.sub(r'[\n\t\s]+$', '', s)


def sub_enz_pair(df, ec_col, protein_col, substrate_col):
    data = []
    for _, row in df.iterrows():
        ec = row[ec_col]
        for prot in row[protein_col]:
            prot_id = prot[1]
            for sub in row[substrate_col]:
                if sub[0] == prot[0]:  # Matching reference numbers
                    km_info = None
                    for km in row.get('km_values', []):
                        if km[0] == prot[0] and km[1].lower() in sub[1].lower():
                            km_info = km[2]  # Just the value
                            break
                    data.append({
                        'EC_ID': ec,
                        'Uni_SwissProt': prot_id,
                        'Substrate': sub[1],
                        'standard_value': km_info if km_info else None
                    })
    return pd.DataFrame(data)


def inh_enz_pair(df, ec_col, protein_col, inhibitor_col):
    data = []
    for _, row in df.iterrows():
        ec = row[ec_col]
        for prot in row[protein_col]:
            prot_id = prot[1]
            for inh in row[inhibitor_col]:
                if inh[0] == prot[0]:  # Matching reference numbers
                    ic50_info = None
                    ki_info = None
                    for ic50 in row.get('ic50_values', []):
                        if ic50[0] == prot[0] and ic50[1].lower() in inh[1].lower():
                            ic50_info = ic50[2]  # Just the value
                            break
                    for ki in row.get('ki_values', []):
                        if ki[0] == prot[0] and ki[1].lower() in inh[1].lower():
                            ki_info = ki[2]  # Just the value
                            break
                    data.append({
                        'EC_ID': ec,
                        'Uni_SwissProt': prot_id,
                        'Inhibitor': inh[1],
                        'IC50_value': ic50_info if ic50_info else None,
                        'Ki_value': ki_info if ki_info else None
                    })
    return pd.DataFrame(data)


def chebi2smiles(chebi_ids):
    chebi_to_smiles = {}
    for chebi_id in chebi_ids:
        try:
            entity = ChebiEntity(chebi_id)  # Provided this class works as expected
            smiles = entity.get_smiles()
            if smiles:  # Handle None or empty strings
                chebi_to_smiles[chebi_id] = smiles
                print(f"Retrieved SMILES for {chebi_id}: {smiles}")
            else:
                print(f"No SMILES found for ChEBI ID {chebi_id}")
        except Exception as e:
            print(f"Error retrieving SMILES for ChEBI ID {chebi_id}: {e}")
    return chebi_to_smiles


def kegg2smiles(kegg_ids):
    kegg_to_smiles = {}
    for kegg_id in kegg_ids:
        try:
            kegg_url = f"http://rest.kegg.jp/get/cpd:{kegg_id}"
            response = requests.get(kegg_url)
            response.raise_for_status()
            pubchem_cid = None
            for line in response.text.splitlines():
                if line.strip().startswith("PubChem:"):
                    pubchem_cid = line.strip().split("PubChem:")[1].strip().split()[0]
                    break
                elif "PubChem:" in line:
                    pubchem_cid = line.split("PubChem:")[1].strip().split()[0]
                    break
            if pubchem_cid:
                smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_cid}/property/SMILES/TXT"
                smiles_response = requests.get(smiles_url)
                smiles_response.raise_for_status()
                smiles = smiles_response.text.strip()
                kegg_to_smiles[kegg_id] = smiles
                print(f"Retrieved SMILES for {kegg_id} via PubChem CID {pubchem_cid}: {smiles}")
            else:
                print(f"No PubChem CID found for {kegg_id}")
        except Exception as e:
            print(f"Error processing {kegg_id}: {e}")
    return kegg_to_smiles


def get_pubchem_ids(molecule_names):
    results = {}
    for name in molecule_names:
        try:
            url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/TXT'
            response = requests.get(url)
            if response.status_code == 200:
                cid = response.text.strip().split('\n')[0]
                results[name] = cid
                print(f"Found PubChem CID for '{name}': {cid}")
            else:
                results[name] = None
                print(f"No PubChem CID found for '{name}'. Status code: {response.status_code}")
        except Exception as e:
            results[name] = None
            print(f"Error occurred while fetching PubChem CID for '{name}': {e}")
    return results


def pubchem2smiles(pubchem_ids, max_retries=3, retry_delay=5):
    cid_to_smiles = {}
    for cid in pubchem_ids:
        retries = 0
        while retries < max_retries:
            try:
                url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES/JSON'
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['SMILES']
                if smiles:
                    cid_to_smiles[cid] = smiles
                    print(f"Successfully retrieved SMILES for CID {cid}: {smiles}")
                else:
                    print(f"SMILES not found for CID {cid}")
                break
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error for CID {cid}: {http_err}")
                break
            except (KeyError, IndexError) as parse_err:
                print(f"Parsing error for CID {cid}: {parse_err}")
                break
            except requests.exceptions.RequestException as req_err:
                retries += 1
                print(f"Network error for CID {cid} (attempt {retries}): {req_err}")
                if retries < max_retries:
                    sleep(retry_delay)
                else:
                    print(f"Failed to retrieve SMILES for CID {cid} after {max_retries} attempts")
            except Exception as err:
                print(f"Unexpected error for CID {cid}: {err}")
                break
    return cid_to_smiles


def sids_to_cids(sids, max_retries=3, delay=1):
    sid_cid_map = {}
    num_side = len(sids)
    for sid in sids:
        retries = 0
        while retries < max_retries:
            try:
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/cids/TXT"
                response = requests.get(url)
                response.raise_for_status()
                cid = response.text.strip()

                if cid:
                    sid_cid_map[sid] = cid
                    num_side -= 1
                    print(f"SID {sid} → CID {cid}, remaining {num_side}")
                else:
                    print(f"No CID found for SID {sid}")
                break
            except requests.exceptions.RequestException as e:
                retries += 1
                print(f"Attempt {retries} failed for SID {sid}: {e}")
                if retries < max_retries:
                    sleep(delay)
                else:
                    print(f"Max retries reached for SID {sid}")
    return sid_cid_map


def read_uniprot_ids(file_path=None, df=None):
    if file_path:
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]
    elif df is not None:
        return df['Uni_SwissProt'].dropna().unique().tolist()
    else:
        raise ValueError("Either file_path or df must be provided")


def write_uniprot_ids(file_path, uniprot_ids):
    with open(file_path, 'w') as file:
        for uniprot_id in uniprot_ids:
            file.write(f"{uniprot_id}\n")


def map_embeddings(dataframe, path, file_pattern, identifier_column, load_fn):
    embeddings_dict = {identifier: None for identifier in dataframe[identifier_column].unique()}
    for i in range(20):
        file_path = join(path, file_pattern.format(i))
        if not os.path.exists(file_path):
            continue
        if load_fn == torch.load:
            rep_dict = load_fn(file_path, weights_only=False)
        else:
            rep_dict = load_fn(file_path)
        print(f"Loaded embeddings from {file_path}")
        for identifier in dataframe[identifier_column].unique():
            if identifier in rep_dict:
                if embeddings_dict[identifier] is None and 'SMILES' in file_pattern:
                    embeddings_dict[identifier] = rep_dict[identifier][0]
                elif embeddings_dict[identifier] is None and "Protein" in file_pattern:
                    embeddings_dict[identifier] = rep_dict[identifier]
                else:
                    continue
    return embeddings_dict


def data_report(df, display_limit=None):
    nan_check = df.isnull().sum()
    nan_check.name = 'NaN'
    empty_check = pd.Series(0, index=df.columns, name='Empty')
    for col in df.select_dtypes(include=['object']).columns:
        empty_check[col] = df[col].apply(lambda x: isinstance(x, str) and x == '').sum()
    empty_list = df.apply(
        lambda col: col.apply(lambda x: isinstance(x, list) and len(x) == 0 if x is not None else False)).sum()
    empty_list = pd.Series(empty_list, name='empty_list')

    def count_unique(x):
        if isinstance(x.iloc[0], (list, np.ndarray)):
            return len(x)
        else:
            return len(pd.Series(x).dropna().unique()) if x is not None else 0
    unique_count = df.apply(count_unique)
    unique_count = pd.Series(unique_count, name='Unique')
    result = pd.concat([nan_check, empty_check, empty_list, unique_count], axis=1)
    if display_limit:
        result = result.iloc[:, :display_limit]
    caller_frame = inspect.currentframe().f_back
    df_name = [var_name for var_name, var in caller_frame.f_locals.items() if var is df][0]
    print(f"Dimension for {df_name}: {str(df.shape)}")
    return result


def two_split_report(train_set, test_set):
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set], axis=1)
    result.columns = ['NaNTrain', 'NullTrain', 'NaNTest', 'NullTest']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set)), 2)
    number_data = len(train_set) + len(test_set)
    return result, number_data, test_to_data


def three_split_report(train_set, test_set, val_set):
    nan_check_train_set = train_set.isnull().sum()
    empty_check_train_set = train_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_test_set = test_set.isnull().sum()
    empty_check_test_set = test_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    nan_check_val_set = val_set.isnull().sum()
    empty_check_val_set = val_set.applymap(lambda x: isinstance(x, str) and x == '').sum()
    result = pd.concat(
        [nan_check_train_set, empty_check_train_set, nan_check_test_set, empty_check_test_set, nan_check_val_set,
         empty_check_val_set], axis=1)
    result.columns = ['nanTrain', 'NullTrain', 'NaNTest', 'NullTest', 'nanVal', 'NullVal']
    test_to_data = round(len(test_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    val_to_data = round(len(val_set) / (len(test_set) + len(train_set) + len(val_set)), 2)
    number_data = len(train_set) + len(test_set) + len(val_set)
    return result, number_data, test_to_data, val_to_data


def parse_log(file_path):
    with open(file_path, 'r') as file:
        log_lines = file.readlines()
    iteration_data = []
    unique_losses = set()
    for line in log_lines:
        match_iteration = re.search(r'Iteration (\d+)', line)
        match_loss = re.search(r'Best loss so far: (\d+\.\d+)', line)
        if match_iteration:
            current_iteration = int(match_iteration.group(1))
        elif match_loss:
            current_loss = float(match_loss.group(1))
            if current_loss not in unique_losses:
                iteration_data.append({'iteration': current_iteration, 'loss': current_loss})
                unique_losses.add(current_loss)
    return pd.DataFrame(iteration_data)


###########################################################################################################
def label_based_split(data, split_method, split_size=None, stratification=None, epsilon=None, delta=None):
    data['ids'] = ['ID' + str(index) for index in data.index]
    results = []
    for label in [0, 1, 2]:
        label_data = data[data["Binding"] == label].copy()
        label_data['split'] = np.nan
        ratio_mol_to_pro = len(label_data["molecule_ID"].unique()) / len(label_data["Uni_SwissProt"].unique())
        if ratio_mol_to_pro > 1:
            split_tech = split_method + "f"
            print(split_tech)
            _, f_splits, _ = datasail_wrapper(split_tech, label_data, split_size, stratification=stratification,
                                              epsilon=epsilon, delta=delta)
            for key in f_splits.keys():
                label_data['split'].fillna(label_data['ids'].map(f_splits[key][0]), inplace=True)
                results.append(label_data)
        elif ratio_mol_to_pro < 1:
            split_tech = split_method + "e"
            print(split_tech)
            e_splits, _, _ = datasail_wrapper(split_tech, label_data, split_size, stratification=stratification,
                                              epsilon=epsilon, delta=delta)
            for key in e_splits.keys():
                label_data['split'].fillna(label_data['ids'].map(e_splits[key][0]), inplace=True)
                results.append(label_data)
    final_data = pd.concat(results)
    data_filtered = final_data[
        (final_data['split'] == "train") | (final_data['split'] == "test") | (final_data['split'] == "val")]
    data_filtered.reset_index(drop=True, inplace=True)
    train = data_filtered[data_filtered["split"] == "train"]
    train.reset_index(drop=True, inplace=True)
    test = data_filtered[data_filtered["split"] == "test"]
    test.reset_index(drop=True, inplace=True)
    return train, test


def datasail_wrapper(split_method, DataFrame, split_size, stratification=False, epsilon=0, delta=0):
    names = ["train", "test"]
    if len(split_size) == 3:
        names.append("val")
    # Common arguments for all split methods
    common_args = {
        "techniques": [split_method],
        "splits": split_size,
        "names": names,
        "solver": "GUROBI",
        "epsilon": epsilon,
        "max_sec": 100000
    }
    if stratification:
        common_args["delta"] = delta

    if split_method in ["C1e", "I1e"]:
        kwargs = {**common_args}
        if stratification:
            kwargs["e_strat"] = dict(DataFrame[["ids", "Binding"]].values.tolist())

        e_splits, f_splits, inter_sp = datasail(
            **kwargs,
            e_type="M",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            e_sim="ecfp"
        )
    elif split_method in ["C1f", "I1f"]:
        kwargs = {**common_args}
        if stratification:
            kwargs["f_strat"] = dict(DataFrame[["ids", "Binding"]].values.tolist())
        e_splits, f_splits, inter_sp = datasail(
            **kwargs,
            f_type="P",
            f_data=dict(DataFrame[["ids", "Protein_Sequence"]].values.tolist()),
            f_sim="cdhit"  # cdhit, mmseqspp, diamond
        )
    elif split_method in ["C2", "I2"]:
        kwargs = {**common_args}
        if stratification:
            kwargs.update({
                "e_strat": dict(DataFrame[["ids", "Binding"]].values.tolist()),
                "f_strat": dict(DataFrame[["ids", "Binding"]].values.tolist())
            })
        e_splits, f_splits, inter_sp = datasail(
            **kwargs,
            inter=[(x[0], x[0]) for x in DataFrame[["ids"]].values.tolist()],
            e_type="M",
            e_sim="ecfp",
            e_data=dict(DataFrame[["ids", "SMILES"]].values.tolist()),
            f_type="P",
            f_sim="cdhit",  # cdhit, mmseqspp, diamond
            f_data=dict(DataFrame[["ids", "Protein_Sequence"]].values.tolist())
        )
    else:
        raise ValueError("Invalid split method provided. Use one of ['C2','C1e', 'C1f', 'I1e', 'I1f','I2']")
    return e_splits, f_splits, inter_sp


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def get_substrate_IDs(IDs):
    IDs = IDs.replace(",", " + ")
    if " = " in IDs:
        IDs = IDs.split(" = ")[0]
        IDs = set(IDs.replace(" + ", ";").split(";"))
    elif " => " in IDs:
        IDs = IDs.split(" => ")[0]
        IDs = set(IDs.replace(" + ", ";").split(";"))
    elif " <=> " in IDs:
        ID = IDs.split(" <=> ")
        ID1 = ID[0].replace(" + ", ";").split(";")
        ID2 = ID[1].replace(" + ", ";").split(";")
        IDs = set(ID1 + ID2)
    return ([ID.split(" ")[-1] for ID in IDs])


def get_protein_sequences_with_retry(uniprot_ids, retries=3, backoff_factor=0.5):
    base_url = 'https://www.uniprot.org/uniprot/'
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    sequences = {}
    found_count = 0
    not_found_count = 0
    for uniprot_id in uniprot_ids:
        response = session.get(f'{base_url}{uniprot_id}.fasta')
        if response.status_code == 200:
            lines = response.text.split('\n')
            sequence = ''.join(lines[1:])
            sequences[uniprot_id] = sequence
            found_count += 1
            print(f"Sequence found for UniProt ID {uniprot_id}. Total found: {found_count}")
        else:
            sequences[uniprot_id] = None
            not_found_count += 1
            print(f"No sequence found for UniProt ID {uniprot_id}. Total not found: {not_found_count}")
    return sequences


def is_inhibitory(description):
    if description is not None:
        return bool(re.search(r'inhibitory'
                              r'|inhibitor|inhibits|inhibition|inhibiting'
                              r'|antagonist|antagonism'
                              r'|repression|repressive|suppress|suppresses'
                              r'|suppression|inhbitor|inhibiton|inhibtory', description, re.IGNORECASE))
    return False


def merge_protein_emb_files(output_dir, outpath, fasta_file, prot_emb_no):
    new_dict = {}
    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file), 'fasta')
    for k, fasta in enumerate(fasta_sequences):
        if k % prot_emb_no == 0 and k > 0:
            torch.save(new_dict, join(outpath, "Protein_Embedding", f"Protein_embeddings_V{version}.pt"))
            new_dict = {}
            version += 1
        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(join(output_dir, f"{name}.pt"))
        new_dict[name] = rep_dict["mean_representations"][33].numpy()
    torch.save(new_dict, join(outpath, "Protein_Embedding", f"Protein_embeddings_V{version}.pt"))
    shutil.rmtree(output_dir)


def create_fasta_file(valid_df, filename):
    with open(filename, "w") as seq_file:
        for uni, seq in valid_df[["Uni_SwissProt", "Protein_Sequence"]].values.tolist():
            seq_file.write(f">{uni}\n{seq[:1018]}\n")


def validate_sequence(seq, uni_id, invalid_seqs_file):
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa in valid_aas for aa in seq):
        with open(invalid_seqs_file, "a") as file:
            file.write(f"{uni_id}: {seq}\n")
        print(f"Invalid sequence: {uni_id}")
        return False
    return True


def ven_ec(dataframe, path):
    # Enzyme class mapping
    enzyme_class_mapping = {
        '1': '1',
        '5': '5',
        '2': '2',
        '7': '7',
        '3': '3',
        '4': '4',
        '6': '6'

    }
    # Extract Major Class from EC_ID by splitting at '.'
    dataframe['Major_Class'] = dataframe['EC_ID'].str.split('.').str[0]
    # Map Major Class to Enzyme Class names
    dataframe['Enzyme_Class_Name'] = dataframe['Major_Class'].map(enzyme_class_mapping)
    # Calculate frequency of each Major Class and sort by class
    frequency = dataframe['Enzyme_Class_Name'].value_counts().sort_index()
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 20), gridspec_kw={'height_ratios': [3, 1]})
    # Pie chart: Distribution of enzyme classes
    ax1.pie(frequency, labels=frequency.index, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Paired(range(len(frequency))), shadow=False,
            pctdistance=0.85, textprops={'fontsize': 14})
    ax1.set_title('Distribution of Enzyme Classes', fontsize=16, pad=20)
    # Prepare data for the table
    table_data = frequency.reset_index()
    table_data.columns = ['Enzyme Class', 'Frequency']
    # Create table below the pie chart
    table = ax2.table(cellText=table_data.values, colLabels=table_data.columns,
                      loc='center', cellLoc='center',
                      colColours=["#f5f5f5", "#f5f5f5"],
                      cellColours=[["#e0e0e0"] * 2] * len(table_data))
    # Hide the axes of the table
    ax2.axis('off')
    # Format table text
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1, 2)
    # Adjust layout to reduce space between plots
    plt.subplots_adjust(hspace=0.000005)
    # Save and show the plot
    plt.savefig(path)
    plt.show()


def ven_ec_v3(dataframe1, dataframe2, split_tech, path):
    # Enzyme class mapping
    enzyme_class_mapping = {
        '1': '1',
        '5': '5',
        '2': '2',
        '7': '7',
        '3': '3',
        '4': '4',
        '6': '6'
    }
    def process_dataframe(df):
        df['Major_Class'] = df['EC_ID'].str.split('.').str[0]
        df['Enzyme_Class_Name'] = df['Major_Class'].map(enzyme_class_mapping)
        counts = df['Enzyme_Class_Name'].value_counts().sort_index()
        # Reorder classes to separate 7 and 6
        custom_order = ['1', '2', '7', '3', '5', '4', '6']
        counts = counts.reindex(custom_order).fillna(0)
        return counts
    # Process both dataframes
    frequency1 = process_dataframe(dataframe1)
    frequency2 = process_dataframe(dataframe2)
    # Custom colors to ensure visual separation
    colors = plt.cm.Paired(range(len(frequency1)))
    # Create the figure with two pie charts side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    # Plot first dataframe pie chart
    axes[0].pie(frequency1, labels=frequency1.index, autopct='%1.1f%%', startangle=90,
                colors=colors, shadow=False, pctdistance=0.85, textprops={'fontsize': 14})
    axes[0].set_title(f'Train set of {split_tech} split', fontsize=20, pad=15)
    # Plot second dataframe pie chart
    axes[1].pie(frequency2, labels=frequency2.index, autopct='%1.1f%%', startangle=90,
                colors=colors, shadow=False, pctdistance=0.85, textprops={'fontsize': 14})
    axes[1].set_title(f'Test set of {split_tech} split', fontsize=20, pad=15)
    plt.tight_layout(pad=1.0)
    plt.savefig(path)
    plt.show()


def add_binding_source(df, binding_value, source):
    """Add binding and source columns to DataFrame."""
    df['Binding'] = binding_value
    df['Source'] = source
    return df


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        canonical_smiles = Chem.MolToSmiles(
            mol,
            isomericSmiles=True,
            allHsExplicit=False,
            canonical=True
        )

        return canonical_smiles if canonical_smiles else None
    except Exception:
        return None


def canonicalize_and_deduplicate_and_len_constraint(df):
    """Canonicalize SMILES and remove duplicates."""
    data = df.copy()
    # Canonicalize SMILES
    data = data.dropna(subset=["SMILES"])
    data = data[data["SMILES"].apply(lambda s: isinstance(s, str))]
    data["SMILES"] = data["SMILES"].apply(canonicalize_smiles)
    canonical_to_id = (data.drop_duplicates(subset=['SMILES']).set_index('SMILES')['molecule_ID'])
    data['molecule_ID'] = data['SMILES'].map(canonical_to_id)
    seq_to_id = (data.drop_duplicates(subset=['Protein_Sequence']).set_index('Protein_Sequence')['Uni_SwissProt'])
    data['Uni_SwissProt'] = data['Protein_Sequence'].map(seq_to_id)
    # remove duplicates
    data.drop_duplicates(subset=['Protein_Sequence', 'SMILES'], keep='first', inplace=True)
    data.drop_duplicates(subset=['Protein_Sequence', 'molecule_ID'], keep='first', inplace=True)
    data.drop_duplicates(subset=['Uni_SwissProt', 'SMILES'], keep='first', inplace=True)
    data.drop_duplicates(subset=['Uni_SwissProt', 'molecule_ID'], keep='first', inplace=True)
    data.dropna(subset=['SMILES', "Protein_Sequence"], inplace=True)
    # length constraint
    data['SMILES_length'] = data['SMILES'].apply(len)
    data = data[data["SMILES_length"] <= 512]
    data['Seq_length'] = data['Protein_Sequence'].apply(len)
    data = data[data["Seq_length"] <= 1500]
    data.drop(['Seq_length', 'SMILES_length'], axis=1, inplace=True)
    return data


def filter_invalid_sequences(df):
    """Filter out invalid protein sequences."""
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    invalid_uni_ids = \
        df[~df['Protein_Sequence'].apply(lambda seq: isinstance(seq, str) and all(aa in valid_aas for aa in seq))][
            'Uni_SwissProt'].tolist()
    return df[~df['Uni_SwissProt'].isin(invalid_uni_ids)]


def chembl_low_affinity_pairs(df):
    data = df.copy()
    data = data.dropna(subset=["standard_value"])
    data.dropna(subset=["standard_units"], inplace=True)
    data = data[(data['standard_units'] == "nM") |
                (data['standard_units'] == "uM") |
                (data['standard_units'] == "%")]
    data = data[data["standard_value"] > 0]
    data.loc[data['standard_type'] == "IC50", 'standard_value'] *= 0.5
    data.loc[data['standard_type'] == "Ka", 'standard_value'] = 1 / data.loc[
        data['standard_type'] == "Ka", 'standard_value']
    data.loc[data['standard_type'] == 'KA', 'standard_value'] = 1 / data.loc[
        data['standard_type'] == 'KA', 'standard_value']
    # ====== Filter for COMMENTED data ======
    commented_data = data.copy()
    inactive_keywords = ["not active", "inactive", "no activity", "non-active"]
    pattern = "|".join(inactive_keywords)
    commented_data = commented_data[commented_data["activity_comment"].str.contains(pattern, case=False, na=False)]
    grouped_dict_commented = commented_data.groupby('standard_type')['standard_units'].apply(set).to_dict()
    print(f"Candidate assays for non-interacting_commented:{grouped_dict_commented}")
    inhibition_types = ["Inhibition", "Inhib", "Inihibition"]
    inh_commented = commented_data[
        (commented_data["standard_units"] == "%") &
        (commented_data["standard_type"].str.contains("|".join(inhibition_types), case=False)) &
        (commented_data["standard_value"] <= EnNi_inh_perc)]
    ic_ki_nM_commented = commented_data[
        (commented_data["standard_units"] == "nM") &
        (commented_data["standard_type"].isin(["IC50", "Ki", "Kd", "EC50"])) &
        (commented_data["standard_value"] >= EnNi_KiECKdIC_nM)]
    non_interacting_commented = pd.concat([inh_commented, ic_ki_nM_commented])
    # ====== Filter for NON-COMMENTED data ======
    notcommented_data = data.copy()
    notcommented_data = notcommented_data[notcommented_data['activity_comment'].isna() &
                                          notcommented_data['data_validity_comment'].isna()]

    grouped_dict_notcommented = notcommented_data.groupby('standard_type')['standard_units'].apply(set).to_dict()
    print(f"Candidate assays for non-interacting_not-commented:{grouped_dict_notcommented}")

    inh_notcommented = notcommented_data[
        (notcommented_data['standard_units'] == '%') &
        (notcommented_data['standard_type'].isin([
            "Inhibition", "% of inhibition", "INH", "% Inhibition of Control Values",
            "Inhibition effect", "Enzyme inhibition",
        ])) &
        (notcommented_data['standard_value'] <= EnNi_inh_perc)]
    KiECKdIC_nM_notcommented_1 = notcommented_data[
        (notcommented_data['standard_units'] == 'nM') &
        (notcommented_data['standard_type'].isin([
            "IC50",
            "EC15", "EC30", "EC50",
            "Ki",'Ki,app', 'Kic', 'Kii', 'Kis', "Ka", "KA",'Kd,app','Kdapp','Kdiss', "Kd"
        ])) &
        (notcommented_data['standard_value'] >= EnNi_KiECKdIC_nM)]
    KiECKdIC_uM_notcommented = notcommented_data[
        (notcommented_data['standard_units'] == 'uM') &
        (notcommented_data['standard_type'].isin([
             "IC50",
            "EC1.5", "EC2", "EC5", "EC16", "EC24",
            "Ki", "Kd", 'Kic', 'Kdiss', 'Kii', 'Kis', "Ki''", "Ki'", "KI'", "Ka", "KA"
        ]))]
    KiECKdIC_uM_notcommented['standard_value'] = KiECKdIC_uM_notcommented['standard_value'] * 1000
    KiECKdIC_nM_notcommented_2 = KiECKdIC_uM_notcommented.copy()
    KiECKdIC_nM_notcommented_2['standard_units'] = 'nM'
    KiECKdIC_nM_notcommented_2 = KiECKdIC_nM_notcommented_2[
        KiECKdIC_nM_notcommented_2['standard_value'] >= EnNi_KiECKdIC_nM]

    non_interacting_notcommented = pd.concat([
        inh_notcommented,
        KiECKdIC_nM_notcommented_1,
        KiECKdIC_nM_notcommented_2,

    ])
    return non_interacting_commented, non_interacting_notcommented


def chembl_enzyme_inhibitor(df):
    data = df.copy()
    data = data.dropna(subset=["standard_value"])
    data.dropna(subset=["standard_units"], inplace=True)
    data = data[(data['standard_units'] == "nM") |
                (data['standard_units'] == "uM") |
                (data['standard_units'] == "%")]
    data = data[data["standard_value"] > 0]

    data.loc[data['standard_type'] == "IC50", 'standard_value'] *= 0.50
    # ====== Filter for COMMENTED inhibitors ======
    commented_data = data.copy()
    commented_data = commented_data[
        ~commented_data['activity_comment'].str.contains('Not Active|inactive', case=False, na=False)]
    commented_data = commented_data[
        commented_data['activity_comment'].str.contains('active', case=False, na=False)]
    to_remove = ("Activation of", "Reactivation of", "Agonist activity", "Ex vivo reactivation of",
                 "Induction of", "Reversal of", "Upregulation of", "Binding affinity to")
    commented_data = commented_data[~commented_data["assay_description"].str.startswith(to_remove)]
    grouped_dict = commented_data.groupby('standard_type')['standard_units'].apply(set).to_dict()
    print(f"Candidate assays for inhibitors_commented:{grouped_dict}")
    potent_nM_commented = commented_data[
        (commented_data['standard_units'] == 'nM') &
        (commented_data['standard_type'].isin(["IC50", "Ki"])) &
        (commented_data['standard_value'] > 0) &
        (commented_data['standard_value'] <= EnInh_KiIcMic_nM)
        ]
    strong_inh_commented_1 = commented_data[
        (commented_data['standard_units'] == '%') &
        (commented_data['standard_type'] == "Inhibition") &
        (commented_data['standard_value'] >= EnInh_inh_perc)]
    strong_inh_commented_2 = commented_data[
        (commented_data['standard_units'] == 'uM') &
        (commented_data['standard_type'].isin(["Full inhibition", "Inhibition"]))]
    strong_inh_commented_2['standard_value'] = strong_inh_commented_2['standard_value'] * 1000
    strong_inh_commented_2['standard_units'] = 'nM'
    strong_inh_commented_2 = strong_inh_commented_2[
        (strong_inh_commented_2['standard_value'] > 0) &
        (strong_inh_commented_2['standard_value'] <= EnInh_KiIcMic_nM)
        ]
    inhibitors_commented = pd.concat([potent_nM_commented, strong_inh_commented_1, strong_inh_commented_2])
    # ====== Filter for NON-COMMENTED inhibitors ======
    notcommented_data = data.copy()
    notcommented_data = notcommented_data[notcommented_data['activity_comment'].isna() &
                                          notcommented_data['data_validity_comment'].isna()]
    grouped_dict = notcommented_data.groupby('standard_type')['standard_units'].apply(set).to_dict()
    print(f"Candidate assays for inhibitors_notcommented:{grouped_dict}")

    enzyme_assay_types = [
        "IC50",
        "Ki", "KI'", 'Ki', "Ki'", "Ki''", 'Kic', 'Kii', 'Kis', "Ki,app", "Ki_app"
        "MIC", "MIC90", "MIC99",
        "Inhibition", "Inihibition", "Enzyme inhibition", "INH"]
    potent_nM_notcommented = notcommented_data[
        (notcommented_data['standard_units'] == 'nM') &
        (notcommented_data['standard_type'].isin(enzyme_assay_types)) &
        (notcommented_data['standard_value'] > 0) &
        (notcommented_data['standard_value'] <= EnInh_KiIcMic_nM)
        ]
    potent_uM_notcommented = notcommented_data[
        (notcommented_data['standard_units'] == 'uM') &
        (notcommented_data['standard_type'].isin(enzyme_assay_types))]
    potent_uM_notcommented['standard_value'] = potent_uM_notcommented['standard_value'] * 1000
    potent_uM_notcommented['standard_units'] = 'nM'
    potent_nM_notcommented_2 = potent_uM_notcommented.copy()
    potent_nM_notcommented_2 = potent_nM_notcommented_2[
        (potent_nM_notcommented_2['standard_value'] > 0) &
        (potent_nM_notcommented_2['standard_value'] <= EnInh_KiIcMic_nM)
        ]

    strong_inh_notcommented = notcommented_data[
        (notcommented_data['standard_units'] == '%') &
        (notcommented_data['standard_type'].isin(["Inhibition", "Enzyme inhibition", "Inihibition", "INH",
                                                  "% of inhibition",
                                                  "% Inhibition of Control Specific Binding (Mean n=2)",
                                                  "% Inhibition of Control Values",
                                                  "% Inhibition of Control Values (Mean n=2)","Imax"])) &
        (notcommented_data['standard_value'] >= EnInh_inh_perc)]
    inhibitors_notcommented = pd.concat([potent_nM_notcommented, potent_nM_notcommented_2, strong_inh_notcommented])
    return inhibitors_commented, inhibitors_notcommented


def smi_to_fp(smi):
    """Convert SMILES to Morgan Fingerprint (ECFP4) with reproducible settings."""
    # Explicit sanitization and parsing parameters
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    if mol is None:
        return None
    # Explicit fingerprint parameters
    return AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=3,
        nBits=2048,
        invariants=[],
        useFeatures=False,
        useChirality=True
    )


def precompute_fingerprints(df_inh, df_ni):
    df_inh = df_inh.sort_values(by=['Uni_SwissProt', 'molecule_ID']).copy()
    df_ni = df_ni.sort_values(by=['Uni_SwissProt', 'molecule_ID']).copy()
    print("Computing fingerprints for inhibitor molecules...")
    df_inh['fp'] = [smi_to_fp(smi) for smi in tqdm(df_inh['SMILES'])]
    df_inh = df_inh.dropna(subset=['fp']).copy()
    print("Computing fingerprints for non-interacting molecules...")
    df_ni['fp'] = [smi_to_fp(smi) for smi in tqdm(df_ni['SMILES'])]
    df_ni = df_ni.dropna(subset=['fp']).copy()
    return df_inh, df_ni


def downsample_pubchem(group, n=50):
    return group.sample(min(len(group), n), random_state=42)


def down_sampling(
        df_enz_inh,
        df_enz_ni,
        max_inh_per_enzyme=None,  # Max clusters for inhibitors
        max_ni_per_enzyme=None,  # Max clusters for Non-interactor
        random_state=42,
):
    # Define pickle file paths
    selected_inh_path = os.path.join(CURRENT_DIR, "..", "data", "processed_data", "selected_inh.pkl")
    selected_ni_path = os.path.join(CURRENT_DIR, "..", "data", "processed_data", "selected_ni.pkl")
    # Try to load cached results from pickle files
    if os.path.exists(selected_inh_path) and os.path.exists(selected_ni_path):
        try:
            with open(selected_inh_path, 'rb') as f:
                selected_inh = pickle.load(f)
            with open(selected_ni_path, 'rb') as f:
                selected_ni = pickle.load(f)
            print("Since K-means results are non-deterministic, cached results are loaded to reproduce the results.")
            return selected_inh, selected_ni
        except Exception as e:
            print(f"Warning: Failed to load cached results ({e}), recomputing...")
    np.random.seed(random_state)
    # Deduplicate (keep first occurrence)
    df_inh = df_enz_inh.drop_duplicates(subset=['molecule_ID', 'fp'], keep='first')
    df_ni = df_enz_ni.drop_duplicates(subset=['molecule_ID', 'fp'], keep='first')
    # Group by enzyme
    enzyme_inh = defaultdict(list)
    enzyme_ni = defaultdict(list)
    for _, row in df_inh.iterrows():
        enzyme_inh[row['Uni_SwissProt']].append(row.to_dict())
    for _, row in df_ni.iterrows():
        enzyme_ni[row['Uni_SwissProt']].append(row.to_dict())
    enzymes = set(enzyme_inh.keys()).union(set(enzyme_ni.keys()))
    selected_inh = []
    selected_ni = []
    for enzyme in tqdm(enzymes, desc="Down-sampling inhibitors and non-interacting molecules."):
        # Process INHIBITORS
        inhibitors = enzyme_inh.get(enzyme, [])
        if inhibitors:
            selected_inh.extend(clustering(inhibitors, max_inh_per_enzyme, random_state))
        # Process Non-interacting
        non_interacting = enzyme_ni.get(enzyme, [])
        if non_interacting:
            selected_ni.extend(clustering(non_interacting, max_ni_per_enzyme, random_state))
    # Save results as pickle files
    os.makedirs(os.path.dirname(selected_inh_path), exist_ok=True)
    with open(selected_inh_path, 'wb') as f:
        pickle.dump(selected_inh, f)
    with open(selected_ni_path, 'wb') as f:
        pickle.dump(selected_ni, f)
    return selected_inh, selected_ni


def mixed_key(m):
    val = m['molecule_ID']
    if isinstance(val, int):
        return (0, val)  # numbers first
    elif isinstance(val, str):
        return (1, val)  # strings after
    else:
        raise TypeError(f"Unsupported type for molecule_ID: {type(val)}")


def clustering(molecules, n_clusters, random_state):
    """Deterministic clustering with KMeans and tie-breaking."""
    # Step 1: Sort molecules initially for reproducibility
    molecules = sorted(molecules, key=mixed_key)
    if len(molecules) <= n_clusters:
        return [m['molecule_ID'] for m in molecules]
    # Step 2: Stack fingerprints
    np_fps = np.array([np.array(m['fp']) for m in molecules])
    # Step 3: Deterministic KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,      # fixed number of initializations
        max_iter=300,   # default is fine, can lower for speed
    )
    cluster_labels = kmeans.fit_predict(np_fps)
    # Step 4: Pick closest molecule in each cluster with tie-break
    selected = []
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        distances = kmeans.transform(np_fps[cluster_indices])[:, cluster_id]
        min_dist = np.min(distances)
        # Find all molecules at this min distance
        tied_indices = [
            idx for idx, dist in zip(cluster_indices, distances)
            if dist == min_dist
        ]
        # Break ties using mixed_key on molecule_ID
        chosen_idx = min(tied_indices, key=lambda idx: mixed_key(molecules[idx]))
        selected.append(molecules[chosen_idx]['molecule_ID'])
    return selected


def is_enzyme(uniprot_id):
    """Returns EC number only if experimentally validated AND not a receptor/transporter."""
    if not uniprot_id:
        return None
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        # 1. Check if it's an enzyme (EC number with evidence)
        protein_desc = data.get("proteinDescription", {})
        ec_numbers = []
        def extract_valid_ec_numbers(name_block):
            for ec in name_block.get("ecNumbers", []):
                if ec.get("evidences"):  # Only validated EC numbers
                    ec_numbers.append(ec["value"])
        if "recommendedName" in protein_desc:
            extract_valid_ec_numbers(protein_desc["recommendedName"])
        for alt_name in protein_desc.get("alternativeNames", []):
            extract_valid_ec_numbers(alt_name)
        if not ec_numbers:
            return None
        return ec_numbers[0]
    except Exception as e:
        print(f"Error checking {uniprot_id}: {str(e)}")
        return None


def molecule_distribution(df_original, df_downsampled, title_prefix=""):
    # Count molecules per enzyme
    original_counts = df_original['Uni_SwissProt'].value_counts()
    downsampled_counts = df_downsampled['Uni_SwissProt'].value_counts()
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Plot 1: Distribution comparison
    sns.boxplot(
        data=pd.DataFrame({
            'Original': original_counts,
            'Downsampled': downsampled_counts
        }),
        ax=ax1
    )
    ax1.set_title(f'{title_prefix} Distribution Comparison')
    ax1.set_ylabel('Molecules per Enzyme')
    # Plot 2: Top 20 enzymes
    combined = pd.DataFrame({
        'Original': original_counts,
        'Downsampled': downsampled_counts
    }).sort_values('Original', ascending=False).head(20)
    combined.plot(kind='bar', ax=ax2)
    ax2.set_title(f'{title_prefix} Top 20 Enzymes')
    ax2.set_ylabel('Molecule Count')
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    # Save instead of showing if in PyCharm
    plt.savefig(f'{title_prefix}_distribution.png')
    plt.close()
    # Print statistics
    print(f"\n{title_prefix} Summary Statistics:")
    print(pd.DataFrame({
        'Original': original_counts.describe(),
        'Downsampled': downsampled_counts.describe()
    }).round(2))


def calculate_metrics(inter_y_true, inter_y_pred, sub_y_true, sub_y_pred):
    """Calculate metrics for both heads for a single seed"""
    metrics = {}
    # Interaction head (binary classification)
    inter_pred_class = (inter_y_pred >= 0.5).astype(int)
    metrics['inter_accuracy'] = accuracy_score(inter_y_true, inter_pred_class)
    metrics['inter_f1'] = f1_score(inter_y_true, inter_pred_class)
    metrics['inter_auroc'] = roc_auc_score(inter_y_true, inter_y_pred)
    # Subclass head (only on interacting pairs)
    interacting_mask = (inter_y_true == 1)
    if interacting_mask.sum() > 0:
        sub_pred_class = (sub_y_pred[interacting_mask] >= 0.5).astype(int)
        if len(np.unique(sub_y_true[interacting_mask])) == 2:  # Ensure binary
            metrics['sub_accuracy'] = accuracy_score(
                sub_y_true[interacting_mask], sub_pred_class)
            metrics['sub_f1'] = f1_score(
                sub_y_true[interacting_mask], sub_pred_class)
            metrics['sub_auroc'] = roc_auc_score(
                sub_y_true[interacting_mask], sub_y_pred[interacting_mask])
    return metrics


def aggregate_results(split_methods, seeds, data_dir):
    """Calculate metrics across all seeds and splits"""
    results = {}
    for split in split_methods:
        split_results = {
            'inter_accuracy': [],
            'inter_f1': [],
            'inter_auroc': [],
            'sub_accuracy': [],
            'sub_f1': [],
            'sub_auroc': []
        }
        for seed in seeds:
            try:
                # Load data from files
                inter_true = np.load(join(data_dir, f"interaction_y_test_true_{split}_RS{seed}.npy"))
                inter_pred = np.load(join(data_dir, f"interaction_y_test_pred_{split}_RS{seed}.npy"))
                sub_true = np.load(join(data_dir, f"subclass_y_test_true_{split}_RS{seed}.npy"))
                sub_pred = np.load(join(data_dir, f"subclass_y_test_pred_{split}_RS{seed}.npy"))
                # Calculate metrics
                metrics = calculate_metrics(inter_true, inter_pred, sub_true, sub_pred)
                # Store results
                for key in split_results:
                    if key in metrics:
                        split_results[key].append(metrics[key])
            except FileNotFoundError:
                print(f"Warning: Missing data for {split} seed {seed}")
                continue
        results[split] = split_results
    return results


def create_results_table(results, split_methods):
    """Create formatted tables with mean ± std"""
    tables = {}
    # Interaction head table
    inter_data = []
    for split in split_methods:
        row = {
            'Split': split,
            'AUROC': f"{np.mean(results[split]['inter_auroc']):.3f} ± {np.std(results[split]['inter_auroc']):.3f}",
            'Accuracy': f"{np.mean(results[split]['inter_accuracy']):.3f} ± {np.std(results[split]['inter_accuracy']):.3f}",
            'F1': f"{np.mean(results[split]['inter_f1']):.3f} ± {np.std(results[split]['inter_f1']):.3f}"
        }
        inter_data.append(row)
    inter_df = pd.DataFrame(inter_data)
    # Subclass head table
    sub_data = []
    for split in split_methods:
        if results[split]['sub_accuracy']:  # Check if metrics exist
            row = {
                'Split': split,
                'AUROC': f"{np.mean(results[split]['sub_auroc']):.3f} ± {np.std(results[split]['sub_auroc']):.3f}",
                'Accuracy': f"{np.mean(results[split]['sub_accuracy']):.3f} ± {np.std(results[split]['sub_accuracy']):.3f}",
                'F1': f"{np.mean(results[split]['sub_f1']):.3f} ± {np.std(results[split]['sub_f1']):.3f}"
            }
        else:
            row = {
                'Split': split,
                'AUROC': 'N/A',
                'Accuracy': 'N/A',
                'F1': 'N/A',
            }
        sub_data.append(row)
    sub_df = pd.DataFrame(sub_data)
    return inter_df, sub_df


def plot_leakage(df_results):
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 6
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['xtick.major.size'] = 3
    mpl.rcParams['ytick.major.size'] = 3
    mpl.rcParams['xtick.major.width'] = 0.5
    mpl.rcParams['ytick.major.width'] = 0.5
    mpl.rcParams['figure.dpi'] = 150
    mpl.rcParams['savefig.dpi'] = 600
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    # Create a 2x2 grid with the top plot spanning both columns
    fig = plt.figure(figsize=(7.0, 8.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.4, wspace=0.3)
    # Top plot: Grouped bar chart (spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    # Prepare data for grouped bars
    split_methods = df_results['Split method']
    categories = ['MSL', 'ESL']
    values = df_results[['train_test_smiles_leakage', 'train_test_protein_leakage']].values.T
    # Create grouped bars
    x = range(len(split_methods))
    width = 0.3
    colors = ['#8da0cb', '#fc8d62']
    bars1 = ax1.bar([i - width/2 for i in x], values[0], width, color=colors[0],
                   label='MSL', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar([i + width/2 for i in x], values[1], width, color=colors[1],
                   label='ESL', alpha=0.8, edgecolor='black', linewidth=0.5)
    # Customize top plot
    ax1.set_xlabel('Split Method')
    ax1.set_ylabel('Leakage Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(split_methods)
    ax1.set_ylim(0, 0.55)
    ax1.set_yticks(np.arange(0, 0.6, 0.1))
    ax1.tick_params(axis='both', which='major', labelsize=6)
    # Add legend
    ax1.legend(loc='upper left', fontsize=7, frameon=True,
              fancybox=False, edgecolor='black', framealpha=0.9)
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Add grid
    ax1.grid(True, linestyle=':', alpha=0.3)
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    # Add title to the top plot
    ax1.set_title('a', fontsize=8, fontweight='bold', pad=10)
    # Left bottom plot: Correlation with AUROC_Inter
    ax2 = fig.add_subplot(gs[1, 0])
    # Plot correlation with AUROC_Inter (without individual legends)
    ax2.scatter(df_results['train_test_smiles_leakage'], df_results['AUROC_Inter'],
               color='#8da0cb', s=60, alpha=0.8, edgecolor='black', linewidth=0.5, label='MSL')
    ax2.scatter(df_results['train_test_protein_leakage'], df_results['AUROC_Inter'],
               color='#fc8d62', s=60, alpha=0.8, edgecolor='black', linewidth=0.5, label='ESL')
    # Add regression lines
    z1 = np.polyfit(df_results['train_test_smiles_leakage'], df_results['AUROC_Inter'], 1)
    p1 = np.poly1d(z1)
    ax2.plot(df_results['train_test_smiles_leakage'], p1(df_results['train_test_smiles_leakage']),
            color='#8da0cb', linestyle='--', alpha=0.7, linewidth=1)
    z2 = np.polyfit(df_results['train_test_protein_leakage'], df_results['AUROC_Inter'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df_results['train_test_protein_leakage'], p2(df_results['train_test_protein_leakage']),
            color='#fc8d62', linestyle='--', alpha=0.7, linewidth=1)
    # Customize middle plot
    ax2.set_xlabel('Leakage Score')
    ax2.set_ylabel('AUROC')
    ax2.grid(True, linestyle=':', alpha=0.3)
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Add correlation coefficient annotations
    corr_msl_inter = df_results['train_test_smiles_leakage'].corr(df_results['AUROC_Inter'])
    corr_esl_inter = df_results['train_test_protein_leakage'].corr(df_results['AUROC_Inter'])
    ax2.text(0.07, 0.95, f'MSL r = {corr_msl_inter:.3f}', transform=ax2.transAxes,
             fontsize=6, color='#8da0cb', verticalalignment='top')
    ax2.text(0.07, 0.85, f'ESL r = {corr_esl_inter:.3f}', transform=ax2.transAxes,
             fontsize=6, color='#fc8d62', verticalalignment='top')
    # Add title to the left bottom plot
    ax2.set_title('b', fontsize=8, fontweight='bold', pad=10)
    # Right bottom plot: Correlation with AUROC_Sub
    ax3 = fig.add_subplot(gs[1, 1])
    # Plot correlation with AUROC_Sub (without individual legends)
    ax3.scatter(df_results['train_test_smiles_leakage'], df_results['AUROC_Sub'],
               color='#8da0cb', s=60, alpha=0.8, edgecolor='black', linewidth=0.5, label='MSL')
    ax3.scatter(df_results['train_test_protein_leakage'], df_results['AUROC_Sub'],
               color='#fc8d62', s=60, alpha=0.8, edgecolor='black', linewidth=0.5, label='ESL')
    # Add regression lines
    z1_sub = np.polyfit(df_results['train_test_smiles_leakage'], df_results['AUROC_Sub'], 1)
    p1_sub = np.poly1d(z1_sub)
    ax3.plot(df_results['train_test_smiles_leakage'], p1_sub(df_results['train_test_smiles_leakage']),
            color='#8da0cb', linestyle='--', alpha=0.7, linewidth=1)
    z2_sub = np.polyfit(df_results['train_test_protein_leakage'], df_results['AUROC_Sub'], 1)
    p2_sub = np.poly1d(z2_sub)
    ax3.plot(df_results['train_test_protein_leakage'], p2_sub(df_results['train_test_protein_leakage']),
            color='#fc8d62', linestyle='--', alpha=0.7, linewidth=1)
    # Customize bottom plot
    ax3.set_xlabel('Leakage Score')
    # ax3.set_ylabel('AUROC (Substrate)')
    ax3.grid(True, linestyle=':', alpha=0.3)
    # Remove top and right spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    # Add correlation coefficient annotations
    corr_msl_sub = df_results['train_test_smiles_leakage'].corr(df_results['AUROC_Sub'])
    corr_esl_sub = df_results['train_test_protein_leakage'].corr(df_results['AUROC_Sub'])
    ax3.text(0.07, 0.95, f'MSL r = {corr_msl_sub:.3f}', transform=ax3.transAxes,
             fontsize=6, color='#8da0cb', verticalalignment='top')
    ax3.text(0.07, 0.85, f'ESL r = {corr_esl_sub:.3f}', transform=ax3.transAxes,
             fontsize=6, color='#fc8d62', verticalalignment='top')
    # Add title to the right bottom plot
    ax3.set_title('c', fontsize=8, fontweight='bold', pad=10)
    # Adjust layout and save
    plt.tight_layout()
    OUTPUT_DIR= join(CURRENT_DIR, "..", "data", "final_dataset_analysis","similarity_leakage.pdf")
    plt.savefig(OUTPUT_DIR, dpi=600, bbox_inches='tight')
    plt.show()


def get_multi_binding(df, column, leak):
    if leak == "SLSL":
        result = df.groupby(column)['Binding'].apply(
            lambda x: len(set(x.unique())) == 1
        )
    elif leak == "MLSL":
        result = df.groupby(column)['Binding'].apply(
            lambda x: len(set(x.unique())) >= 2
        )
    return set(result[result].index)