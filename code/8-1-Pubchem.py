import requests
import pandas as pd
import sys
import os
import re
import json
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append("./../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

# Configure requests session with retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[408, 429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Constants
MAX_WORKERS = 3
REQUEST_DELAY = 2
CHUNK_SIZE = 500
CHECKPOINT_FILE = join(CURRENT_DIR, "..", "data", "processed_data", "PubChem", "chunks", "checkpoint.json")
RESULTS_DIR = join(CURRENT_DIR, "..", "data", "processed_data", "PubChem", "chunks")
main_DIR = join(CURRENT_DIR, "..", "data", "processed_data", "PubChem")


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(current_chunk, processed_chunks):
    checkpoint_data = {
        "current_chunk": current_chunk,
        "processed_chunks": processed_chunks
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"current_chunk": 0, "processed_chunks": []}


def save_assay_chunks(assay_list, chunk_size=CHUNK_SIZE):
    create_directory(RESULTS_DIR)
    chunks = [assay_list[i:i + chunk_size] for i in range(0, len(assay_list), chunk_size)]
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(RESULTS_DIR, f"assay_chunk_{i}.json")
        with open(chunk_file, 'w') as f:
            json.dump(chunk, f)
    return len(chunks)


def load_assay_chunk(chunk_index):
    chunk_file = os.path.join(RESULTS_DIR, f"assay_chunk_{chunk_index}.json")
    if os.path.exists(chunk_file):
        with open(chunk_file, 'r') as f:
            return json.load(f)
    return None


def process_chunk(chunk_index, assay_chunk):
    chunk_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_assay, aid): aid for aid in assay_chunk}
        for future in as_completed(futures):
            aid = futures[future]
            try:
                compounds = future.result()
                chunk_results.extend(compounds)
                print(f"Completed assay {aid} with {len(compounds)} compounds")
                # if len(compounds) > 0:
                #     with open(join(main_DIR, "aids_with_ec_number.txt"), 'a') as f:
                #         f.write(f"{aid}\n")
                # elif len(compounds) == 0:
                #     with open(join(main_DIR, "aids_without_ec_number.txt"), 'a') as f:
                #         f.write(f"{aid}\n")
            except Exception as e:
                print(f"Error processing assay {aid}: {str(e)}")
    if chunk_results:
        df = pd.DataFrame(chunk_results)
        output_file = os.path.join(RESULTS_DIR, f"results_chunk_{chunk_index}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved results for chunk {chunk_index} to {output_file}")
    return len(chunk_results)


def get_existing_aids():
    """Read and return a set of assay IDs already processed"""
    existing_aids_with_ec = set()
    existing_aids_without_ec = set()
    file_path_1 = join(main_DIR, "aids_with_ec_number.txt")
    file_path_2 = join(main_DIR, "aids_without_ec_number.txt")

    if os.path.exists(file_path_1):
        with open(file_path_1, 'r') as f:
            existing_aids_with_ec = {line.strip() for line in f if line.strip()}

    if os.path.exists(file_path_2):
        with open(file_path_2, 'r') as f:
            existing_aids_without_ec = {line.strip() for line in f if line.strip()}

    return existing_aids_with_ec, existing_aids_without_ec


##########################################################################################################


def extract_pubchem_bioassay_id(term):
    """Fetch only assays that measure IC50, Ki, Kd, or Km values"""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pcassay",
        "term": term,
        "retmode": "json",
        "retmax": 10_000_000
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except requests.exceptions.RequestException as e:
        print(f"Error searching for assays: {str(e)}")
        return []


def get_assay_description(aid):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/JSON"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        assay_data = data.get("PC_AssaySubmit", {})
        if isinstance(assay_data, list):
            assay_data = assay_data[0]
        descr = assay_data.get("assay", {}).get("descr", {})
        description = " ".join(descr.get("description", [])) if isinstance(descr.get("description", []), list) else ""
        return description
    except Exception as e:
        print(f"Failed to fetch description for assay {aid}: {str(e)}")
        return None


def process_assay(aid):
    description = get_assay_description(aid)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/JSON"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        compounds = []

        for item in data.get("PC_AssaySubmit", {}).get("data", []):
            cid = item.get("sid")
            outcome = item.get("outcome")
            activity = "Active" if outcome == 2 else "Inactive"

            assay_type = None
            assay_value_uM = None
            assay_value_nM = None
            uniprot_id = None
            ec_number = None

            # Extract assay type and values
            for result in item.get("data", []):
                tid = result.get("tid")
                value = result.get("value", {})
                if tid == 1:  # Measurement type (Ki, Kd, IC50)
                    assay_type = value.get("sval")
                elif tid in [2, 3, 5]:  # Concentration (uM)
                    if assay_value_uM is None and "fval" in value:
                        assay_value_uM = value["fval"]
                    else:
                        if "fval" in value:
                            assay_value_nM = value["fval"]
                elif tid in [4, 6, 7, 8]:  # Target identifiers
                    potential_id = value.get("sval")
                    if potential_id and isinstance(potential_id, str):
                        # Clean and validate UniProt ID
                        clean_id = potential_id.strip().split('|')[-1]  # Handle "sp|P12345|..."
                        if re.fullmatch(r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}|A0A[A-Z0-9]{7}$', clean_id):
                            uniprot_id = clean_id
                            ec_number = is_enzyme(uniprot_id)  # Check if enzyme

            # Only keep if:
            # 1. Has a UniProt ID (protein target)
            # 2. Has an EC number (enzyme)
            if (
                    uniprot_id
                    and ec_number
            ):
                compounds.append({
                    "Assay ID": aid,
                    "Compound_SID": cid,
                    "Active/Inactive": activity,
                    "assay_type": assay_type,
                    "assay_value_uM": assay_value_uM,
                    "assay_value_nM": assay_value_nM,
                    "UniProt ID": uniprot_id,
                    "EC Number": ec_number,
                    "Assay Description": description
                })

        return compounds
    except Exception as e:
        print(f"Failed to process assay {aid}: {str(e)}")
        return []


def main():
    term_1 = """
(IC50[ResultType] OR binding[AssayType] OR "IC50"[Title] OR IC50 OR Ki OR Kd OR validated OR "dose response" OR "concentration response"
 AND
  (
    protein[TargetType] OR enzyme[TargetType]
    
      AND
  (
    IC50 OR Ki OR Kd OR Km OR EC50 OR "IC 50" OR "K i" OR "K d" OR "K m" OR
    "inhibition constant" OR "dissociation constant" OR "Michaelis constant" OR
    "binding affinity" OR "binding constant" OR "dose response"
  )
"""

    create_directory(RESULTS_DIR)
    if not any(f.startswith("assay_chunk_") for f in os.listdir(RESULTS_DIR)):

        # assay_ids = extract_pubchem_bioassay_id(term_1)
        # existing_aids_with_ec,existing_aids_without_ec = get_existing_aids()
        # assay_list = [aid for aid in assay_ids if aid not in existing_aids_with_ec]
        # assay_list = [aid for aid in assay_list if aid not in existing_aids_without_ec]
        existing_aids_with_ec, _ = get_existing_aids()
        assay_list = list(existing_aids_with_ec)
        if not assay_list:
            print("No assays found or failed to fetch assay list.")
            return

        print(f"Found {len(assay_list)} assays . Creating chunks...")
        total_chunks = save_assay_chunks(assay_list)
        print(f"Created {total_chunks} chunks of {CHUNK_SIZE} assays each.")
        checkpoint = {"current_chunk": 0, "processed_chunks": []}
    else:
        checkpoint = load_checkpoint()
        print(f"Resuming from checkpoint. Current chunk: {checkpoint['current_chunk']}")
        chunk_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("assay_chunk_")]
        total_chunks = len(chunk_files)

    total_compounds = 0
    for chunk_index in range(checkpoint["current_chunk"], total_chunks):
        if chunk_index in checkpoint["processed_chunks"]:
            print(f"Skipping already processed chunk {chunk_index}")
            continue
        print(f"\nProcessing chunk {chunk_index + 1}/{total_chunks}")
        assay_chunk = load_assay_chunk(chunk_index)
        if not assay_chunk:
            print(f"Failed to load chunk {chunk_index}")
            continue
        chunk_size = process_chunk(chunk_index, assay_chunk)
        total_compounds += chunk_size
        checkpoint["current_chunk"] = chunk_index + 1
        checkpoint["processed_chunks"].append(chunk_index)
        save_checkpoint(checkpoint["current_chunk"], checkpoint["processed_chunks"])
        if chunk_index + 1 < total_chunks:
            sleep(REQUEST_DELAY)
    print(f"\nProcessing complete. Total compounds collected: {total_compounds}")

    if total_compounds >= 0:
        combined_df = pd.DataFrame()
        for chunk_index in range(total_chunks):
            chunk_file = os.path.join(RESULTS_DIR, f"results_chunk_{chunk_index}.csv")
            if os.path.exists(chunk_file):
                df = pd.read_csv(chunk_file)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        if not combined_df.empty:
            combined_df.to_csv(join(main_DIR, "8-1-pubchem_assays_combined.csv"), index=False)
            print(f"Combined results saved")


if __name__ == "__main__":
    main()
