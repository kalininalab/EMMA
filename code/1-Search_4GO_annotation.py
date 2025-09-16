import pandas as pd
import os
from os.path import join
import gzip
from Bio.UniProt.GOA import _gpa11iterator
from multiprocessing import Pool, Lock, Value
import warnings
import time
from datetime import datetime
import math

# Suppress warnings
warnings.filterwarnings("ignore")

# Constants
# Download goa_uniprot_all.gpa.gz from here https://www.ebi.ac.uk/GOA/news
CURRENT_DIR = os.getcwd()
RAW_DATA_PATH = join(CURRENT_DIR, "..", "data", "raw_data")
GOA_FILE = join(RAW_DATA_PATH, 'goa_uniprot_all.gpa.gz')
GAF_MAPPING_FILE = join(RAW_DATA_PATH, "gaf-eco-mapping-derived.txt")
OUTPUT_DIR = join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental")

# Load ECO to GAF mapping
ECO_to_GAF = pd.read_csv(
    GAF_MAPPING_FILE,
    sep='\t',
    header=None,
    names=["ECO", "Evidence", "Default"],
    skiprows=29
)

# Experimental Evidence
exp_evidence = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"}

# Multiprocessing lock
lock = Lock()
remaining_runs = Value('i', 0)


def log_message(run, processed, total_runs):
    """
    Prints a synchronized log message with timestamps and remaining runs.
    """
    with lock:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Run {run}: Processed {processed:,} annotations in this batch. "
              f"Remaining runs: {total_runs - run - 1}")


def search_goa_database(run, total_runs, batch_size):
    """
    Processes annotations from the GOA database in chunks until EOF.
    """
    rows = []
    processed_annotations = 0

    with gzip.open(GOA_FILE, 'rt') as fp:
        for overall_count, annotation in enumerate(_gpa11iterator(fp)):
            if overall_count // batch_size == run:
                UID = annotation['DB_Object_ID']
                GO_ID = annotation['GO_ID']
                ECO_Evidence_code = annotation["ECO_Evidence_code"]
                try:
                    evidence = ECO_to_GAF.loc[ECO_to_GAF["ECO"] == ECO_Evidence_code, "Evidence"].iloc[0]
                except IndexError:
                    evidence = ""
                if evidence in exp_evidence:
                    rows.append({
                        "Uniprot ID": UID,
                        "GO Term": GO_ID,
                        'ECO_Evidence_code': ECO_Evidence_code,
                        'evidence': "exp"
                    })
                processed_annotations += 1
                if processed_annotations % 100_000 == 0:
                    log_message(run, processed_annotations, total_runs)
    if rows:
        df_GO_UID = pd.DataFrame(rows)
        output_file = join(OUTPUT_DIR, f"experimental_df_GO_UID_part_{run}.pkl")
        df_GO_UID.to_pickle(output_file)

        with lock:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Run {run} completed. Processed {len(rows):,} annotations. Output saved to {output_file}.")
    else:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
              f"Run {run} completed. No annotations to process.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parallelize processing
    print("Starting annotation processing...")
    start_time = time.time()

    # Determine total number of annotations
    # with gzip.open(GOA_FILE, 'rt') as fp:
    #     total_annotations = sum(1 for _ in _gpa11iterator(fp))
    #     print(f"Total annotations: {total_annotations}")

    total_annotations = 1236682101
    batch_size = 10 ** 6
    max_batches = math.ceil(total_annotations / batch_size)
    with Pool(processes=os.cpu_count()) as pool:
        pool.starmap(search_goa_database, [(run, max_batches, batch_size) for run in range(max_batches)])

    end_time = time.time()
    print(f"All runs completed! Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

    df_GO_UID = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental",
                                    "experimental_df_GO_UID_part_" + str(0) + ".pkl"))

    for i in range(0, 1237):
        try:
            df_new = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental",
                                         "experimental_df_GO_UID_part_" + str(i) + ".pkl"))
            df_GO_UID = pd.concat([df_GO_UID, df_new], ignore_index=True)
        except FileNotFoundError:
            print("Error", i)
    df_GO_UID.to_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental_df_GO_UID.pkl"))
