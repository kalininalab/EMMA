import pandas as pd
from chembl_webresource_client.new_client import new_client
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from os.path import join
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import ConnectionError, HTTPError

warnings.filterwarnings("ignore")
activity_client = new_client.activity
batch_size = 20
CURRENT_DIR = os.getcwd()


# Retry decorator for fault tolerance
@retry(
    retry=retry_if_exception_type((ConnectionError, HTTPError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def fetch_batch(start, end):
    activities = activity_client.filter(assay_type="B").only(
        ['molecule_chembl_id', 'target_chembl_id', 'assay_chembl_id', 'assay_type',
         'standard_type', 'assay_description', 'canonical_smiles', 'standard_value',
         'activity_comment', 'data_validity_comment', 'standard_units']
    )[start:end]
    return pd.DataFrame(activities)


# Main processing loop
start_time = time.time()
total_activities = len(activity_client.filter(assay_type="B"))
number_of_batches = (total_activities + batch_size - 1) // batch_size
remaining_batches = number_of_batches

print(f"Total activities to process: {total_activities}")
print(f"Number of batches: {number_of_batches}")

results = []
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_batch, i, i + batch_size): i // batch_size + 1
               for i in range(0, total_activities, batch_size)}

    for future in as_completed(futures):
        batch_number = futures[future]
        try:
            batch_df = future.result()
            if not batch_df.empty:
                results.append(batch_df)
            remaining_batches -= 1
            print(f"Batch {batch_number} processed with {len(batch_df)} records. Remaining: {remaining_batches}")
        except Exception as exc:
            print(f"Batch {batch_number} failed: {exc}. Remaining: {remaining_batches}")

filtered_activities_df = pd.concat(results, ignore_index=True)
filtered_activities_df['standard_value'] = pd.to_numeric(filtered_activities_df['standard_value'], errors='coerce')
filtered_activities_df.reset_index(drop=True, inplace=True)
# Save data
save_dir = join(CURRENT_DIR, "..", "data", "processed_data", "chembl")
filtered_activities_df.to_parquet(join(save_dir, "6-1-chembl_binding_activities.parquet"))
filtered_activities_df.to_pickle(join(save_dir, "6-1-chembl_binding_activities.pkl"))
print(f"Total execution time: {(time.time() - start_time) / 3600:.2f} hours")
