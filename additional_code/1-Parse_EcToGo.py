import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import re
import os
from os.path import join

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

# Create an empty DataFrame
df = pd.DataFrame(columns=['EC_ID', "GO_term"])

# List to hold new rows
rows = []

# ec2go.txt file has been downloaded from here https://www.ebi.ac.uk/GOA/EC2GO
with open(join(CURRENT_DIR, "..", "data", "raw_data", "ec2go.txt"), 'r') as file:
    for line in file:
        match = re.search(r'GO:\d+', line)
        if match:
            EC = line.split('>')[0].split(':')[1].strip()
            go_term = match.group()
            rows.append({'EC_ID': EC, "GO_term": go_term})

# Convert list of rows to DataFrame
df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
df.to_csv(join(CURRENT_DIR, "..", "data", "raw_data", "ec2go.csv"), index=False)

print(df)
