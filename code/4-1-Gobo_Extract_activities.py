import pandas as pd
from os.path import join
import os
import sys
sys.path.append("./../utilities")
from utilities.helper_functions import *
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

###################################################
# Download go.obo from here
# https://geneontology.org/docs/download-ontology/
###################################################
path_go_obo = join(CURRENT_DIR, "..", "data", "raw_data", 'go.obo')
with open(path_go_obo, 'r') as file:
    Lines = file.read()
df = pd.DataFrame(columns=["GO ID", "Definition", "Name", "RHEA ID"])
starters = ["Catalysis of the reaction", "Another possible starter"]
for term in Lines.split('[Term]\n')[1:]:
    try:
        GO_ID = term.split("id: ")[1].split("\n")[0].strip()
        name = term.split("name: ")[1].split("\n")[0].strip()
        definition = term.split("def: ")[1].split('"')[1].strip()
    except IndexError:
        continue
    if any(start in definition for start in starters):
        RHEA_ID = (term.split("xref: RHEA:")[-1].split("\n")[0].strip() if "xref: RHEA:" in term else np.nan)
        MetaCyc_ID = (term.split("xref: MetaCyc:")[-1].split("\n")[0].strip() if "xref: MetaCyc:" in term else np.nan)
        Reactome_ID = (term.split("xref: Reactome:")[-1].split(" ")[0].strip() if "xref: Reactome:" in term else np.nan)
        EC_ID = (term.split("xref: EC:")[-1].split("\n")[0].strip() if "xref: EC:" in term else np.nan)
        KEGG_ID = (
            term.split("xref: KEGG_REACTION:")[-1].split("\n")[0].strip() if "xref: KEGG_REACTION:" in term else np.nan)
        substrate_part = ""
        for delimiter in [" <=> ", " -> ", " => ", " = ", "="]:
            if delimiter in definition:
                substrate_part = definition.split(delimiter)[0].split(': ')[-1]
                break
        substrates = substrate_part.replace(" + ", ";").split(";") if substrate_part else []
        df = pd.concat([df, pd.DataFrame({
            "GO ID": [GO_ID], "Definition": [definition], "Name": [name],
            "RHEA ID": [RHEA_ID], "Substrates": [substrates], "MetaCyc ID": MetaCyc_ID, "Reactome ID": Reactome_ID,
            "EC_ID": EC_ID, "KEGG_ID": KEGG_ID
        })], ignore_index=True)

df.reset_index(drop=True, inplace=True)
df['reaction ID'] = [
    row['RHEA ID'] if pd.notnull(row['RHEA ID']) else
    row['MetaCyc ID'] if pd.notnull(row['MetaCyc ID']) else
    row['Reactome ID'] if pd.notnull(row['Reactome ID']) else
    row['KEGG_ID']
    for index, row in df.iterrows()
]
print(data_report(df))

###################################################
# Download Following files from https://www.rhea-db.org/help/download
# 1) rhea2metacyc
# 2) rhea2reactome
# 3) rhea2kegg_reaction
# 4) rhea2uniprot_sprot
###################################################
rhea2metacyc = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2metacyc.tsv"), sep="\t")
rhea2reactome = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2reactome.tsv"), sep="\t")
rhea2kegg = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2kegg_reaction.tsv"), sep="\t")
rhea2metacyc = dict(zip(rhea2metacyc['ID'], rhea2metacyc['RHEA_ID']))
rhea2reactome = dict(zip(rhea2reactome['ID'], rhea2reactome['RHEA_ID']))
rhea2kegg = dict(zip(rhea2kegg['ID'], rhea2kegg['RHEA_ID']))
df['RHEA ID'].fillna(df['MetaCyc ID'].map(rhea2metacyc), inplace=True)
df['RHEA ID'].fillna(df['MetaCyc ID'].map(rhea2reactome), inplace=True)
df['RHEA ID'].fillna(df['MetaCyc ID'].map(rhea2kegg), inplace=True)
df['RHEA ID'] = df['RHEA ID'].str.extract(r'(\d+)').astype(float)
print(data_report(df))
rhea2uniprot = pd.read_csv(join(CURRENT_DIR, "..", "data", "raw_data", "rhea2uniprot_sprot.tsv"), sep="\t")
rhea2uniprot.drop(columns=['DIRECTION', 'MASTER_ID'], inplace=True)
df_obo_uniprot = pd.merge(df, rhea2uniprot, left_on=['RHEA ID'], right_on=['RHEA_ID'], how='left')
df_obo_uniprot.drop(columns=['RHEA_ID'], inplace=True)
df_obo_uniprot.rename(columns={'ID': 'Uniprot_ID'}, inplace=True)
df_obo_uniprot.dropna(subset=['Uniprot_ID'], inplace=True)

experimental_df = pd.read_pickle(join(CURRENT_DIR, "..", "data", "raw_data", "GOA_data", "experimental_df_GO_UID.pkl"))
df_obo_uniprot = pd.merge(df_obo_uniprot, experimental_df, left_on=['Uniprot_ID', "GO ID"],
                          right_on=['Uniprot ID', "GO Term"], how='inner')
print(data_report(df_obo_uniprot))
df_obo_uniprot.drop(columns=['Uniprot ID'], inplace=True)
df_obo_uniprot = df_obo_uniprot.loc[df_obo_uniprot['evidence'] == 'exp']

df_obo_uniprot['Substrate'] = df_obo_uniprot['Substrates'].apply(
    lambda x: x.split(',') if isinstance(x, str) else (x if isinstance(x, list) else []))

df_obo_uniprot = df_obo_uniprot.explode('Substrate')
print(data_report(df_obo_uniprot))
substrates_to_filter = ['H2O', 'H2O2', 'H+', '2 H2O', 'H(+)', 'CO2', '2 H(+)', '2 CO2', 'O2', 'Ca2', 'Mn2', 'Na+',
                        'Zn2+', 'K+', 'Mg2+', 'Mg+', 'Cd2+', 'Cu+', 'phosphate', 'Cu2+']
df_obo_uniprot = df_obo_uniprot[df_obo_uniprot['Substrate'].apply(
    lambda x: isinstance(x, str) and not any(sub in x for sub in substrates_to_filter)
)]
df_obo_uniprot.reset_index(drop=True, inplace=True)
df_obo_uniprot.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "GO", "4-1-gobo_enz_sub.pkl"))
print(data_report(df_obo_uniprot))
