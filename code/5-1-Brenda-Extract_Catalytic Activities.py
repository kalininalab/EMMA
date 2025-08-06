import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("/../utilities")
from utilities.helper_functions import *

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

# Characters to skip lines starting with
skip_chars = ['SYNONYMS', 'SOURCE_TISSUE', 'LOCALIZATION', 'TURNOVER_NUMBER', 'PH_OPTIMUM', 'PH_RANGE',
              'SPECIFIC_ACTIVITY', 'TEMPERATURE_OPTIMUM', 'TEMPERATURE_RANGE', 'METALS_IONS', 'MOLECULAR_WEIGHT',
              'POSTTRANSLATIONAL_MODIFICATION', 'SUBUNITS', 'PI_VALUE', 'APPLICATION', 'ENGINEERING', 'CLONED',
              'CRYSTALLIZATION', 'PURIFICATION', 'RENATURED', 'GENERAL_STABILITY', 'ORGANIC_SOLVENT_STABILITY',
              'OXIDATION_STABILITY', 'PH_STABILITY', 'STORAGE_STABILITY', 'TEMPERATURE_STABILITY', 'REFERENCE',
              'KCAT_KM_VALUE', 'EXPRESSION', 'GENERAL_INFORMATION', 'RECOMMENDED_NAME']

# Read the contents of the file
with open(join(CURRENT_DIR, "..", "data", "raw_data", "brenda_2024_1.txt"), "r") as file:
    # Skip the first line
    next(file)
    file_content = file.read()

# Split the file content into sections based on '///'
ec_blocks = file_content.split('///')
blocks_data = []
for block in range(1, len(ec_blocks)):
    parts = split_on_empty_lines(ec_blocks[block])
    ID = None
    substrate_reaction = []
    substrate_NSP = []
    substrate_SP = []
    protein_ID = []
    protein_ref = []
    inhibitors = []
    km_values = []
    ic50_values = []
    ki_values = []

    for part in parts:
        if part.startswith(tuple(skip_chars)):
            continue
        elif part.startswith('ID'):
            ID = part.split()[1]
        elif part.startswith('PROTEIN'):
            pattern = r'PR\s#(\d+)#\s.*?([A-Z0-9]+)\s(UniProt|SwissProt)'
            matches = re.findall(pattern, part)
            for match in matches:
                protein_ID.append((int(match[0]), match[1]))
                protein_ref.append(int(match[0]))
        elif part.startswith('REACTION'):
            reactions = part.strip().split('RE\t')[1:]
            for reaction in reactions:
                reaction_parts = reaction.split("=")
                substrates = reaction_parts[0].split(" + ")
                for sub in substrates:
                    substrate_reaction.append(remove_whitespace_end(sub))
        elif part.startswith('NATURAL_SUBSTRATE_PRODUCT'):
            NSPs = part.strip().split('NSP\t')[1:]
            for nsp in NSPs:
                refs_reacts = re.split(r'# |#\n\t', nsp)
                NSP_refs = re.findall(r'\d+', refs_reacts[0])
                NSPs_list = refs_reacts[1].split("=")[0].strip().split(' + ')
                for NSP_ref in NSP_refs:
                    if int(NSP_ref) in protein_ref:
                        for ns in NSPs_list:
                            substrate_NSP.append((int(NSP_ref), ns))
        elif part.startswith('SUBSTRATE_PRODUCT'):
            SPs = part.strip().split('SP\t')[1:]
            for sp in SPs:
                refs_reactions = re.split(r'# |#\n\t', sp)
                SP_refs = re.findall(r'\d+', refs_reactions[0])
                SPs_list = refs_reactions[1].split("=")[0].strip().split(' + ')
                for SP_ref in SP_refs:
                    if int(SP_ref) in protein_ref:
                        for s in SPs_list:
                            substrate_SP.append((int(SP_ref), s))
        elif part.startswith('INHIBITORS'):
            INs = part.strip().split('IN\t')[1:]
            for inh in INs:
                refs_inhibitors = re.split(r'# |#\n\t', inh)
                patt1 = r'\d#\s*([\s\S]*?)\s*\(#\d'
                patt2 = r'\d#\s*([\s\S]+?)\s*<'
                match = re.findall(patt1, inh)
                if not match:
                    match = re.findall(patt2, inh)
                IN_refs = re.findall(r'\d+', refs_inhibitors[0])
                for IN_ref in IN_refs:
                    if int(IN_ref) in protein_ref:
                        try:
                            inhibitors.append((int(IN_ref), match[0]))
                        except:
                            print(f"Failed to parse inhibitor: {inh}")
        elif part.startswith('KM_VALUE'):
            km_entries = part.strip().split('KM\t')[1:]
            for km in km_entries:
                try:
                    ref_match = re.search(r'#(\d+)#', km)
                    value_match = re.search(r'#\d+#\s+([0-9\.-]+)\s*{', km)
                    compound_match = re.search(r'{(.*?)}', km)

                    if ref_match and value_match and compound_match:
                        ref = int(ref_match.group(1))
                        value = value_match.group(1)
                        compound = compound_match.group(1)

                        if ref in protein_ref:
                            km_values.append((ref, compound, value))
                except Exception as e:
                    print(f"Error parsing KM_VALUE entry: {km}\nError: {str(e)}")

        elif part.startswith('IC50_VALUE'):
            ic50_entries = part.strip().split('IC50\t')[1:]
            for ic50 in ic50_entries:
                try:
                    ref_match = re.search(r'#(\d+)#', ic50)
                    value_match = re.search(r'#\d+#\s+([0-9\.-]+)\s*{', ic50)
                    compound_match = re.search(r'{(.*?)}', ic50)

                    if ref_match and value_match and compound_match:
                        ref = int(ref_match.group(1))
                        value = value_match.group(1)
                        compound = compound_match.group(1)

                        if ref in protein_ref:
                            ic50_values.append((ref, compound, value))
                except Exception as e:
                    print(f"Error parsing IC50_VALUE entry: {ic50}\nError: {str(e)}")

        elif part.startswith('KI_VALUE'):
            ki_entries = part.strip().split('KI\t')[1:]
            for ki in ki_entries:
                try:
                    ref_match = re.search(r'#(\d+)#', ki)
                    value_match = re.search(r'#\d+#\s+([0-9\.-]+)\s*{', ki)
                    compound_match = re.search(r'{(.*?)}', ki)

                    if ref_match and value_match and compound_match:
                        ref = int(ref_match.group(1))
                        value = value_match.group(1)
                        compound = compound_match.group(1)

                        if ref in protein_ref:
                            ki_values.append((ref, compound, value))
                except Exception as e:
                    print(f"Error parsing KI_VALUE entry: {ki}\nError: {str(e)}")

    blocks_data.append({
        'EC_ID': ID,
        'Uni_SwissProt': protein_ID,
        'reactions_NSP': substrate_NSP,
        'reactions_SP': substrate_SP,
        'substrates_reaction': substrate_reaction,
        'Inhibitor': inhibitors,
        'km_values': km_values,
        'ic50_values': ic50_values,
        'ki_values': ki_values
    })

brenda = pd.DataFrame(blocks_data)
print(data_report(brenda))
columns_to_check = ['Uni_SwissProt', 'reactions_NSP', 'reactions_SP', 'Inhibitor']
brenda = brenda[brenda.apply(lambda row: all(len(row[col]) > 0 for col in columns_to_check), axis=1)]
# print(data_report(brenda))
brenda.reset_index(drop=True, inplace=True)

# metals, ions  to filter
substrates_to_filter = ['H2O', 'H+', 'CO2', 'O2', 'Ca2', 'Mn2', 'Na+', 'Zn2+', 'K+', 'Mg2+', 'Mg+', 'Cd2+', 'Cu+',
                        'phosphate', 'Cu2+']

enz_nsub = sub_enz_pair(brenda, 'EC_ID', 'Uni_SwissProt', 'reactions_NSP')
enz_sub = sub_enz_pair(brenda, 'EC_ID', 'Uni_SwissProt', 'reactions_SP')
enz_inh = inh_enz_pair(brenda, 'EC_ID', 'Uni_SwissProt', 'Inhibitor')

# Filter out small molecules based on name
enz_nsub = enz_nsub[
    enz_nsub['Substrate'].apply(lambda x: not any(sub in x for sub in substrates_to_filter))
].drop_duplicates()

enz_sub = enz_sub[
    enz_sub['Substrate'].apply(lambda x: not any(sub in x for sub in substrates_to_filter))
].drop_duplicates()

enz_inh = enz_inh[
    enz_inh['Inhibitor'].apply(lambda x: not any(sub in x for sub in substrates_to_filter))
].drop_duplicates()

# Remove unwanted char in 'Substrate' and 'Inhibitors' columns
enz_nsub = enz_nsub[~enz_nsub['Substrate'].str.match(r'^(\d|.)$')]
enz_nsub['Substrate'] = enz_nsub['Substrate'].str.replace(r'[\n\t]', '', regex=True)

enz_sub = enz_sub[~enz_sub['Substrate'].str.match(r'^(\d|.)$')]
enz_sub['Substrate'] = enz_sub['Substrate'].str.replace(r'[\n\t]', '', regex=True)

enz_inh = enz_inh[~enz_inh['Inhibitor'].str.match(r'^(\d|.)$')]
enz_inh['Inhibitor'] = enz_inh['Inhibitor'].str.replace(r'[\n\t]', '', regex=True)

enz_nsub['standard_value'] = enz_nsub['standard_value'].replace('-999', None)
enz_sub['standard_value'] = enz_sub['standard_value'].replace('-999', None)
enz_sub = enz_sub[enz_sub['standard_value'] != '-999']
enz_inh['IC50_value'] = enz_inh['IC50_value'].replace('-999', None)
enz_inh['Ki_value'] = enz_inh['Ki_value'].replace('-999', None)

# Reset index and save to pickle
brenda_enz_sub = pd.concat([enz_nsub, enz_sub])
brenda_enz_sub.drop_duplicates(inplace=True)
brenda_enz_sub.reset_index(drop=True, inplace=True)
brenda_enz_sub.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-1-brenda_enz_sub.pkl"))
print(data_report(brenda_enz_sub))
enz_inh.reset_index(drop=True, inplace=True)
enz_inh.to_pickle(join(CURRENT_DIR, "..", "data", "processed_data", "brenda", "5-1-brenda_enz_inh.pkl"))
print(data_report(enz_inh))
