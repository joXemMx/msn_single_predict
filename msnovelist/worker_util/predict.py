# do the following to work around the problem of having yaml files with paths

import os

# Get the absolute path of the directory containing the script
parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Specify the path to your YAML file
yaml_file_path = f'{parent_folder}/config.DOCKER-LIGHT.yaml'

# Read the existing YAML content
with open(yaml_file_path, 'r') as yaml_file:
    yaml_lines = yaml_file.readlines()

# Update the first line with the new 'base_folder' value
yaml_lines[0] = f'base_folder: {parent_folder}/\n'

# Write the modified content back to the file
with open(yaml_file_path, 'w') as yaml_file:
    yaml_file.writelines(yaml_lines)

dir = os.getcwd()
os.chdir(parent_folder)

os.environ['PYTHONPATH']=parent_folder
os.environ['COMPUTERNAME']='DOCKER-LIGHT'


import load_model
import utility

os.chdir(dir)

class MSNovelist:
    def predict(mf, fp):
        try: 
            legality, composition_counter = utility.mol_form_processing(mf)
            if legality:
                results_df = utility.predict(composition_counter, mf, fp, load_model.k, load_model.model_encode, load_model.decoder)
                results_df['smiles'] = results_df['smiles'].str.replace('?', '')
                return results_df
            else: 
                print(f"The molecular formula {mf} does not fit into the list of allowed elements {utility.ELEMENTS_RDKIT}.")
        except Exception as e:
            print(e)