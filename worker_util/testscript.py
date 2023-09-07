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


os.environ['COMPUTERNAME']='DOCKER-LIGHT'


from predict import MSNovelist
import numpy as np

fp = np.loadtxt(f'{parent_folder}/worker_util/laudanosine_fp.txt', delimiter=',')
mf = 'C21H27NO4'
result = MSNovelist.predict(mf, fp)
print(result)