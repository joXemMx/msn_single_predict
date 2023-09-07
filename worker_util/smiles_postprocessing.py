import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def normalize_smiles(smiles):
    smiles = smiles.replace('?', '')
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol), mol
    return None, None


def check_atom_count(smiles, mol, rnn_score, formula, valid_unique_smiles):
    # Get molecular formula, including implicit hydrogens
    mf = rdMolDescriptors.CalcMolFormula(mol)

    # Compare to input molecular formula, add as valid if equal
    if mf == formula:
        valid_unique_smiles[smiles] = rnn_score
    

def filter_and_unique_smiles(df, formula):
    unique_entries = {}
    valid_unique_smiles = {}

    for smiles, rnn_score in zip(df['smiles'], df['score']):
        normalized_smiles, mol = normalize_smiles(smiles)
        # mol is not hashable and can not be used in set(), do dictionary using smiles instead
        if normalized_smiles is not None and normalized_smiles not in unique_entries:
            unique_entries[normalized_smiles] = [mol, rnn_score]
        # keep highest rnn_score of same (normalized!) SMILES
        elif normalized_smiles in unique_entries and rnn_score > unique_entries[normalized_smiles][1]:
            unique_entries[normalized_smiles][1] = rnn_score

    for smiles, info in unique_entries.items():
        mol = info[0]
        rnn_score = info[1]
        check_atom_count(smiles, mol, rnn_score, formula, valid_unique_smiles)
            
    return pd.DataFrame(list(valid_unique_smiles.items()), columns=['smiles', 'score'])
