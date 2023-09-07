from queue import Queue
from collections import Counter
import molmass
import infrastructure.generator as gen
import tensorflow as tf
import smiles_postprocessing

access_token = ''
job_queue = Queue(maxsize=10)
result_queue = Queue()

ELEMENTS_RDKIT = ['C','F','I','Cl','N','O','P','Br','S','H']


# Check if unique elements of molecular formula are withing the 10 allowed
def mol_form_processing(formula):
    composition_counter = Counter({e[0]: e[1] for e in molmass.Formula(formula).composition()})
    elements = list(composition_counter.keys())
    legality = set(elements).issubset(ELEMENTS_RDKIT)
    del elements
    return legality, composition_counter


def predict(composition_counter, formula, fp, filtered, k, model_encode, decoder):
    fo = [composition_counter]
    fo_ = gen.mf_pipeline(fo).astype('float32')
    nh = fo_[:,-1]
    
    data = {'fingerprint_selected': [fp], 
            'mol_form': fo_,
            'n_hydrogen': nh}
        
    data_k = {key: tf.repeat(x, k, axis=0) for key, x in data.items()}
    states_init = model_encode.predict(data_k)
    # predict k sequences for each query.
    sequences, y, scores = decoder.decode_beam(states_init)
    seq, score, _ = decoder.beam_traceback(sequences, y, scores)
    smiles = decoder.sequence_ytoc(seq)
    
    results_df = decoder.format_results(smiles, score)
    if filtered:
        results_df = smiles_postprocessing.filter_and_unique_smiles(results_df, formula)
    return results_df
