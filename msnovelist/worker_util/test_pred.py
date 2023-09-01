from venv import logger
import worker
import model

import numpy as np
import importlib
from importlib import reload

import tensorflow as tf
import numpy as np
import random

from fp_management import database as db
import smiles_config as sc
import infrastructure.generator as gen
import infrastructure.decoder as dec

print("Initializing models.")

# Randomness is relevant in the (rare) case of using stochastic sampling
random_seed = sc.config['random_seed_global']
if random_seed != '':
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.experimental.set_seed(random_seed)

weights = sc.config['weights_folder'] + sc.config['weights']

k = sc.config["eval_k"]
kk = sc.config["eval_kk"]
steps = sc.config["eval_steps"]

TRAINING_SET = sc.config['training_set']
VALIDATION_SET = sc.config['validation_set']
pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']

decoder_name = sc.config["decoder_name"]


# File for CSI:FingerID validation data
# We need to load some DB to get blueprints!
data_eval_ = sc.config["db_path_template"]
# Load mapping table for the CSI:FingerID predictors
# Load dataset and process appropriately
db_eval = db.FpDatabase.load_from_config(data_eval_)
dataset_val = db_eval.get_all()

pipeline_options =  db_eval.get_pipeline_options()
pipeline_options['fingerprint_selected'] = "fingerprint"

# Load dataset and sampler, apply sampler to dataset
# (so we can also evaluate from fingerprint_sampled)
fp_dataset_val_ = gen.smiles_pipeline(dataset_val, 
                                    batch_size = 1,
                                    map_fingerprints=False,
                                    **pipeline_options)


sampler_name = sc.config['sampler_name']
round_fingerprints = True
if sampler_name != '':
    spl = importlib.import_module(sampler_name, 'fp_sampling')
    sf = spl.SamplerFactory(sc.config)
    round_fingerprints = sf.round_fingerprint_inference()

pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']
fp_dataset_val = gen.dataset_zip(fp_dataset_val_, 
                                 pipeline_encoder, pipeline_reference,
                                 **pipeline_options)
fp_dataset_iter = iter(fp_dataset_val)
blueprints = gen.dataset_blueprint(fp_dataset_val_)

# Load models
model_encode = model.EncoderModel(
                 blueprints = blueprints,
                 config = sc.config,
                 round_fingerprints = round_fingerprints)
model_decode = model.DecoderModel(
                 blueprints = blueprints,
                 config = sc.config,)
model_transcode = model.TranscoderModel(
                blueprints = blueprints,
                 config = sc.config,
                 round_fingerprints = round_fingerprints)

# Build models by calling them
y_ = model_transcode(blueprints)
enc = model_encode(next(fp_dataset_iter)[0])
_ = model_decode(enc)

model_transcode.load_weights(weights, by_name=True)
model_encode.copy_weights(model_transcode)
model_decode.copy_weights(model_transcode)

decoder = dec.get_decoder(decoder_name)(
    model_encode, model_decode, steps, 1, k, kk, config = sc.config)

print("Finished loading models, starting worker!")

import utility

fp = np.loadtxt("/home/joxem/worker-msnovelist/worker_util/laudanosine_fp.txt", delimiter=',')
print(fp)
mf = 'C21H27NO4'
legality, composition_counter = utility.mol_form_processing(mf)
print(legality)
print(composition_counter)
results_df = utility.predict(composition_counter, mf, fp, k, model_encode, decoder)
print(results_df)
results_df = utility.df_to_results(results_df)
print(results_df)