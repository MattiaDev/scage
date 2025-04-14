import csv
import json
import logging
import os
import pickle
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
#import tensorflow as tf
import yaml

from .config import Config


def _ind_to_json(individual):
    return {
        'id': individual.id,
        'phenotype': individual.phenotype,
        'fitness': individual.fitness,
        'history': individual.history,
        'trainable_parameters': individual.trainable_parameters,
        'num_epochs': individual.num_epochs,
        'time': individual.time,
        'train_time': individual.train_time,
        'path': str(individual.path),
        'parent_path': str(individual.parent_path),
    }


def save_pop(population, run_path, gen):
    json_dump = [_ind_to_json(ind) for ind in population]

    with open(Path(f'{run_path}/gen_{gen:02d}.json'), 'w') as f_json:
        f_json.write(json.dumps(json_dump, indent=4))


def save_best(best, run_path):
    best_json = _ind_to_json(best)

    with open(Path(f'{run_path}/best.json'), 'w') as f_json:
        f_json.write(json.dumps(best_json, indent=4).replace('NaN', '"NaN"'))


def save_parent(parent, run_path, generation):
    parent_json = _ind_to_json(parent)

    with open(Path(f'{run_path}/parent_{generation:02d}.json'), 'w') as f_json:
        f_json.write(json.dumps(parent_json, indent=4).replace('NaN', '"NaN"'))


def save_individual_into_csv(csv_path, run, generation, individual):
    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            run,
            generation,
            individual.fitness,
            individual.history['metrics']['tge'],
            individual.history['metrics']['ge'],
            individual.num_epochs,
            individual.time,
            individual.trainable_parameters,
            individual.train_time,
            individual.id,
            individual.phenotype,
            individual.path,
            individual.parent_path,
        ])


def load_config(config_file, run, output_folder, verbose):
    with open(Path(config_file), 'r') as f:
        config = yaml.safe_load(f)

    c = Config(config, run, output_folder)
    configure_logging(c.setup.log_path, verbose)

    logger = logging.getLogger(__name__)
    logger.debug('Config successfully imported')
    logger.debug(c)

    #If run it triggers a CUDA NOT INITIALIZED error which is unsolvable
    #we're better of detecting the GPU model by runnning nvidia-smi at the beginning
    #gpu = tf.config.list_physical_devices('GPU')[0]
    #details = tf.config.experimental.get_device_details(gpu)
    #logger.debug(f"GPU: {details['device_name']}")

    fields = ['Run', 'Generation', 'Fitness', 'Tge', 'GE', 'Epochs',
              'Time', 'Parameters', 'MaxTime', 'Id', 'Phenotype', 'ModelPath',
              'ParentModelPath']
    with open(c.setup.csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    return c


def load_dataset(dataset, target_byte):
    ds_name, ds_type = dataset.split('-')
    from .dataset_parameters import ascadf, ascadr, chesctf, eshard
    from .read_ascad import ReadASCAD
    from .read_chesctf import ReadCHESCTF
    from .read_eshard import ReadEshard
    if ds_name == 'f':
        params = ascadf
        kwargs = {'fixed': True, 'target_byte': target_byte}
        reader = ReadASCAD
    elif ds_name == 'r':
        params = ascadr
        kwargs = {'fixed': False, 'target_byte': target_byte}
        reader = ReadASCAD
    elif ds_name == 'ches':
        params = chesctf
        reader = ReadCHESCTF
        if 'desync' not in ds_type:
            kwargs = {'reshape_to_cnn': False}
        else:
            kwargs = {}
    elif ds_name == 'eshard':
        params = eshard
        reader = ReadEshard
        kwargs = {'target_byte': target_byte}
    else:
        raise ValueError()
    #n_profiling = params["n_profiling"]
    #n_attack_evo = params["n_attack_evo"]
    #n_attack = params["n_attack"]
    #n_validation = params["n_validation"]
    file_path, npoi = params['files'][ds_type]
    desync = 'desync' in ds_type
    logger = logging.getLogger(__name__)
    logger.debug(f'Dataset file: {file_path}')
    logger.debug(f'Is desync: {desync}')

    ascad_dataset = reader(
        params["n_profiling"],
        params["n_validation"],
        params["n_attack_evo"],
        params["n_attack"],
        number_of_samples=npoi,
        file_path=file_path,
        is_desync=desync,
        **kwargs,
    )
    logger.debug(f'Dataset {dataset} loaded with target byte {target_byte}')

    return ascad_dataset


def configure_logging(log_path, verbose):
    # set up logging to file - see previous section for more details
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=log_path,
        filemode='a',
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler(sys.stdout)
    if verbose:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)
