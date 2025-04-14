import contextlib
import logging
from multiprocessing import Pool
from time import time

import keras
import numpy as np

from .module import Module


logger = logging.getLogger(__name__)


class Individual:
    def __init__(self, network_structure, macro_rules, output_rule, ind_id):
        self.network_structure = network_structure
        self.output_rule = output_rule
        self.macro_rules = macro_rules
        self.modules = []
        self.output = None
        self.macro = []
        self.phenotype = None
        self.fitness = None
        self.metrics = None
        self.num_epochs = 0
        self.trainable_parameters = None
        self.time = None
        self.current_time = 0
        self.train_time = 0
        self.id = ind_id
        self.path = None
        self.parent_path = None

    def initialise(self, grammar, reuse, init_max):
        for start_symbol, min_expansions, max_expansions in self.network_structure:
            new_module = Module(
                start_symbol,
                min_expansions,
                max_expansions,
            )
            new_module.initialise(grammar, reuse, init_max[start_symbol])
            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self

    def random(self, grammar):
        for start_symbol, min_expansions, max_expansions in self.network_structure:
            new_module = Module(
                start_symbol,
                min_expansions,
                max_expansions,
            )
            new_module.initialise(grammar, 0, list(range(min_expansions, max_expansions)))
            self.modules.append(new_module)

        # Initialise output
        self.output = grammar.initialise(self.output_rule)

        # Initialise the macro structure: learning, data augmentation, etc.
        for rule in self.macro_rules:
            self.macro.append(grammar.initialise(rule))

        return self

    def decode(self, grammar):
        phenotype = ''
        for module in self.modules:
            for layer_genotype in module.layers:
                phenotype += grammar.decode(module.start_symbol, layer_genotype) + ' '

        phenotype += grammar.decode(self.output_rule, self.output) + ' '

        for rule_idx, macro_rule in enumerate(self.macro_rules):
            phenotype += grammar.decode(macro_rule, self.macro[rule_idx]) + ' '

        self.phenotype = phenotype.strip()
        return self.phenotype

    def evaluate(self, grammar, cnn_eval, weights_save_path, parent_weights_path=''):
        phenotype = self.decode(grammar)
        start = time()

        if self.current_time == 0:
            logger.debug('Current time is 0: training will start from scratch')
            # do not load previous weights
            parent_weights_path = ''

        train_time = self.train_time - self.current_time

        num_pool_workers = 1
        with contextlib.closing(Pool(num_pool_workers)) as po:
            pool_results = po.map_async(
                tf_evaluate,
                [(cnn_eval, phenotype, weights_save_path, parent_weights_path,
                  train_time, self.num_epochs)]
            )
            history = pool_results.get()[0]

        if history is not None:
            self.history = history
            try:
                self.fitness = self.history['fitness']
                self.num_epochs += len(self.history['loss'])
                self.trainable_parameters = self.history['trainable_parameters']
            except KeyError:
                logger.warning('Some Keys are unavailable')
                logger.warning(self.history)
            self.current_time += (self.train_time-self.current_time)
        else:
            logger.warning(f'Failed to train individual {self.id}')
            self.history = None
            self.fitness = float('nan')
            self.num_epochs = 0
            self.trainable_parameters = -1
            self.current_time = 0

        self.time = time() - start

        logger.info(f'Fitness of {self.id}: {self.fitness}')
        return self.fitness


def tf_evaluate(args):
    import traceback
    import tensorflow as tf

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except IndexError:
        logger.debug('Evaluation will run on CPU')

    cnn_eval, phenotype, weights_save_path, \
        parent_weights_path, train_time, num_epochs = args

    try:
        return cnn_eval.evaluate(
            phenotype,
            weights_save_path,
            parent_weights_path,
            train_time,
            num_epochs,
        )
    except tf.errors.ResourceExhaustedError:
        logger.warning('Memory Error: ResourceExhaustedError')
        keras.backend.clear_session()
        return None
    except TypeError as e:
        logger.warning(f'Memory Error: TypeError: {e}')
        logger.debug(traceback.format_exc())
        keras.backend.clear_session()
        return None
    except Exception as e:
        logger.warning(f'Other Error: TypeError: {e}')
        logger.debug(traceback.format_exc())
        keras.backend.clear_session()
        return None
