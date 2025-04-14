import logging
import json
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
from copy import deepcopy
from os import makedirs
from pathlib import Path
from shutil import copyfile

import numpy as np

from .evaluator import Evaluator
from .execution import (
    configure_logging,
    save_best,
    save_individual_into_csv,
    save_parent,
    save_pop,
)
from .individual import Individual


logger = logging.getLogger(__name__)


class FitnessComparator:
    def __init__(self, minimize=False):
        self.minimize = minimize

    def best_fit(self, fitness_array):
        if self.minimize:
            return np.nanmin(fitness_array)
        else:
            return np.nanmax(fitness_array)

    def idx_best(self, fitness_array):
        if self.minimize:
            return np.nanargmin(fitness_array)
        else:
            return np.nanargmax(fitness_array)

    def is_better(self, fitness_a, fitness_b):
        if math.isnan(fitness_a) and math.isnan(fitness_b):
            raise ValueError
        if math.isnan(fitness_a):
            return False
        if math.isnan(fitness_b):
            return True
        if self.minimize:
            return fitness_a < fitness_b
        else:
            return fitness_a > fitness_b

    def initial_fitness(self):
        if self.minimize:
            return float('inf')
        else:
            return float('-inf')


def select_fittest(population, population_fits, grammar, cnn_eval, gen,
                   run_path, default_train_time, fitness_comparator):
    # Get best individual just according to fitness
    idx_best = fitness_comparator.idx_best(population_fits)
    parent = population[idx_best]

    # however if the parent is not the elite, and the parent is trained for
    # longer, the elite is granted the same evaluation time.
    if parent.train_time > default_train_time:
        retrain_elite = False
        if idx_best != 0 and \
           population[0].train_time > default_train_time and \
           population[0].train_time < parent.train_time:
            retrain_elite = True
            elite = population[0]
            elite.train_time = parent.train_time
            elite.evaluate(
                grammar,
                cnn_eval,
                '%s/best_%d_%d.keras' % (run_path, gen, elite.id),
                '%s/best_%d_%d.keras' % (run_path, gen, elite.id),
            )
            population_fits[0] = elite.fitness

        min_train_time = min([ind.current_time for ind in population])

        # also retrain the best individual that is trained just for the
        # default time
        retrain_10min = False
        if min_train_time < parent.train_time:
            ids_10min = [ind.current_time == min_train_time
                         for ind in population]

            if sum(ids_10min) > 0:
                retrain_10min = True
                indvs_10min = np.array(population)[ids_10min]
                best_fitness_10min = fitness_comparator.best_fit([ind.fitness for ind in indvs_10min])
                idx_best_10min = fitness_comparator.idx_best(best_fitness_10min)
                logger.warning(f'Individual to retrain has index: {idx_best_10min}')
                parent_10min = indvs_10min[idx_best_10min]

                parent_10min.train_time = parent.train_time

                parent_10min.evaluate(
                    grammar,
                    cnn_eval,
                    '%s/best_%d_%d.keras' % (run_path, gen, parent_10min.id),
                    '%s/best_%d_%d.keras' % (run_path, gen, parent_10min.id),
                )

                population_fits[population.index(parent_10min)] = parent_10min.fitness

        # variable reassign to get a shorter name
        fc = fitness_comparator
        # select the fittest amont all retrains and the initial parent
        if retrain_elite:
            if retrain_10min:
                if fc.is_better(parent_10min.fitness, elite.fitness) and \
                   fc.is_better(parent_10min.fitness, parent.fitness):
                    return deepcopy(parent_10min)
                elif fc.is_better(elite.fitness, parent_10min.fitness) and \
                     fc.is_better(elite.fitness, parent.fitness):
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
            else:
                if fc.is_better(elite.fitness, parent.fitness):
                    return deepcopy(elite)
                else:
                    return deepcopy(parent)
        elif retrain_10min:
            if fc.is_better(parent_10min.fitness, parent.fitness):
                return deepcopy(parent_10min)
            else:
                return deepcopy(parent)
        else:
            return deepcopy(parent)

    return deepcopy(parent)


def mutation_dsge(layer, grammar):
    nt_keys = sorted(list(layer.keys()))
    nt_key = random.choice(nt_keys)
    nt_idx = random.randint(0, len(layer[nt_key])-1)

    sge_possibilities = []
    random_possibilities = []
    if len(grammar.grammar[nt_key]) > 1:
        sge_possibilities = list(
            set(range(len(grammar.grammar[nt_key])))
            - set([layer[nt_key][nt_idx]['ge']])
        )
        random_possibilities.append('ge')

    if layer[nt_key][nt_idx]['ga']:
        random_possibilities.extend(['ga', 'ga'])

    if random_possibilities:
        mt_type = random.choice(random_possibilities)

        if mt_type == 'ga':
            var_name = random.choice(sorted(list(layer[nt_key][nt_idx]['ga'].keys())))
            var_type, min_val, max_val, values = layer[nt_key][nt_idx]['ga'][var_name]
            value_idx = random.randint(0, len(values)-1)

            if var_type == 'int':
                new_val = random.randint(min_val, max_val)
            elif var_type == 'float':
                new_val = values[value_idx]+random.gauss(0, 0.15)
                new_val = np.clip(new_val, min_val, max_val)

            layer[nt_key][nt_idx]['ga'][var_name][-1][value_idx] = new_val

        elif mt_type == 'ge':
            layer[nt_key][nt_idx]['ge'] = random.choice(sge_possibilities)

        else:
            return NotImplementedError


def mutation(individual, grammar, add_layer, re_use_layer, remove_layer,
             dsge_layer, macro_layer, train_longer, default_train_time):
    # copy so that elite is preserved
    ind = deepcopy(individual)
    logger.debug('Mutation of parent individual:')

    # Train individual for longer - no other mutation is applied
    if random.random() <= train_longer:
        ind.train_time += default_train_time
        logger.debug(f'Mutation on {ind.id}: increase time to {ind.train_time}')
        return ind

    # in case the individual is mutated in any of the structural parameters
    # the training time is reseted
    ind.current_time = 0
    ind.num_epochs = 0
    ind.train_time = default_train_time

    if random.random() <= add_layer:
        module = random.choice(ind.modules)
        if len(module.layers) < module.max_expansions:
            if random.random() <= re_use_layer:
                new_layer = random.choice(module.layers)
            else:
                new_layer = grammar.initialise(module.start_symbol)

            insert_pos = random.randint(0, len(module.layers))
            module.layers.insert(insert_pos, new_layer)
            logger.debug(f'Mutation on {ind.id}: add layer')

    if random.random() <= remove_layer:
        module = random.choice(ind.modules)
        if len(module.layers) > module.min_expansions:
            remove_idx = random.randint(0, len(module.layers)-1)
            del module.layers[remove_idx]
            logger.debug(f'Mutation on {ind.id}: remove layer')

    if random.random() <= dsge_layer:
        module = random.choice(ind.modules)
        if len(module.layers) > 0:
            layer = random.choice(module.layers)
            mutation_dsge(layer, grammar)
            logger.debug(f'Mutation on {ind.id}: dsge mutation')

    # macro level mutation (at the moment learning only)
    if random.random() <= macro_layer:
        macro = random.choice(ind.macro)
        mutation_dsge(macro, grammar)
        logger.debug(f'Mutation on {ind.id}: macro mutation')

    return ind


def main(run, dataset, config, grammar):
    run_path = config.setup.run_path

    # set random seeds
    random.seed(config.setup.random_seeds[run])
    np.random.seed(config.setup.numpy_seeds[run])

    # create evaluator
    cnn_eval = Evaluator(dataset, config.evo.fitness_metric)

    # status variables
    last_gen = -1
    total_epochs = 0

    fitness_comparator = FitnessComparator(minimize=config.evo.minimize)
    best_fitness = fitness_comparator.initial_fitness()

    for gen in range(last_gen+1, config.evo.max_generations):
        if total_epochs >= config.evo.max_epochs:
            break
        if best_fitness < 10002:
            break

        if gen == 0:
            logger.info('[%d] Creating the initial population' % (run))
            logger.info('[%d] Performing generation: %d' % (run, gen))

            at_least_one_is_good = False
            while not at_least_one_is_good:
                # create initial population
                population = [
                    Individual(
                        config.network.structure,
                        config.network.macro_structure,
                        config.network.output,
                        _id_,
                    ).initialise(
                        grammar,
                        config.evo.mutations["reuse_layer"],
                        config.network.initial,
                    )
                    for _id_ in range(config.evo.es_lambda)
                ]
                logger.info('Population Initialized!')

                # set initial population variables and evaluate population
                population_fits = []
                for idx, ind in enumerate(population):
                    ind.train_time = config.evo.default_train_time
                    ind.path = f'{run_path}/best_{gen}_{idx}.keras'
                    population_fits.append(
                        ind.evaluate(
                            grammar,
                            cnn_eval,
                            f'{run_path}/best_{gen}_{idx}.keras',
                        )
                    )
                logger.info('Initial Population Evaluated!')

                for f in population_fits:
                    if not math.isnan(f):
                        at_least_one_is_good = True

        else:
            logger.info('[%d] Performing generation: %d' % (run, gen))

            # generate offspring (by mutation)
            offspring = [
                mutation(
                    parent,
                    grammar,
                    config.evo.mutations["add_layer"],
                    config.evo.mutations["reuse_layer"],
                    config.evo.mutations["remove_layer"],
                    config.evo.mutations["dsge_layer"],
                    config.evo.mutations["macro_layer"],
                    config.evo.mutations["train_longer"],
                    config.evo.default_train_time,
                )
                for _ in range(config.evo.es_lambda)
            ]

            # population of 5 element where at index 0 there is the parent
            population = [parent] + offspring

            # set elite variables to re-evaluation
            parent.current_time = 0
            parent.num_epochs = 0
            parent_id = parent.id

            # evaluate population
            population_fits = []
            for idx, ind in enumerate(population):
                ind.id = idx
                ind.path = f'{run_path}/best_{gen}_{idx}.keras'
                ind.parent_path = f'{run_path}/best_{gen-1}_{parent_id}.keras'
                population_fits.append(
                    ind.evaluate(
                        grammar,
                        cnn_eval,
                        f'{run_path}/best_{gen}_{idx}.keras',
                        f'{run_path}/best_{gen-1}_{parent_id}.keras',
                    )
                )

        parent = select_fittest(
            population, population_fits, grammar, cnn_eval, gen, run_path,
            config.evo.default_train_time, fitness_comparator,
	    )
        logger.debug(f'Fittest individual is {parent.id}')
        logger.debug(f'Fittest individual has phenotype: {parent.phenotype}')
        save_parent(parent, run_path, gen)
        save_individual_into_csv(config.setup.csv_path, run, gen, parent)

        # remove temporary files to free disk space
        if gen > 1:
            for x in range(len(population)):
                if os.path.isfile(Path(run_path, f'best_{gen-2}_{x}.keras')):
                    os.remove(Path(run_path, f'best_{gen-2}_{x}.keras'))

        # update best individual
        if fitness_comparator.is_better(parent.fitness, best_fitness):
            best_fitness = parent.fitness
            if os.path.isfile(Path(run_path, f'best_{gen}_{parent.id}.keras')):
                copyfile(
                    Path(run_path, f'best_{gen}_{parent.id}.keras'),
                    Path(run_path, 'best.keras'),
                )
            save_best(parent, run_path)

        logger.info(f'[{run}] Best fitness of generation {gen}: {parent.fitness}')
        logger.info(f'[{run}] Best overall fitness: {best_fitness}')

        # save population
        save_pop(population, run_path, gen)

        total_epochs += sum([ind.num_epochs for ind in population])
        logger.debug(f'Total epochs: {total_epochs}')

    # compute testing performance of the fittest network
    # dataset need to be shuffled or whatever before loading so please choose
    # seeds and other method to reach reproducibility there in the
    # load_dataset method
    # set random seeds cause the above may not be true
    # no random seed, we split ascad in a reproducible way which should be enough
    best_test_metrics = cnn_eval.testset_metrics(Path(run_path, 'best.keras'))
    json_metrics = {
        'ge': best_test_metrics['ge'].tolist(),
        'sr': best_test_metrics['sr'].tolist(),
        'tge': best_test_metrics['tge'],
    }
    with open(run_path / 'best_metrics.json', 'w') as f_json:
        f_json.write(json.dumps(json_metrics, indent=4))
        
    logger.info(f'[{run}] Best test fitness: {best_test_metrics}')
