import argparse
import os
import sys


def scage_cli():
    parser = argparse.ArgumentParser(
        prog='scage',
        description='A neuroevolution engine to evolve CNNs',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-c", "--config", help="YAML config file for the engine", required=True)
    parser.add_argument("-d", "--dataset", help="The name of the built-in dataset to load", required=True)
    parser.add_argument("-g", "--grammar", help="The grammar file", required=True)
    parser.add_argument("-o", "--output-folder", help="The path of the folder to save results to", required=True)
    parser.add_argument("-r", "--run", help="The number of the current run", default=0, type=int)
    parser.add_argument("-v", "--verbose", help="Increase verbosity of output", action='store_true')
    parser.add_argument("-b", "--byte-no", help="Number of the key byte")

    args = parser.parse_args()

    # check if files exist
    if not os.path.isfile(args.grammar):
        print('Grammar file does not exist.')
        sys.exit(-1)

    if not os.path.isfile(args.config):
        print('Configuration file does not exist.')
        sys.exit(-1)

    from .engine import main
    from .execution import (
        load_config,
        load_dataset,
    )
    # from .grammar_alt import Grammar
    from .grammar import Grammar

    # load config file
    config = load_config(args.config, args.run, args.output_folder, args.verbose)

    # load grammar
    grammar = Grammar.from_file(args.grammar)

    # load dataset
    if args.byte_no is None:
        evo_dataset = load_dataset(args.dataset, config.evo.target_byte)
    else:
        evo_dataset = load_dataset(args.dataset, int(args.byte_no))

    # execute
    main(args.run, evo_dataset, config, grammar)



def scage_tester():
    parser = argparse.ArgumentParser(
        prog='scagetester',
    )
    parser.add_argument("dataset", help="The name of the built-in dataset to load")
    parser.add_argument("model", help="The path of the model to test")
    parser.add_argument("byte", help="Number of the key byte (0 index)", type=int)

    args = parser.parse_args()

    from .execution import (
        load_dataset,
    )
    # load dataset
    evo_dataset = load_dataset(args.dataset, args.byte)

    from .evaluator import Evaluator
    # create evaluator
    evaluator = Evaluator(evo_dataset, 'tge')

    from pathlib import Path
    # execute
    best_test_metrics = evaluator.testset_metrics(Path(args.model))
    print(best_test_metrics)


def scage_trainer():
    parser = argparse.ArgumentParser(
        prog='scagetrainer',
    )
    parser.add_argument("dataset", help="The name of the built-in dataset to load")
    parser.add_argument("phenotype", help="The path of the model to test")
    parser.add_argument("time", help="The max time", type=int)
    parser.add_argument("byte", help="Number of the key byte (0 index)", type=int)
    parser.add_argument("-v", "--verbose", help="Increase verbosity of output", action='store_true')

    args = parser.parse_args()

    from .execution import (
        load_dataset,
    )
    # load dataset
    evo_dataset = load_dataset(args.dataset, args.byte)

    from .evaluator import Evaluator
    # create evaluator
    evaluator = Evaluator(evo_dataset, 'tge')

    import uuid
    mpath = uuid.uuid4()
    evaluator.evaluate(
        args.phenotype,
        f'/tmp/scage_{mpath}.keras',
        '',
        args.time,
        0,
    )
    best_test_metrics = evaluator.testset_metrics(f'/tmp/scage_{mpath}.keras')
    print(best_test_metrics)
