import argparse
import contextlib
import csv
import time

from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# DO NOT IMPORT TENSORFLOW IN THE MAIN SCRIPT, IMPORT IT LOCALLY IN THE TRAIN FUNC
#import tensorflow as tf

from .execution import load_dataset
from .sca_metrics import sca_metrics
from .feature_selection_dlsca.random_cnn import cnn_random
from .feature_selection_dlsca.random_mlp import mlp_random


def create_and_train_network(args):
    """
    Creation and training of the model must be in the same function and that
    function must be executed as a whole using Pool.map_async in order to
    properly free memory after the execution.
    This will prevent the search from crashing while training on bigger
    datasets like ascad random.
    Tensorflow library must not be invoked anywhere else in the code, otherwise
    we risk a "GPU already initialized" error.
    """
    model_name, dataset, regularization, epochs, save_path = args

    import tensorflow as tf

    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except IndexError:
        print('Evaluation will run on CPU', flush=True)

    """ Create random model """
    if model_name == "mlp":
        model, hp = mlp_random(dataset.classes, dataset.ns, regularization=regularization)
    else:
        model, hp = cnn_random(dataset.classes, dataset.ns, regularization=regularization)

    hp["epochs"] = epochs

    try:
        model.fit(
            x=dataset.x_profiling,
            y=dataset.y_profiling,
            batch_size=hp["batch_size"],
            epochs=hp["epochs"],
            shuffle=True,
            validation_data=(dataset.x_validation, dataset.y_validation),
            verbose=0,
            callbacks=[])
        experiment_datetime = datetime.now().isoformat()
        model_path = f'{save_path}/model_{experiment_datetime}.keras'
        model.save(model_path)
        """ Compute GE, SR and NT for attack set """
        metrics = sca_metrics(
            model, dataset.x_attack, 3000, dataset.labels_key_hypothesis_attack, dataset.correct_key)
        return (*metrics, model.count_params()), experiment_datetime
    except Exception as e:
        print('ERROR', flush=True)
        print(e, flush=True)
        return None, experiment_datetime
    finally:
        tf.keras.backend.clear_session()


def random_search():
    parser = argparse.ArgumentParser(
        prog='rs',
        description='Perform a random search for the given dataset using parameter from feature selection paper',
    )
    parser.add_argument("dataset", help="The name of the built-in dataset to load")
    parser.add_argument("output", help="The name of the output folder")
    parser.add_argument("n", type=int, help="The number of models to randomly generate and assess")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, target_byte=2)
    print(f'Loaded dataset: {args.dataset}')

    number_of_searches = args.n
    #number_of_searches = 500
    epochs = 100
    regularization = True
    if 'ches' in args.dataset and 'desync' not in args.dataset:
        model_name = 'mlp'
    else:
        model_name = 'cnn'
    print(f'Model type: {model_name}')

    save_path = Path(args.output)
    save_path.mkdir(exist_ok=True, parents=True)
    csv_path = Path(f'{save_path}/rs.csv')
    if not csv_path.exists():
        fields = ['Datetime', 'Tge', 'GEi', 'GEf', 'Epochs', 'Time', 'Parameters']
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    """ Start search """
    for i in range(1, number_of_searches+1):
        start_time = time.time()

        """ Train model """
        with contextlib.closing(Pool(1)) as pool:
            pool_results = pool.map_async(
                create_and_train_network, [(model_name, dataset, regularization, epochs, save_path)]
            )
            metrics, experiment_datetime = pool_results.get()[0]


        if metrics is None:
            print(f'[{i:03d}/{number_of_searches:03d}] Model {experiment_datetime}. FAIL')
            with open(csv_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    experiment_datetime, -1, -1, -1, -1, -1, -1,
                ])
            continue

        ge, sr, nt_attack, params = metrics
        print(f'[{i:03d}/{number_of_searches:03d}] Model {experiment_datetime}. GE={ge[-1]:.2f}; TGE={nt_attack}')

        total_time = time.time() - start_time
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                experiment_datetime,
                nt_attack,
                ge[0],
                ge[-1],
                epochs,
                total_time,
                params,
            ])
