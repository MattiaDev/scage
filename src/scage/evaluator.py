import logging
import math
import os

import keras
from keras.callbacks import ModelCheckpoint

from .sca_metrics import sca_metrics
from .timed_stopping import TimedStopping


logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, dataset, fitness_metric):
        # define the dataset on which the models will be evaluated
        self.dataset = dataset
        # define the metric used to evaluate the models
        self.fitness_metric = fitness_metric

    def _get_layers_description(self, phenotype):
        # head and tail split to discard first item which will
        # be empty as the phenotype starts with 'layer:'
        _, *layers_raw = phenotype.split('layer:')

        layers = []
        for layer_raw in layers_raw:
            layer_type, *properties = layer_raw.split()
            property_dict = {}
            for prop in properties:
                pname, pvalue = prop.split(':')
                if ',' in pvalue:
                    property_dict[pname] = list(pvalue.split(','))
                else:
                    property_dict[pname] = pvalue
            layers.append((layer_type, property_dict))
        return layers

    def _get_learning_description(self, learning):
        raw_learning = learning.split(' ')
        learning_dict = dict(i.split(':') for i in raw_learning)
        for k, v in learning_dict.items():
            if ',' in v:
                learning_dict[k] = v.split(',')
        return learning_dict

    def _assemble_model(self, layers_description: list):
        # input layer
        input_size = self.dataset.x_profiling[0].shape
        inputs = x = keras.layers.Input(shape=input_size)
        #inputs = Input(shape=(number_of_samples, 1))

        is_first_fc = True
        # Create layers -- ADD NEW LAYERS HERE
        for layer_type, layer_params in layers_description:
            if layer_type == 'batch-norm':
                batch_norm = keras.layers.BatchNormalization()
                x = batch_norm(x)
            elif layer_type == 'conv1d':    # todo initializer
                conv1d = keras.layers.Conv1D(
                    filters=int(layer_params['num-filters']),
                    kernel_size=int(layer_params['filter-shape']),
                    strides=int(layer_params['stride']),
                    padding='same',
                    activation=layer_params['act'],
                )
                x = conv1d(x)
            elif layer_type == 'pool-avg1d':
                pool_avg1d = keras.layers.AveragePooling1D(
                    pool_size=int(layer_params['kernel-size']),
                    strides=int(layer_params['stride']),
                    padding='same',
                )
                x = pool_avg1d(x)
            elif layer_type == 'pool-max1d':
                pool_max1d = keras.layers.MaxPooling1D(
                    pool_size=int(layer_params['kernel-size']),
                    strides=int(layer_params['stride']),
                    padding='same',
                )
                x = pool_max1d(x)
            # fully-connected layer
            elif layer_type == 'fc':
                if is_first_fc:
                    x = keras.layers.Flatten()(x)
                    is_first_fc = False
                fc = keras.layers.Dense(
                    int(layer_params['num-units']),
                    activation=layer_params['act'],
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(0.0005),
                )
                outputs = x = fc(x)
            elif layer_type == 'dropout':
                if is_first_fc:
                    x = keras.layers.Flatten()(x)
                    is_first_fc = False
                dropout = keras.layers.Dropout(
                    rate=float(layer_params['rate']))
                x = dropout(x)
            else:
                raise ValueError(f'Uknown layer type: {layer_type}')

        # inputs will remain assigned to the first x
        # while output will be the last fc layer assigned (and there
        # always is the last fc with the sigmoid)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model

    def assemble_optimiser(self, learning):
        learning_rate = float(learning['lr'])

        if learning['learning'] == 'adam':
            return keras.optimizers.Adam(learning_rate=learning_rate)
        elif learning['learning'] == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError('Unknown optimizer')


    def evaluate(self, phenotype, weights_save_path, parent_weights_path, max_train_time, num_epochs):
        model_phenotype, learning_phenotype = phenotype.split('learning:')
        learning_phenotype = 'learning:' + learning_phenotype.strip()
        model_phenotype = model_phenotype.strip()

        keras_layers = self._get_layers_description(model_phenotype)
        keras_learning = self._get_learning_description(learning_phenotype)

        if parent_weights_path != '' and os.path.exists(parent_weights_path):
            logger.debug('loading previous weights')
            model = keras.models.load_model(parent_weights_path)
        else:
            model = self._assemble_model(keras_layers)
            opt = self.assemble_optimiser(keras_learning)

            model.compile(
                optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'],
            )

        callbacks = []
        # early stopping
        if 'early_stop' in keras_learning:
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=int(keras_learning['early_stop']),
                restore_best_weights=True,
            )
            callbacks.append(early_stop)
        # time based stopping
        time_stop = TimedStopping(seconds=max_train_time)
        callbacks.append(time_stop)

        trainable_count = model.count_params()

        logger.debug(f'Starting model training from epoch {num_epochs}')
        history = model.fit(
            x=self.dataset.x_profiling,
            y=self.dataset.y_profiling,
            batch_size=int(keras_learning['batch_size']),
            epochs=int(keras_learning['epochs']),
            initial_epoch=num_epochs,
            validation_data=(self.dataset.x_validation, self.dataset.y_validation),
            callbacks=callbacks,
            verbose=0,
        )

        history_tmp = history.history

        # save final model to file
        model.save(weights_save_path)

        ge, sr, tge = sca_metrics(
            model,
            self.dataset.x_attack_evo,
            3000,
            self.dataset.labels_key_hypothesis_attack_evo,
            self.dataset.correct_key,
        )
        metrics = {'ge': ge[3000-1], 'sr': sr[3000-1], 'tge': tge}

        history_tmp['trainable_parameters'] = trainable_count
        history_tmp['fitness'] = (ge[3000-1] * 10000) + (tge) + (ge[tge-1] / 1000)
        history_tmp['metrics'] = metrics
        logger.debug(f"Fitness metrics: {history_tmp['metrics']}")

        keras.backend.clear_session()

        return history_tmp

    def testset_metrics(self, model_path):
        model = keras.models.load_model(model_path)
        ge, sr, tge = sca_metrics(
            model,
            self.dataset.x_attack,
            3000,
            self.dataset.labels_key_hypothesis_attack,
            self.dataset.correct_key,
        )
        metrics = {'ge': ge, 'sr': sr, 'tge': tge}
        return metrics
