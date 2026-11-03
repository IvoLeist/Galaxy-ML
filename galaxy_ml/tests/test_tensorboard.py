"""TensorBoard integration for Galaxy's Keras estimators."""

import copy
from pathlib import Path
import subprocess
import sys
import zipfile

import keras
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from galaxy_ml.keras_galaxy_models import KerasGClassifier, KerasGRegressor
from galaxy_ml.utils import clean_params


@pytest.mark.parametrize('estimator_type', [KerasGClassifier, KerasGRegressor])
@pytest.mark.parametrize('log_dir', [None, 'custom'])
def test_tensorboard_logs_training_and_validation(
        tmp_path, monkeypatch, estimator_type, log_dir):
    monkeypatch.chdir(tmp_path)
    model = keras.Sequential([
        keras.Input(shape=(2,)),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    options = {'callback_type': 'TensorBoard', 'histogram_freq': 1,
               'write_graph': False, 'update_freq': 'epoch'}
    if log_dir is not None:
        options['log_dir'] = log_dir
    callbacks = [{'callback_selection': options}]
    original = copy.deepcopy(callbacks)
    estimator = estimator_type(
        model.get_config(), loss='binary_crossentropy', metrics=['accuracy'],
        epochs=1, batch_size=4, verbose=0, callbacks=callbacks)
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
                 dtype='float32')
    y = np.array([0, 1, 0, 1], dtype='float32')
    estimator.fit(X, y, validation_data=(X, y))

    for split in ('train', 'validation'):
        events = EventAccumulator(str(tmp_path / (log_dir or 'logs') / split))
        events.Reload()
        assert 'epoch_loss' in events.Tags()['tensors']
        assert 'epoch_accuracy' in events.Tags()['tensors']
        assert len(events.Tensors('epoch_loss')) == 1
    assert callbacks == original


def test_clean_params_tensorboard():
    estimator = KerasGClassifier(
        {'layers': []}, callbacks=[{'callback_selection': {
            'callback_type': 'TensorBoard', 'log_dir': '/outside/job',
            'histogram_freq': 2, 'update_freq': 'batch',
        }}])
    pipeline = Pipeline([('model', estimator)])
    assert clean_params(pipeline) is pipeline
    assert estimator.callbacks[0]['callback_selection'] == {
        'callback_type': 'TensorBoard', 'log_dir': None,
        'histogram_freq': 2, 'update_freq': 'batch',
    }


def test_tensorboard_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert KerasGClassifier({})._callbacks is None
    assert not Path('logs').exists()


def test_tensorboard_archive_preserves_runs(tmp_path):
    for split in ('train', 'validation'):
        directory = tmp_path / 'logs' / split
        directory.mkdir(parents=True)
        (directory / 'events.out.tfevents.test').write_bytes(b'event data')
    subprocess.run([sys.executable, '-m', 'zipfile', '-c', 'logs.zip', 'logs'],
                   cwd=tmp_path, check=True)
    with zipfile.ZipFile(tmp_path / 'logs.zip') as archive:
        for split in ('train', 'validation'):
            assert archive.read(
                f'logs/{split}/events.out.tfevents.test') == b'event data'
