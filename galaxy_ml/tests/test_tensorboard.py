"""TensorBoard logging for Galaxy Keras estimators."""
import os
from pathlib import Path

import numpy as np

import pytest

import tensorflow as tf

from keras.callbacks import TensorBoard
from keras.layers import Dense, Input
from keras.models import Sequential

from galaxy_ml.keras_galaxy_models import (
    KerasGBatchClassifier, KerasGClassifier, KerasGRegressor,
)


def _config():
    model = Sequential([
        Input(shape=(3,)),
        Dense(4, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    return model.get_config()


def _xy():
    rng = np.random.RandomState(0)
    x = rng.rand(40, 3).astype('float32')
    y = (x[:, 0] > 0.5).astype('int32')
    return x, y


def _classifier(**kwargs):
    options = dict(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        epochs=2,
        batch_size=8,
        validation_split=0.25,
        verbose=0,
        seed=0,
    )
    options.update(kwargs)
    return KerasGClassifier(_config(), **options)


def _scalar_number(value):
    """Read a scalar summary stored as a simple value or a tensor."""
    if value.HasField('simple_value'):
        return float(value.simple_value)
    if not value.HasField('tensor'):
        return None
    try:
        array = np.array(tf.make_ndarray(value.tensor))
    except (TypeError, ValueError):
        return None
    if array.shape == () and array.dtype.kind == 'f':
        return float(array)
    if array.shape == (1,) and array.dtype.kind == 'f':
        return float(array[0])
    return None


def _summary_tags(log_dir):
    """Map train/validation (and any other) split names to summary tags."""
    tags = {}
    for path in Path(log_dir).rglob('events.out.tfevents.*'):
        found = tags.setdefault(path.parent.name, set())
        for event in tf.compat.v1.train.summary_iterator(str(path)):
            for value in event.summary.value:
                found.add(value.tag)
    return tags


def _event_scalars(log_dir):
    """Return ``{(split, tag): [values]}`` for scalar summaries."""
    values = {}
    for path in Path(log_dir).rglob('events.out.tfevents.*'):
        split = path.parent.name
        for event in tf.compat.v1.train.summary_iterator(str(path)):
            for value in event.summary.value:
                number = _scalar_number(value)
                if number is None:
                    continue
                values.setdefault((split, value.tag), []).append(number)
    return values


def _has_hparam(tags, name):
    full = 'hparams/' + name
    return full in tags or any(
        tag.startswith(full + '/') for tag in tags)


def test_tensorboard_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GALAXY_ML_TENSORBOARD_LOG_DIR', raising=False)
    classifier = _classifier()

    assert classifier.tensorboard is False
    assert classifier.tensorboard_log_dir == 'logs'
    assert classifier.get_params()['tensorboard'] is False
    assert classifier.get_params()['tensorboard_log_dir'] == 'logs'

    classifier.fit(*_xy())

    assert list(tmp_path.rglob('events.out.tfevents.*')) == []


def test_tensorboard_logs_fit_metrics_and_hyperparameters(tmp_path):
    log_dir = tmp_path / 'tb-runs'
    classifier = _classifier(
        tensorboard=True,
        tensorboard_log_dir=str(log_dir),
        learning_rate=0.01,
        optimizer='adam',
    )

    history = classifier.fit(*_xy())
    classifier.fit(*_xy())

    runs = [path for path in log_dir.iterdir() if path.is_dir()]
    assert len(runs) == 2

    tags = _summary_tags(log_dir)
    train = tags['train']
    validation = tags['validation']
    for key in history.history:
        if key.startswith('val_'):
            assert 'epoch_' + key[4:] in validation, key
        else:
            assert 'epoch_' + key in train, key

    assert 'epoch_learning_rate' in train
    for name in (
        'model_type', 'optimizer', 'loss', 'metrics',
        'learning_rate', 'epochs', 'batch_size', 'validation_split',
    ):
        assert _has_hparam(train, name), name

    scalars = _event_scalars(log_dir)
    assert scalars[('train', 'hparams/epochs')][0] == pytest.approx(2)
    assert scalars[('train', 'hparams/batch_size')][0] == pytest.approx(8)
    assert scalars[('train', 'hparams/validation_split')][0] == \
        pytest.approx(0.25)
    assert scalars[('train', 'hparams/learning_rate')][0] == \
        pytest.approx(0.01)


def test_tensorboard_log_dir_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GALAXY_ML_TENSORBOARD_LOG_DIR', raising=False)
    classifier = _classifier(tensorboard=True, tensorboard_log_dir=None)

    classifier.fit(*_xy())

    assert list((tmp_path / 'logs').rglob('events.out.tfevents.*'))


def test_tensorboard_log_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_dir = tmp_path / 'from-env'
    monkeypatch.setenv('GALAXY_ML_TENSORBOARD_LOG_DIR', str(env_dir))
    classifier = _classifier(tensorboard=True, tensorboard_log_dir='')

    classifier.fit(*_xy())

    assert list(env_dir.rglob('events.out.tfevents.*'))
    assert not (tmp_path / 'logs').exists()


def test_explicit_log_dir_ignores_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        'GALAXY_ML_TENSORBOARD_LOG_DIR', str(tmp_path / 'from-env'))
    chosen = tmp_path / 'chosen'
    classifier = _classifier(
        tensorboard=True, tensorboard_log_dir=str(chosen))

    classifier.fit(*_xy())

    assert list(chosen.rglob('events.out.tfevents.*'))
    assert not (tmp_path / 'from-env').exists()


def test_regressor_and_batch_classifier_accept_tensorboard_params(tmp_path):
    regressor = KerasGRegressor(
        _config(), loss='mse', metrics=['mae'], epochs=1,
        tensorboard=True, tensorboard_log_dir=str(tmp_path / 'reg'),
    )
    assert regressor.get_params()['tensorboard'] is True

    batch = KerasGBatchClassifier(
        _config(), data_batch_generator=None, epochs=1,
        tensorboard=True, tensorboard_log_dir=str(tmp_path / 'batch'),
    )
    params = batch.get_params()
    assert params['tensorboard'] is True
    assert params['tensorboard_log_dir'] == str(tmp_path / 'batch')

    callbacks = batch._callbacks
    assert len(callbacks) == 1
    assert callbacks[0].__class__.__name__ == '_GalaxyMLTensorBoard'
    assert (tmp_path / 'batch').exists()


def test_existing_tensorboard_callback_is_not_duplicated(tmp_path):
    explicit = tmp_path / 'explicit'
    other = tmp_path / 'other'
    callbacks = [{
        'callback_selection': {
            'callback_type': 'TensorBoard',
            'log_dir': str(explicit),
        },
    }]
    classifier = _classifier(
        callbacks=callbacks,
        tensorboard=True,
        tensorboard_log_dir=str(other),
    )

    built = classifier._callbacks

    tensorboard_callbacks = [
        callback for callback in built
        if isinstance(callback, TensorBoard)]
    assert len(tensorboard_callbacks) == 1
    assert os.path.abspath(tensorboard_callbacks[0].log_dir) == \
        os.path.abspath(explicit)
    assert not other.exists()


def test_tensorboard_flag_must_be_boolean():
    with pytest.raises(ValueError, match='tensorboard must be a boolean'):
        _classifier(tensorboard='yes')
