from pathlib import Path

import numpy as np
from keras import Input, Sequential
from keras.layers import Dense

from galaxy_ml.keras_galaxy_models import KerasGClassifier

# Synthetic binary classification dataset
rng = np.random.RandomState(42)
X = rng.rand(100, 3).astype("float32")
y = (X[:, 0] > 0.5).astype("int32")

# Define the neural network
model = Sequential([
    Input(shape=(3,)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid"),
])

# Galaxy-ML estimator
classifier = KerasGClassifier(
    config=model.get_config(),
    optimizer="adam",
    learning_rate=0.01,
    loss="binary_crossentropy",
    metrics=["accuracy"],
    epochs=5,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
    seed=42,
    tensorboard=True,
    tensorboard_log_dir="tb-runs",
)

# Train twice to verify unique run directories
for run in range(2):
    print(f"\nTraining run {run + 1}")

    history = classifier.fit(X, y)

    print("Recorded metrics:", list(history.history))

# Inspect the generated files
for path in Path("tb-runs").rglob("events.out.tfevents.*"):
    print(path)