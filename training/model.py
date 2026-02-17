from __future__ import annotations

from typing import Literal

import keras
from keras import layers, models, optimizers, losses, metrics
from keras.applications import DenseNet121, DenseNet169


def build_model(
    input_shape: tuple[int, int, int],
    backbone: Literal["densenet121", "densenet169"] = "densenet121",
    freeze_backbone: bool = True,
    learning_rate: float = 1e-4,
) -> keras.Model:
    """
    Build and compile a DenseNet-based binary classification model.

    Args:
        input_shape: (H, W, C)
        backbone: 'densenet121' or 'densenet169'
        freeze_backbone: whether to freeze pretrained layers
        learning_rate: Adam learning rate

    Returns:
        Compiled keras.Model
    """

    if backbone == "densenet121":
        base_model = DenseNet121(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    elif backbone == "densenet169":
        base_model = DenseNet169(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        raise ValueError("Invalid backbone. Use 'densenet121' or 'densenet169'.")

    # Freeze backbone for transfer learning
    base_model.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="pneumonia_densenet")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.AUC(name="auc"),
            metrics.Precision(name="precision"),
            metrics.Recall(name="recall"),
        ],
    )

    return model
