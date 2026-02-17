from __future__ import annotations

import yaml
from pathlib import Path

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from training.dataset import get_datasets
from training.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
CONFIG_PATH = PROJECT_ROOT / "training" / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = load_config(CONFIG_PATH)

    img_height = config["data"]["img_height"]
    img_width = config["data"]["img_width"]
    batch_size = config["data"]["batch_size"]

    backbone = config["model"]["backbone"]
    freeze_backbone = config["model"]["freeze_backbone"]
    learning_rate = config["model"]["learning_rate"]

    epochs = config["training"]["epochs"]
    patience = config["training"]["patience"]

    # ------------------------------------------------------------------
    # Prepare output directory
    # ------------------------------------------------------------------
    MODEL_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    train_ds, val_ds = get_datasets(
        img_height=img_height,
        img_width=img_width,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = build_model(
        input_shape=(img_height, img_width, 3),
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        learning_rate=learning_rate,
    )

    model.summary()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_model.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    model.save(MODEL_DIR / "final_model.keras")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
