#!/usr/bin/env python3
"""Train all available image classification backbones on the bird dataset."""

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

from ai_engine import MODEL_BUILDERS

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset" / "images"
DEFAULT_SAVE_DIR = PROJECT_ROOT / "trained_models"
IMAGE_SIZE = (224, 224)


def get_data_augmentation(image_size):
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )


def get_dataset(dataset_dir, image_size, batch_size, validation_split, seed, augment=False):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    if augment:
        augmentation = get_data_augmentation(image_size)
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(name, num_classes, input_shape, preprocess_fn, base_trainable):
    base_model = MODEL_BUILDERS[name][0](
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=input_shape,
    )
    base_model.trainable = base_trainable

    inputs = layers.Input(shape=input_shape)
    x = inputs
    if preprocess_fn is not None:
        x = layers.Lambda(lambda image: preprocess_fn(image), name=f"{name}_preprocess")(x)
    x = base_model(x, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model_name,
    train_ds,
    val_ds,
    class_names,
    save_dir,
    epochs,
    batch_size,
    base_trainable,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = save_dir / f"{model_name}.keras"
    label_map_path = save_dir / f"{model_name}_class_names.json"

    print(f"\n--- Training {model_name} ---")
    print(f"Saving model to: {model_save_path}")

    model = build_model(
        name=model_name,
        num_classes=len(class_names),
        input_shape=(*IMAGE_SIZE, 3),
        preprocess_fn=MODEL_BUILDERS[model_name][1],
        base_trainable=base_trainable,
    )

    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=str(save_dir / f"{model_name}-best.h5"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
        ),
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
    )

    # Attempt to save the full model; fall back to HDF5, then weights+architecture,
    # and finally try re-saving from the best checkpoint if available.
    saved_model_desc = None
    h5_path = save_dir / f"{model_name}.h5"
    weights_path = save_dir / f"{model_name}_weights.h5"
    arch_path = save_dir / f"{model_name}_architecture.json"

    try:
        model.save(model_save_path)
        saved_model_desc = f"final model at {model_save_path}"
    except Exception as exc_full:
        print(f"Full model save to {model_save_path} failed: {exc_full}")
        try:
            model.save(str(h5_path), save_format="h5")
            saved_model_desc = f"HDF5 model at {h5_path}"
        except Exception as exc_h5:
            print(f"HDF5 save to {h5_path} failed: {exc_h5}")
            # Save weights and architecture separately as a last-resort fallback
            try:
                model.save_weights(str(weights_path))
                with open(arch_path, "w", encoding="utf-8") as f:
                    f.write(model.to_json())
                saved_model_desc = f"weights at {weights_path} and architecture at {arch_path} (full model save failed: {exc_full}; h5 fallback failed: {exc_h5})"
            except Exception as exc_weights:
                print(f"Saving weights failed: {exc_weights}")
                saved_model_desc = f"failed to save model or weights (errors: {exc_full}; {exc_h5}; {exc_weights})"

    # If no saved artifact exists but a checkpoint was produced, try to re-load checkpoint and save.
    if not any(p.exists() for p in (model_save_path, h5_path, weights_path)):
        best_checkpoint = save_dir / f"{model_name}-best.h5"
        if best_checkpoint.exists():
            try:
                ck_model = tf.keras.models.load_model(str(best_checkpoint))
                ck_model.save(model_save_path)
                saved_model_desc = f"final model re-saved from checkpoint at {model_save_path}"
            except Exception as exc_ck:
                print(f"Re-saving from checkpoint failed: {exc_ck}")

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    print(f"Finished training {model_name}. Saved {saved_model_desc} and class names.")
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train all available bird classification backbones.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="Path to dataset images directory.")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="Directory to save trained models.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Fraction of images used for validation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling.")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation during training.")
    parser.add_argument(
        "--model-names",
        nargs="*",
        help="Optional subset of model names to train. Defaults to all available models.",
    )
    parser.add_argument(
        "--unfreeze-base",
        action="store_true",
        help="Unfreeze the backbone base model during training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")

    train_ds, val_ds, class_names = get_dataset(
        dataset_path,
        IMAGE_SIZE,
        args.batch_size,
        args.validation_split,
        args.seed,
        augment=args.augment,
    )

    available_models = [
        name
        for name, (builder, preprocess) in MODEL_BUILDERS.items()
        if builder is not None and preprocess is not None
    ]
    if not available_models:
        raise RuntimeError("No available model backbones were found in models.py.")

    model_names = args.model_names or available_models
    unknown = [name for name in model_names if name not in available_models]
    if unknown:
        raise ValueError(f"Unknown or unavailable model names: {unknown}. Available: {available_models}")

    print(f"Training dataset: {dataset_path}")
    print(f"Found {len(class_names)} classes.")
    print(f"Models to train: {model_names}")

    for model_name in model_names:
        train_model(
            model_name=model_name,
            train_ds=train_ds,
            val_ds=val_ds,
            class_names=class_names,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            base_trainable=args.unfreeze_base,
        )

    print("\nAll requested models have finished training.")


if __name__ == "__main__":
    main()
