#!/usr/bin/env python3
"""Fine-tune a saved bird classification model on the validation split."""

import argparse
import json
from pathlib import Path

import tensorflow as tf

from train_all_models import IMAGE_SIZE, build_model
from ai_engine import MODEL_BUILDERS


def get_dataset(dataset_dir, image_size, batch_size, validation_split, seed):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def load_or_build_model(model_name, model_dir, num_classes):
    model_dir = Path(model_dir)
    best_model_path = model_dir / f"{model_name}-best.h5"
    best_model_path_keras = model_dir / f"{model_name}-best.keras"
    final_model_path = model_dir / f"{model_name}.keras"

    model = None
    if best_model_path.exists():
        try:
            model = tf.keras.models.load_model(str(best_model_path))
        except Exception:
            model = None
    if model is None and best_model_path_keras.exists():
        try:
            model = tf.keras.models.load_model(str(best_model_path_keras))
        except Exception:
            model = None
    if model is None and final_model_path.exists():
        try:
            model = tf.keras.models.load_model(str(final_model_path))
        except Exception:
            model = None

    if model is None:
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"Unknown model name: {model_name}")
        builder, preprocess = MODEL_BUILDERS[model_name]
        if builder is None:
            raise ValueError(f"Model builder unavailable for {model_name}")
        model = build_model(
            name=model_name,
            num_classes=num_classes,
            input_shape=(*IMAGE_SIZE, 3),
            preprocess_fn=preprocess,
            base_trainable=True,
        )
    return model


def fine_tune_model(
    model_name,
    dataset_dir,
    model_dir,
    batch_size,
    validation_split,
    seed,
    epochs,
    lr,
    patience,
    output_name,
):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    label_map_path = model_dir / f"{model_name}_class_names.json"

    if not label_map_path.exists():
        raise FileNotFoundError(f"Missing class names file: {label_map_path}")

    with open(label_map_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    train_ds, val_ds, class_names_ds = get_dataset(dataset_dir, IMAGE_SIZE, batch_size, validation_split, seed)
    if len(class_names_ds) != len(class_names):
        print("Warning: dataset class count differs from saved class names; using saved class names.")

    model = load_or_build_model(model_name, model_dir, len(class_names))

    # unfreeze all trainable layers for fine-tuning
    model.trainable = True
    for layer in model.layers:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    output_path = model_dir / output_name
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output_path),
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    # Attempt to persist the fine-tuned model. Prefer a full Keras SavedModel (.keras),
    # fall back to HDF5 (.h5), then weights+architecture JSON. If a best checkpoint
    # exists, attempt to re-load from it and re-save.
    final_save_path = model_dir / f"{model_name}-finetuned.keras"
    h5_path = model_dir / f"{model_name}-finetuned.h5"
    weights_path = model_dir / f"{model_name}_finetuned_weights.h5"
    arch_path = model_dir / f"{model_name}_finetuned_architecture.json"

    saved_desc = None
    try:
        model.save(final_save_path)
        saved_desc = f"final model at {final_save_path}"
    except Exception as exc_full:
        print(f"Full model save to {final_save_path} failed: {exc_full}")
        try:
            model.save(str(h5_path), save_format="h5")
            saved_desc = f"HDF5 model at {h5_path}"
        except Exception as exc_h5:
            print(f"HDF5 save to {h5_path} failed: {exc_h5}")
            try:
                model.save_weights(str(weights_path))
                with open(arch_path, "w", encoding="utf-8") as f:
                    f.write(model.to_json())
                saved_desc = f"weights at {weights_path} and architecture at {arch_path} (fallback)"
            except Exception as exc_weights:
                print(f"Saving weights/architecture failed: {exc_weights}")
                saved_desc = f"failed to save model (errors: {exc_full}; {exc_h5}; {exc_weights})"

    # If nothing was written, try re-loading the best checkpoint and saving that instead.
    if not any(p.exists() for p in (final_save_path, h5_path, weights_path)):
        if output_path.exists():
            try:
                ck_model = tf.keras.models.load_model(str(output_path))
                ck_model.save(final_save_path)
                saved_desc = f"re-saved final model from checkpoint to {final_save_path}"
            except Exception as exc_ck:
                print(f"Re-saving from checkpoint failed: {exc_ck}")

    print(f"Fine-tuning complete. Best checkpoint: {output_path}. Saved: {saved_desc}")
    return history, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a trained bird model.")
    parser.add_argument("--model-name", default="EfficientNetB0", help="Model name to fine-tune")
    parser.add_argument("--dataset-dir", default="dataset/images", help="Dataset images root")
    parser.add_argument("--model-dir", default="trained_models", help="Directory containing saved models")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset shuffling")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--output-name", default=None, help="Output model filename")
    return parser.parse_args()


def main():
    args = parse_args()
    output_name = args.output_name or f"{args.model_name}-finetuned.keras"
    fine_tune_model(
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=args.seed,
        epochs=args.epochs,
        lr=args.learning_rate,
        patience=args.patience,
        output_name=output_name,
    )


if __name__ == "__main__":
    main()
