#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import tensorflow as tf

from train_all_models import IMAGE_SIZE, build_model
from ai_engine import MODEL_BUILDERS


def get_val_dataset(dataset_dir, image_size, batch_size, validation_split, seed):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )
    class_names = ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    val_ds = ds.cache().prefetch(AUTOTUNE)
    return val_ds, class_names


def evaluate_model(model_name, model_dir, dataset_dir, batch_size, validation_split, seed):
    model_dir = Path(model_dir)
    dataset_dir = Path(dataset_dir)
    model_path = model_dir / f"{model_name}-best.h5"
    fallback_model_path = model_dir / f"{model_name}-best.keras"
    fallback_model_path_2 = model_dir / f"{model_name}.keras"
    weights_fallback = model_dir / f"{model_name}_weights.h5"
    class_names_path = model_dir / f"{model_name}_class_names.json"

    if not class_names_path.exists():
        raise FileNotFoundError(f"Missing class names file: {class_names_path}")

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    val_ds, ds_class_names = get_val_dataset(dataset_dir, IMAGE_SIZE, batch_size, validation_split, seed)

    # try to load full saved model first
    model = None
    try:
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))
        elif fallback_model_path.exists():
            model = tf.keras.models.load_model(str(fallback_model_path))
        elif fallback_model_path_2.exists():
            model = tf.keras.models.load_model(str(fallback_model_path_2))
    except Exception:
        model = None

    if model is None:
        # rebuild architecture and load weights if available
        if model_name not in MODEL_BUILDERS:
            raise RuntimeError(f"Model builder for {model_name} not found in MODEL_BUILDERS")
        model = build_model(name=model_name, num_classes=len(class_names), input_shape=(*IMAGE_SIZE, 3), preprocess_fn=MODEL_BUILDERS[model_name][1], base_trainable=False)
        # try loading weights
        if model_path.exists():
            try:
                model.load_weights(str(model_path))
            except Exception:
                pass
        if weights_fallback.exists():
            model.load_weights(str(weights_fallback))

    # evaluate
    results = model.evaluate(val_ds, verbose=1)
    # results typically: [loss, accuracy]
    metrics = {"loss": results[0]}
    if len(results) > 1:
        metrics["accuracy"] = results[1]

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset/images", help="Path to dataset images directory")
    ap.add_argument("--model-dir", default="trained_models", help="Directory with saved models")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--validation-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-names", nargs="*", help="Optional list of model names to evaluate")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    available = []
    for p in model_dir.glob("*-best.h5"):
        name = p.name.replace("-best.h5", "")
        available.append(name)
    for p in model_dir.glob("*-best.keras"):
        name = p.name.replace("-best.keras", "")
        if name not in available:
            available.append(name)

    if args.model_names:
        model_names = [n for n in args.model_names]
    else:
        model_names = available

    if not model_names:
        print("No models found to evaluate in", model_dir)
        return

    out = {}
    for name in model_names:
        print(f"\nEvaluating {name}...")
        try:
            metrics = evaluate_model(name, args.model_dir, args.dataset_dir, args.batch_size, args.validation_split, args.seed)
            out[name] = metrics
            print(f"{name}: {metrics}")
        except Exception as exc:
            out[name] = {"error": str(exc)}
            print(f"Failed {name}: {exc}")

    with open("trained_models_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote trained_models_evaluation.json")


if __name__ == "__main__":
    main()
