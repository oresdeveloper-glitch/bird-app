"""Evaluate NIPE predictions and export a confusion matrix.

Ground truth (Option A): dataset folder names under:
  dataset/images/<SpeciesFolder>/image.jpg

Predictions: existing NIPE pipeline (embedding + nearest reference match).

Outputs:
  - confusion_matrix.json
  - confusion_matrix_top_misclassifications.json

Usage:
  python confusion_matrix_eval.py --limit 200
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, Optional

from ai_engine import predict_bird, precompute_reference_embeddings
from models import Bird, db
from flask import Flask
from identify_cache import ReferenceEmbeddingCache


PROJECT_ROOT = os.path.dirname(__file__)
DATASET_IMAGES = os.path.join(PROJECT_ROOT, "dataset", "images")


def infer_dataset_folder_to_bird_id(birds: List[Bird]) -> Dict[str, int]:
    """Map folder name to Bird.id using Bird.image_url.

    Bird.image_url is expected to look like:
      /bird-image/<folder>/<filename>

    So we map folder -> bird.id.
    """
    mapping: Dict[str, int] = {}
    for b in birds:
        if not b.image_url:
            continue
        parts = b.image_url.strip("/").split("/")
        # /bird-image/<folder>/<filename>
        if len(parts) != 3:
            continue
        _, folder, _filename = parts
        mapping[folder] = b.id
    return mapping


def iter_dataset_images(limit: Optional[int] = None):
    """Yield (folder_name, image_path)."""
    count = 0
    for folder in sorted(os.listdir(DATASET_IMAGES)):
        folder_path = os.path.join(DATASET_IMAGES, folder)
        if not os.path.isdir(folder_path):
            continue
        for fn in sorted(os.listdir(folder_path)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            yield folder, os.path.join(folder_path, fn)
            count += 1
            if limit is not None and count >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max images to evaluate (0 = no limit)")
    ap.add_argument("--output", default=os.path.join(PROJECT_ROOT, "confusion_matrix.json"))
    ap.add_argument("--top-mis", default=os.path.join(PROJECT_ROOT, "confusion_matrix_top_misclassifications.json"))
    ap.add_argument("--normalize", action="store_true", help="Normalize confusion-matrix rows")
    ap.add_argument("--max-classes", type=int, default=0, help="0 = all classes; else include top-N bird classes by DB count")
    args = ap.parse_args()

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///birdhabitat.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()

        birds = Bird.query.all()
        if args.max_classes and args.max_classes > 0:
            birds = Bird.query.limit(args.max_classes).all()

        folder_to_bird_id = infer_dataset_folder_to_bird_id(birds)
        labels = [b.id for b in birds]

        birds_for_ref = Bird.query.all()
        # precompute reference embeddings once
        reference_embeddings = None
        # For best consistency with the running app, we can still compute it directly here.
        embeddings_by_id, model_name_used = precompute_reference_embeddings(birds_for_ref, PROJECT_ROOT, verbose=False)
        reference_embeddings = embeddings_by_id

        y_true: List[int] = []
        y_pred: List[int] = []

        processed = 0
        skipped_no_truth = 0

        for folder, img_path in iter_dataset_images(None if args.limit <= 0 else args.limit):
            true_id = folder_to_bird_id.get(folder)
            if true_id is None:
                skipped_no_truth += 1
                continue

            pred = predict_bird(
                img_path,
                birds_for_ref,
                PROJECT_ROOT,
                reference_embeddings=reference_embeddings,
            )
            pred_id = pred["bird"].id

            y_true.append(true_id)
            y_pred.append(pred_id)
            processed += 1

        from metrics import confusion_matrix, top_misclassifications

        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=args.normalize)

        top_mis = top_misclassifications(y_true, y_pred, labels=labels, top_k=50)

        # compute simple accuracy (fraction correct)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / processed if processed > 0 else 0.0

        # include id->name mapping for convenience
        id_to_name = {b.id: b.common_name for b in birds_for_ref}

        out = {
            "meta": {
                "model_name_used": "NIPE-embeddings",
                "reference_embeddings_model_version": model_name_used,
                "processed": processed,
                "skipped_no_truth": skipped_no_truth,
                "normalize": args.normalize,
                "labels_count": len(labels),
                "correct": correct,
                "accuracy": accuracy,
            },
            "bird_id_to_name": id_to_name,
            "confusion_matrix": cm,
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

        with open(args.top_mis, "w", encoding="utf-8") as f:
            json.dump({"top_misclassifications": top_mis}, f, ensure_ascii=False)

        print(f"Wrote confusion matrix: {args.output}")
        print(f"Wrote top misclassifications: {args.top_mis}")
        print(f"Processed={processed}, skipped_no_truth={skipped_no_truth}, labels={len(labels)}")
        print(f"Correct={correct}, Accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()

