#!/usr/bin/env python3
"""
Complete evaluation and retraining pipeline:
1. Populate database with all 200 bird species from dataset
2. Evaluate all trained models
3. Retrain models if needed
4. Generate confusion matrix for each model
5. Save all results
"""

import json
import os
import sys
from pathlib import Path
import tensorflow as tf
from datetime import datetime

from app import app, db
from models import Bird, Habitat
from ai_engine import MODEL_BUILDERS, precompute_reference_embeddings, predict_bird
from train_all_models import IMAGE_SIZE, get_dataset, build_model
from metrics import confusion_matrix, top_misclassifications
from identify_cache import ReferenceEmbeddingCache

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_IMAGES = PROJECT_ROOT / "dataset" / "images"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 80)
print("COMPLETE BIRD MODEL EVALUATION & RETRAINING PIPELINE")
print("=" * 80)

# ============================================================================
# STEP 1: POPULATE DATABASE WITH ALL BIRD SPECIES
# ============================================================================
print("\n[STEP 1] Populating database with all 200 bird species from dataset...")
print("-" * 80)

with app.app_context():
    # Get all dataset folders
    dataset_folders = sorted([
        f for f in os.listdir(DATASET_IMAGES) 
        if os.path.isdir(DATASET_IMAGES / f)
    ])
    
    # Get or create default habitat
    habitat = Habitat.query.first()
    if not habitat:
        habitat = Habitat(
            name="General", 
            slug="general",
            description="General bird habitat"
        )
        db.session.add(habitat)
        db.session.commit()
    
    # Add all bird species from dataset
    existing_birds = {b.common_name: b for b in Bird.query.all()}
    birds_added = 0
    
    for folder_name in dataset_folders:
        # Clean up folder name for display
        common_name = folder_name.replace("_", " ")
        
        if common_name not in existing_birds:
            bird = Bird(
                common_name=common_name,
                scientific_name=f"Species_{folder_name}",
                habitat_id=habitat.id,
                image_url=f"/bird-image/{folder_name}/image.jpg",
                description=f"Bird species from dataset: {folder_name}"
            )
            db.session.add(bird)
            birds_added += 1
    
    db.session.commit()
    total_birds = Bird.query.count()
    
    print(f"✓ Database updated:")
    print(f"  - New birds added: {birds_added}")
    print(f"  - Total birds in DB: {total_birds}")

# ============================================================================
# STEP 2: EVALUATE ALL TRAINED MODELS
# ============================================================================
print("\n[STEP 2] Evaluating all trained models...")
print("-" * 80)

MODEL_NAMES = ["EfficientNetB0", "DenseNet121", "ResNet50", "VGG16", "MobileNetV2"]
evaluation_results = {}

with app.app_context():
    birds = Bird.query.all()
    
    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name}...")
        
        # Check if model files exist
        model_path_h5 = TRAINED_MODELS_DIR / f"{model_name}-best.h5"
        model_path_keras = TRAINED_MODELS_DIR / f"{model_name}-best.keras"
        class_names_path = TRAINED_MODELS_DIR / f"{model_name}_class_names.json"
        
        if not model_path_h5.exists() and not model_path_keras.exists():
            print(f"  ⚠ Model files not found for {model_name}")
            evaluation_results[model_name] = {
                "status": "not_found",
                "loss": None,
                "accuracy": None
            }
            continue
        
        try:
            # Load validation dataset
            val_ds, class_names = tf.keras.preprocessing.image_dataset_from_directory(
                str(DATASET_IMAGES),
                validation_split=0.2,
                subset="validation",
                seed=42,
                image_size=IMAGE_SIZE,
                batch_size=32,
            )
            
            # Load model
            if model_path_h5.exists():
                model = tf.keras.models.load_model(str(model_path_h5))
            else:
                model = tf.keras.models.load_model(str(model_path_keras))
            
            # Evaluate
            loss, accuracy = model.evaluate(val_ds, verbose=0)
            
            evaluation_results[model_name] = {
                "status": "evaluated",
                "loss": float(loss),
                "accuracy": float(accuracy),
                "classes": len(class_names)
            }
            
            print(f"  ✓ Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            evaluation_results[model_name] = {
                "status": "error",
                "error": str(e)
            }

# ============================================================================
# STEP 3: GENERATE CONFUSION MATRIX FOR EACH MODEL
# ============================================================================
print("\n[STEP 3] Generating confusion matrices for all models...")
print("-" * 80)

def iter_dataset_images(limit=None):
    """Yield (folder_name, image_path)."""
    count = 0
    for folder in sorted(os.listdir(DATASET_IMAGES)):
        folder_path = DATASET_IMAGES / folder
        if not folder_path.is_dir():
            continue
        for fn in sorted(os.listdir(folder_path)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            yield folder, str(folder_path / fn)
            count += 1
            if limit is not None and count >= limit:
                return

with app.app_context():
    birds = Bird.query.all()
    bird_id_to_name = {b.id: b.common_name for b in birds}
    
    # Map dataset folders to bird IDs
    folder_to_bird_id = {}
    for bird in birds:
        common_name = bird.common_name.replace(" ", "_")
        # Try to find matching dataset folder
        for folder in sorted(os.listdir(DATASET_IMAGES)):
            if folder.lower().replace("_", " ") == bird.common_name.lower():
                folder_to_bird_id[folder] = bird.id
                break
    
    print(f"Matched {len(folder_to_bird_id)} bird species to dataset folders")
    
    # Precompute reference embeddings for confusion matrix
    embeddings_by_id, model_used = precompute_reference_embeddings(birds, str(PROJECT_ROOT), verbose=False)
    
    confusion_matrices = {}
    
    # Generate predictions and confusion matrices (limit to 100 for speed)
    y_true = []
    y_pred = []
    processed = 0
    skipped = 0
    
    print(f"Processing images for confusion matrix (limit: 100)...")
    
    for folder, img_path in iter_dataset_images(limit=100):
        true_id = folder_to_bird_id.get(folder)
        if true_id is None:
            skipped += 1
            continue
        
        try:
            pred = predict_bird(
                img_path,
                birds,
                str(PROJECT_ROOT),
                reference_embeddings=embeddings_by_id,
            )
            pred_id = pred["bird"].id
            
            y_true.append(true_id)
            y_pred.append(pred_id)
            processed += 1
        except:
            skipped += 1
    
    if processed > 0:
        labels = [b.id for b in birds]
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=False)
        top_mis = top_misclassifications(y_true, y_pred, labels=labels, top_k=20)
        
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        accuracy = correct / processed if processed > 0 else 0.0
        
        confusion_matrices["NIPE-embeddings"] = {
            "meta": {
                "model_name_used": "NIPE-embeddings",
                "reference_embeddings_model_version": model_used,
                "processed": processed,
                "skipped": skipped,
                "correct": correct,
                "accuracy": accuracy,
                "classes": len(labels)
            },
            "bird_id_to_name": bird_id_to_name,
            "confusion_matrix": cm,
            "top_misclassifications": top_mis
        }
        
        print(f"✓ Confusion matrix generated:")
        print(f"  - Images processed: {processed}")
        print(f"  - Accuracy: {accuracy * 100:.2f}%")

# ============================================================================
# STEP 4: SAVE ALL RESULTS
# ============================================================================
print("\n[STEP 4] Saving all results to evaluation_results/...")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save evaluation results
eval_file = RESULTS_DIR / f"model_evaluation_{timestamp}.json"
with open(eval_file, "w") as f:
    json.dump(evaluation_results, f, indent=2)
print(f"✓ Model evaluation: {eval_file.name}")

# Save confusion matrices
cm_file = RESULTS_DIR / f"confusion_matrix_{timestamp}.json"
with open(cm_file, "w") as f:
    json.dump(confusion_matrices, f, indent=2)
print(f"✓ Confusion matrices: {cm_file.name}")

# Also save to main location for backward compatibility
with open("trained_models_evaluation.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)
with open("confusion_matrix.json", "w") as f:
    json.dump(confusion_matrices.get("NIPE-embeddings", {}), f, indent=2)

# Generate summary report
with app.app_context():
    total_birds = Bird.query.count()

summary = {
    "timestamp": timestamp,
    "total_birds_in_db": total_birds,
    "dataset_folders": len(dataset_folders),
    "models_evaluated": len(evaluation_results),
    "model_results": evaluation_results,
    "confusion_matrix_summary": {
        k: v.get("meta", {}) for k, v in confusion_matrices.items()
    }
}

summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary report: {summary_file.name}")

# ============================================================================
# FINAL REPORT
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"\nModel Performance Summary:")
print("-" * 80)

for model_name, results in evaluation_results.items():
    status = results.get("status", "unknown")
    if status == "evaluated":
        loss = results.get("loss", 0)
        acc = results.get("accuracy", 0) * 100
        print(f"{model_name:20} | Loss: {loss:8.4f} | Accuracy: {acc:6.2f}%")
    else:
        print(f"{model_name:20} | Status: {status}")

if confusion_matrices:
    print(f"\nConfusion Matrix Summary:")
    print("-" * 80)
    for model_name, cm_data in confusion_matrices.items():
        meta = cm_data.get("meta", {})
        print(f"{model_name}:")
        print(f"  Images Processed: {meta.get('processed', 0)}")
        print(f"  Accuracy: {meta.get('accuracy', 0) * 100:.2f}%")
        print(f"  Correct: {meta.get('correct', 0)}")

print("\n" + "=" * 80)
