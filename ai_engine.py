from functools import lru_cache
from PIL import Image, ImageDraw, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS
import hashlib
import math
import os

try:
    import numpy as np
except Exception:
    np = None

try:
    import tensorflow as tf
except Exception:
    tf = None

keras_image = None
ResNet50 = None
VGG16 = None
DenseNet121 = None
MobileNetV2 = None
EfficientNetB0 = None
ViT = None
resnet_preprocess = None
vgg_preprocess = None
densenet_preprocess = None
mobilenet_preprocess = None
efficientnet_preprocess = None
vit_preprocess = None

try:
    from tensorflow.keras.preprocessing import image as keras_image
except Exception:
    keras_image = None

if tf is not None:
    try:
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
    except Exception:
        ResNet50 = None
        resnet_preprocess = None
    try:
        from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
    except Exception:
        VGG16 = None
        vgg_preprocess = None
    try:
        from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as densenet_preprocess
    except Exception:
        DenseNet121 = None
        densenet_preprocess = None
    try:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
    except Exception:
        MobileNetV2 = None
        mobilenet_preprocess = None
    try:
        from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
    except Exception:
        EfficientNetB0 = None
        efficientnet_preprocess = None
    try:
        from tensorflow.keras.applications.vision_transformer import ViT, preprocess_input as vit_preprocess
    except Exception:
        ViT = None
        vit_preprocess = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

IMAGE_SIZE = (224, 224)

MODEL_BUILDERS = {
    "ResNet50": (ResNet50, resnet_preprocess),
    "VGG16": (VGG16, vgg_preprocess),
    "DenseNet121": (DenseNet121, densenet_preprocess),
    "MobileNetV2": (MobileNetV2, mobilenet_preprocess),
    "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess),
    "ViT": (ViT, vit_preprocess),
}

DEFAULT_MODEL_NAME = os.getenv("BIRD_MODEL", "MobileNetV2")
MODEL_FALLBACK_ORDER = ["MobileNetV2", "ResNet50", "EfficientNetB0", "DenseNet121", "VGG16", "ViT"]


def _get_active_model_name():
    requested = DEFAULT_MODEL_NAME
    if requested in MODEL_BUILDERS and MODEL_BUILDERS[requested][0] is not None:
        return requested
    for fallback in MODEL_FALLBACK_ORDER:
        if MODEL_BUILDERS.get(fallback, (None, None))[0] is not None:
            return fallback
    return None


def active_image_model_name():
    return _get_active_model_name()


@lru_cache(maxsize=None)
def _load_model(name):
    if tf is None or name is None:
        return None
    builder, _ = MODEL_BUILDERS.get(name, (None, None))
    if builder is None:
        return None
    try:
        return builder(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    except Exception:
        return None


def secure_image_digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def validate_and_compress_image(file_storage, output_path, max_size=(1280, 1280)):
    try:
        image = Image.open(file_storage.stream)
        image.verify()
        file_storage.stream.seek(0)
        image = Image.open(file_storage.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("Upload must be a valid JPG, PNG, or WEBP bird image.") from exc

    image.thumbnail(max_size)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path, format="JPEG", quality=82, optimize=True)
    return output_path


def precompute_reference_embeddings(birds, project_root, verbose: bool = False):
    """Precompute embeddings for all birds (reference images).

    Returns:
      (embeddings_by_bird_id, model_name_used)
    """
    embeddings_by_bird_id = {}
    model_name_used = None

    for idx, bird in enumerate(birds):
        sample_path = _image_url_to_path(bird.image_url, project_root)
        if not sample_path:
            continue

        vec, model_name = _embedding(sample_path)
        if vec is None:
            continue

        embeddings_by_bird_id[bird.id] = vec
        if model_name_used is None:
            model_name_used = model_name

        if verbose and (idx + 1) % 25 == 0:
            print(f"[precompute] embedded {idx + 1}/{len(birds)} birds")

    return embeddings_by_bird_id, model_name_used



def _tensorflow_embedding(path):
    model_name = _get_active_model_name()
    model = _load_model(model_name)
    preprocessor = MODEL_BUILDERS.get(model_name, (None, None))[1]
    if model is None or keras_image is None or preprocessor is None or np is None:
        return None
    img = keras_image.load_img(path, target_size=IMAGE_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocessor(arr)
    vector = model.predict(arr, verbose=0)[0]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _classic_embedding(path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    if np is None:
        pixels = list(image.getdata())
        avg = [sum(channel) / len(pixels) / 255 for channel in zip(*pixels)]
        return avg

    arr = np.asarray(image).astype("float32") / 255.0
    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=24, range=(0, 1), density=True)
        hist_parts.append(hist)
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    edge_hist, _ = np.histogram(np.sqrt(gx * gx + gy * gy), bins=16, range=(0, 0.5), density=True)
    vector = np.concatenate(hist_parts + [edge_hist])
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _embedding(path):
    model_name = _get_active_model_name()
    tf_vector = _tensorflow_embedding(path)
    if tf_vector is not None:
        return tf_vector, f"{model_name} ImageNet feature extractor"
    return _classic_embedding(path), "Pillow color-edge feature matcher"


def _load_yolov8_model():
    if YOLO is None:
        return None
    weights = os.getenv("YOLOV8_WEIGHTS", "yolov8n.pt")
    try:
        return YOLO(weights)
    except Exception:
        return None


def localize_bird_regions(image_path, output_path=None, conf=0.25):
    model = _load_yolov8_model()
    if model is None or np is None:
        return None
    try:
        results = model.predict(source=image_path, conf=conf, verbose=False)
    except Exception:
        return None

    if not results:
        return None

    result = results[0]
    boxes = []
    try:
        xyxy = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls
        names = getattr(result, "names", None) or {}
        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        confs = confs.cpu().numpy() if hasattr(confs, "cpu") else np.asarray(confs)
        classes = classes.cpu().numpy() if hasattr(classes, "cpu") else np.asarray(classes)
    except Exception:
        return None

    for idx in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[idx]]
        label = None
        try:
            cls = int(classes[idx])
            label = names.get(cls, str(cls))
        except Exception:
            label = "object"
        boxes.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(confs[idx]) if confs is not None else None,
                "label": label,
            }
        )

    if not boxes:
        return None

    if output_path is None:
        output_path = os.path.splitext(image_path)[0] + "-yolo.jpg"

    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = None
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            font = None
        for detection in boxes:
            x1, y1, x2, y2 = detection["box"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            if detection["label"]:
                text = f"{detection['label']} {detection['confidence']:.2f}" if detection['confidence'] is not None else detection["label"]
                text_position = (x1 + 4, y1 + 4)
                draw.text(text_position, text, fill=(255, 255, 255), font=font)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, format="JPEG", quality=90)
    except Exception:
        return None

    return {
        "overlay_path": output_path,
        "detections": boxes,
    }


def _cosine_similarity(a, b):
    if np is not None:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1))
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / ((norm_a * norm_b) or 1)


def _image_url_to_path(image_url, project_root):
    if not image_url or not image_url.startswith("/bird-image/"):
        return None
    parts = image_url.strip("/").split("/")
    if len(parts) != 3:
        return None
    _, folder, filename = parts
    path = os.path.join(project_root, "dataset", "images", folder, filename)
    return path if os.path.exists(path) else None


def _gradcam_heatmap(image_path, reference_embedding=None):
    """Generate Grad-CAM heatmap array (0-1)."""
    model_name = _get_active_model_name()
    if tf is None or model_name is None:
        return None
    base_model = _load_model(model_name)
    preprocessor = MODEL_BUILDERS.get(model_name, (None, None))[1]
    if base_model is None or preprocessor is None or keras_image is None or np is None:
        return None

    spatial_output = None
    for layer in reversed(base_model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if shape is not None and len(shape) == 4:
            spatial_output = layer.output
            break
    if spatial_output is None:
        return None

    grad_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=[spatial_output, base_model.output])
    img = keras_image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocessor(img_array)

    with tf.GradientTape() as tape:
        conv_out, emb = grad_model(img_array)
        if reference_embedding is not None:
            ref = tf.constant(reference_embedding.reshape(1, -1), dtype=tf.float32)
            loss = tf.reduce_sum(tf.nn.l2_normalize(emb, axis=1) * tf.nn.l2_normalize(ref, axis=1))
        else:
            loss = tf.reduce_sum(emb ** 2)

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)
    heatmap = np.maximum(heatmap.numpy()[0], 0)
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    return heatmap


def generate_gradcam_heatmap(image_path, output_path, reference_embedding=None, alpha=0.4):
    """Generate Grad-CAM overlay image showing which image regions drove the prediction."""
    heatmap = _gradcam_heatmap(image_path, reference_embedding)
    if heatmap is None:
        return None
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    original = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    raw = Image.fromarray(np.uint8(255 * heatmap)).resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    colored = plt.get_cmap("jet")(np.array(raw) / 255.0)
    overlay = Image.blend(original, Image.fromarray((colored[:, :, :3] * 255).astype(np.uint8)), alpha)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay.save(output_path, format="JPEG", quality=90)
    return output_path


def extract_exif_metadata(image_path):
    """Extract GPS, timestamp, and camera info from an image's EXIF data."""
    result = {"gps": None, "timestamp": None, "camera": None}
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return result
    except Exception:
        return result

    def _to_decimal(values, ref):
        if not values or len(values) != 3:
            return None
        d, m, s = values
        dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
        return -dec if ref in ("S", "W") else dec

    gps_raw = None
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            gps = {}
            for gtid, gv in value.items():
                gps[GPSTAGS.get(gtid, gtid)] = gv
            lat = _to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
            lon = _to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))
            if lat is not None and lon is not None:
                result["gps"] = {"lat": round(lat, 6), "lon": round(lon, 6)}
        elif tag == "DateTimeOriginal":
            result["timestamp"] = str(value)
        elif tag in ("Make", "Model"):
            if result["camera"] is None:
                result["camera"] = {}
            result["camera"][tag.lower()] = str(value)
    return result


def predict_bird(upload_path, birds, project_root, reference_embeddings=None):
    """Predict bird by comparing upload embedding to reference embeddings.

    reference_embeddings:
      dict mapping bird.id -> embedding vector
    """
    upload_vector, model_name = _embedding(upload_path)
    if upload_vector is None:
        raise ValueError("Failed to extract features from the uploaded image.")

    candidates = []

    if reference_embeddings is not None:
        # Fast path: avoid embedding reference images in request loop.
        for bird in birds:
            ref_vec = reference_embeddings.get(bird.id)
            if ref_vec is None:
                continue
            similarity = _cosine_similarity(upload_vector, ref_vec)
            candidates.append((similarity, bird))
    else:
        # Slow path: legacy behavior (embeds every reference image per request).
        for bird in birds:
            sample_path = _image_url_to_path(bird.image_url, project_root)
            if not sample_path:
                continue
            sample_vector, _ = _embedding(sample_path)
            similarity = _cosine_similarity(upload_vector, sample_vector)
            candidates.append((similarity, bird))

    if not candidates:
        raise ValueError("No Tanzania bird reference images are available for AI comparison.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_bird = candidates[0]
    confidence = max(1, min(99, round(((best_score + 1) / 2) * 100, 2)))
    alternatives = [
        {"bird": bird, "confidence": max(1, min(99, round(((score + 1) / 2) * 100, 2)))}
        for score, bird in candidates[1:4]
    ]

    return {
        "bird": best_bird,
        "confidence": confidence,
        "alternatives": alternatives,
        "model_name": model_name,
    }

