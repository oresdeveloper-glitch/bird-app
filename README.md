# Bird Species Identification Web App

A Flask-based web application for bird species identification using deep learning models (MobileNetV2, ResNet50, DenseNet121, VGG16, EfficientNetB0) and embeddings-based similarity search.

## Features

- 🦅 **Bird Species Database** - 200+ bird species with conservation status, habitat info, and images
- 🖼️ **Image Upload & Identification** - Upload bird images for automatic species identification
- 🗺️ **Interactive Map** - Visualize bird sightings on a map with habitat information
- 📊 **Model Evaluation** - Evaluate trained models with confusion matrices
- 🔐 **User Authentication** - Secure login system for tracking user sightings
- 📈 **Habitat & Park Management** - Browse habitats and national parks with bird species distribution

## Prerequisites

- Python 3.10+
- pip or conda
- Git

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd jack-project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Trained Models

Models are stored separately due to size. Download them using:

```bash
# Option 1: Download from Hugging Face (if hosted)
# Instructions will be added after model hosting setup

# Option 2: Download from provided cloud storage link
# Download the trained_models.zip file and extract to:
# - trained_models/
```

**Model files needed:**
- `MobileNetV2.keras` or `MobileNetV2-best.h5`
- `ResNet50.keras` or `ResNet50-best.h5`
- `DenseNet121.keras` or `DenseNet121-best.h5`
- `VGG16.keras` or `VGG16-best.h5`
- `EfficientNetB0.keras` or `EfficientNetB0-best.h5`
- `*_class_names.json` (for each model)

### 5. Configure Environment

Create a `.env` file in the project root:

```env
FLASK_ENV=development
FLASK_APP=app.py
DATABASE_URL=sqlite:///bird_app.db
BIRD_MODEL=MobileNetV2
MODEL_PATH=./trained_models/MobileNetV2.keras
SECRET_KEY=your-secret-key-here
```

### 6. Initialize Database

```bash
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.create_all()
>>> exit()
```

## Running the Application

### Development Server
```bash
python app.py
```

Access at: `http://localhost:5000`

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn --workers 4 --bind 0.0.0.0:8000 app:app
```

## Project Structure

```
.
├── app.py                          # Main Flask application
├── models.py                       # Database models (Bird, Habitat, Park, etc.)
├── ai_engine.py                    # ML inference & embedding functions
├── metrics.py                      # Confusion matrix & evaluation metrics
├── identify_cache.py               # Caching for embeddings
├── train_all_models.py             # Training script for all models
├── finetune_best_model.py          # Fine-tuning script
├── evaluate_trained_models.py      # Model evaluation script
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── identify.html
│   ├── species.html
│   ├── habitats.html
│   ├── parks.html
│   └── ...
├── static/                         # Static assets
│   ├── css/app.css
│   ├── js/app.js
│   └── img/
├── trained_models/                 # Trained model files (not in repo)
└── dataset/                        # Dataset folder (not in repo)
```

## Training Models

### Train All Models (30 epochs)
```bash
python train_all_models.py --model-names MobileNetV2 ResNet50 DenseNet121 VGG16 EfficientNetB0 --epochs 30 --batch-size 32
```

### Train Single Model
```bash
python train_all_models.py --model-names MobileNetV2 --epochs 30
```

### With Data Augmentation
```bash
python train_all_models.py --model-names MobileNetV2 --epochs 30 --augment
```

### Fine-tune Best Model
```bash
python finetune_best_model.py --model-name MobileNetV2 --epochs 10 --learning-rate 1e-5
```

## Evaluating Models

```bash
# Evaluate all trained models
python evaluate_trained_models.py

# Evaluate specific model
python evaluate_trained_models.py --model-names MobileNetV2
```

## API Endpoints

- `GET /` - Home page
- `GET /species` - Bird species list
- `POST /identify` - Image upload for identification
- `GET /habitats` - Habitat information
- `GET /parks` - National parks information
- `POST /api/login` - User login
- `GET /dashboard` - User dashboard

## Technologies Used

- **Backend**: Flask, SQLAlchemy, Flask-JWT-Extended
- **ML**: TensorFlow/Keras, scikit-learn
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Database**: SQLite (development), PostgreSQL (production)
- **Object Detection**: YOLOv8 (optional bird localization)

## Dataset

The project uses a custom bird species dataset with 200 classes and ~9,400 training images.

**Dataset Structure:**
```
dataset/
└── images/
    ├── American_Crow/
    ├── American_Goldfinch/
    ├── Barn_Swallow/
    └── ... (200 bird species)
```

## Performance Metrics

Current model performance on validation set (200 classes):

| Model | Accuracy | Loss |
|-------|----------|------|
| MobileNetV2 | ~10% | 4.47 |
| ResNet50 | - | - |
| EfficientNetB0 | - | - |
| DenseNet121 | - | - |
| VGG16 | - | - |

*Note: Low accuracy indicates the model needs more training epochs or additional data*

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

[Your Name]

## Acknowledgments

- Bird dataset contributors
- TensorFlow/Keras community
- Flask framework
- YOLOv8 for object detection

## Troubleshooting

### Model Loading Error
**Problem**: "Failed to load model"
**Solution**: Ensure `.keras` or `.h5` files are in `trained_models/` directory

### Out of Memory
**Problem**: GPU/CPU out of memory during training
**Solution**: Reduce `--batch-size` (e.g., `--batch-size 16`)

### Database Issues
**Problem**: "No such table: bird"
**Solution**: Run database initialization:
```bash
python
>>> from app import app, db
>>> with app.app_context():
>>>     db.create_all()
```

## Support

For issues and questions, please create an GitHub issue.

---

**Last Updated**: 2026-07-04
