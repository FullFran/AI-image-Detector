# AI Image Detector

Project for detecting AI-generated images using spectral and gradient analysis.

## Structure

- **src/ai_detector**: Core logic and features extraction (Gradient, DFT, DCT, Color).
- **src/app**: FastAPI application (API) and Services.
- **notebooks/**: Jupyter notebooks for demonstration and analysis.
- **scripts/**: CLI scripts for batch processing.
- **data/**: Data storage (raw images, processed).

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Run API

```bash
uvicorn app.main:app --reload --app-dir src
```

### Run CLI

```bash
python scripts/classify_folder.py data/raw
```

### Notebooks

Open `notebooks/01_gradient_covariance_analysis.ipynb` or `notebooks/02_advanced_spectral_analysis.ipynb` in Jupyter.
