# PersonaVerify: Machine Learning Profile Authenticity Dashboard

PersonaVerify is an end-to-end Machine Learning web application designed to detect fake social media profiles. It utilizes a `RandomForestClassifier` trained on account metadata and linguistic features, and provides a highly technical, data-science-grade dashboard for interpreting those predictions using Explainable AI (XAI) techniques.

## Key Features

- **Random Forest Prediction Engine:** High accuracy classification using 100 decision trees to determine if a profile is genuine or fake based on 11 core features (e.g., follower ratio, profile pictures, description length, etc.).
- **Explainable AI (XAI):** Implements Local Feature Interpretability through **Tree Decision Path Decomposition**. The backend extracts specific feature contributions (+/- impact on probability) and renders them in a localized waterfall chart, completely demystifying the black box.
- **Bulk CSV Analysis:** Support for uploading large datasets of account records via `.csv` to process batches of profiles in milliseconds. 
- **Telemetry Dashboard:** Built-in prediction tracking metrics using a persistent SQLite backend to monitor real-time usage (Fake vs. Real classification stats).
- **Modern Architecture:** 
  - **Backend**: Django REST Framework (DRF), `scikit-learn`, `pandas`, `numpy`
  - **Frontend**: Custom CSS Dashboard API with dynamic DOM rendering

## Project Layout

```
PersonaVerify/
├── backend/            # Django REST API and Machine Learning prediction services
│   ├── predictor/      # Core app logic (models, views, services, explainer engine)
│   ├── model/          # Contains the trained .joblib ML models and scalers
├── frontend/           # Static HTML/CSS/JS single-page application dashboard
└── venv/               # (Excluded) Python Virtual Environment
```

## Running Locally

### 1. Setup Backend
```bash
cd backend
python -m venv ../venv
source ../venv/bin/activate
pip install -r requirements.txt # (Ensure django, djangorestframework, scikit-learn, django-cors-headers, pandas are installed)
python manage.py migrate
python manage.py runserver
```

### 2. Setup Frontend
Simply open `frontend/index.html` in any modern web browser (e.g. Chrome). The API calls will route dynamically to `localhost:8000`.

## Explainable AI Methodology
The XAI engine (`backend/predictor/explainer.py`) operates by tracking the decision path of a given sample through all trees in the Random Forest. It evaluates the shift in class probability node-by-node to determine exactly *why* a profile was flagged as fake, generating human-readable risk factors in the process.
