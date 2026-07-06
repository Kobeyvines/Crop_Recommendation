# Rhea Agricultural Recommendation System

A machine learning–powered crop recommendation system that predicts the most suitable crops using soil nutrients and environmental conditions.

The project demonstrates an end-to-end machine learning workflow, from raw data exploration to a deployable recommendation pipeline.

---

## Project Overview

Rhea recommends crops using seven agronomic measurements:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

The project covers the complete machine learning lifecycle:

- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training
- Model evaluation
- Recommendation generation
- Model serialization for deployment

The final recommendation engine uses a **K-Nearest Neighbors (KNN)** classifier and returns both the **best crop recommendation** and the **Top-3 most suitable crops**.

The project is built using the **Crop Recommendation Dataset** by **Atharva Ingle** on Kaggle.

Dataset Summary:

- **2,200 observations**
- **22 crop classes**
- **100 observations per crop (perfectly balanced)**

> **Note:** The dataset does not contain geographical coordinates. Recommendations are therefore based entirely on soil and climatic conditions rather than location.

---

# Repository Structure

```text
├── models/
│   └── rhea_knn_recommendation_pipeline.joblib
├── notebooks/
│   └── analysis.ipynb
├── reports/
│   ├── figures/
│   └── report.pdf
├── requirements.txt
└── README.md
```

> **Why isn't the `data/` folder included?**
>
> The dataset is intentionally excluded from Git to keep the repository lightweight. Git LFS and DVC are configured for dataset management during development, while the repository only contains the source code and trained model.

---

# Installation

Clone the repository.

```bash
git clone https://github.com/Kobeyvines/Crop_Recommendation.git

cd Crop_Recommendation
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Linux / macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install the dependencies.

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

The dataset is intentionally **not included** in this repository to keep the project lightweight. During development, **Git LFS** and **DVC** were used to manage the dataset locally, while the `data/` directory is excluded from version control via `.gitignore`.

### 1. Create the required directory structure

**Linux / macOS**

```bash
mkdir -p data/raw data/processed
```

**Windows (Command Prompt)**

```cmd
mkdir data\raw
mkdir data\processed
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path data/raw, data/processed
```

### 2. Download the dataset

Download the **Crop Recommendation Dataset** from Kaggle:

https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

### 3. Place the dataset in the raw folder

After downloading, place the CSV file at:

```text
data/raw/Crop_recommendation.csv
```

When the notebook is executed, it will automatically generate the cleaned dataset and save it to:

```text
data/processed/cleaned_crop_data.csv
```

# Running the Project

Open:

```text
notebooks/analysis.ipynb
```

Run every notebook cell sequentially from top to bottom.

The notebook performs:

1. Data cleaning
2. Exploratory Data Analysis
3. Feature Engineering
4. KNN Model Training
5. Model Evaluation
6. Recommendation Generation
7. Model Serialization

The trained pipeline is exported to:

```text
models/rhea_knn_recommendation_pipeline.joblib
```

---

# Preventing Data Leakage

One of the primary design goals of this project was to ensure that model evaluation reflects real-world deployment.

The dataset is first divided into an **80% training set** and **20% test set** using stratified sampling.

Only the training data is used to learn preprocessing parameters and engineered feature statistics.

A custom Scikit-Learn transformer (`AgronomicFeatureEngineer`) is responsible for computing engineered features such as:

- NPK Score
- Climate Suitability Index
- Soil pH Category

These transformations are fitted exclusively on the training data and then applied unchanged to the test set and future farmer inputs through a Scikit-Learn `Pipeline`.

This design prevents information from the test data leaking into the training process and mirrors how the recommendation engine would behave in production.

---

# Model Performance

The final KNN recommendation pipeline was evaluated on a held-out test set containing **440 observations**.

| Metric | Result |
|:---|---:|
| Test Samples | 440 |
| Top-1 Accuracy | **97.50%** |
| Top-3 Accuracy | **99.77%** |
| 5-Fold Cross-Validation | **97.27% ± 0.66%** |

These results indicate that the recommendation engine predicts the correct crop for nearly every unseen observation and almost always includes the correct crop within its top three recommendations.

The low cross-validation variance demonstrates that the model generalizes consistently across different train-validation splits.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- Git LFS
- DVC

---

# Outputs

Running the notebook produces:

- Cleaned dataset
- Exploratory visualizations
- Engineered features
- Trained KNN recommendation model
- Serialized deployment pipeline (`rhea_knn_recommendation_pipeline.joblib`)

---

# License

This project is intended for educational and portfolio purposes.

The dataset remains the property of its original author and must be downloaded separately from Kaggle.