# Rhea Data Science Internship Assessment

## Overview
This project sources, cleans, analyzes, and prepares an agricultural dataset
for use in Rhea's crop recommendation system, following the 5-task
assessment brief (data sourcing, cleaning, EDA, feature engineering, and
recommendation logic).

## Project Structure
```
rhea-assessment/
├── data/
│   ├── raw/            # original, untouched dataset
│   └── processed/      # cleaned dataset used for analysis
├── notebooks/
│   └── analysis.ipynb  # full analysis, sectioned by task
├── src/
│   ├── cleaning.py     # reusable data cleaning functions
│   └── features.py     # reusable feature engineering functions
├── reports/
│   ├── figures/        # exported EDA visualizations
│   └── report.pdf      # 5-page summary report (submission deliverable)
└── requirements.txt
```

## How to Run
1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Open `notebooks/analysis.ipynb` and run all cells top to bottom.

## Status
🚧 In progress — sections will be filled in as each task is completed.
