"""
rhea_features.py

Custom feature engineering transformer used by the Rhea Crop
Recommendation System.

This transformer creates three agronomic features:

1. NPK_Score
2. Climate_Suitability_Index
3. ph_category

The transformer is designed to work inside a Scikit-Learn Pipeline,
ensuring that all statistics are learned only from the training data,
thereby preventing data leakage.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class AgronomicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn transformer for engineering agronomic features.

    Engineered Features
    -------------------
    1. NPK_Score
       Mean of min-max scaled Nitrogen, Phosphorus and Potassium.

    2. Climate_Suitability_Index
       Mean of standardized Temperature, Humidity and Rainfall.

    3. ph_category
       Soil acidity category:
           0 = Acidic
           1 = Neutral
           2 = Alkaline
    """

    def __init__(self):
        self.npk_stats_ = {}
        self.climate_stats_ = {}

    def fit(self, X, y=None):
        """
        Learn feature statistics from the training data only.
        """

        X = X.copy()

        # Learn min/max for NPK scaling
        for feature in ["N", "P", "K"]:
            self.npk_stats_[feature] = {
                "min": X[feature].min(),
                "max": X[feature].max(),
            }

        # Learn mean/std for climate standardization
        for feature in ["temperature", "humidity", "rainfall"]:
            self.climate_stats_[feature] = {
                "mean": X[feature].mean(),
                "std": X[feature].std(),
            }

        return self

    def transform(self, X):
        """
        Apply feature engineering using the statistics learned during fit().
        """

        X = X.copy()

        # =====================================================
        # Feature 1: NPK Score
        # =====================================================

        scaled_features = []

        for feature in ["N", "P", "K"]:

            minimum = self.npk_stats_[feature]["min"]
            maximum = self.npk_stats_[feature]["max"]

            denominator = maximum - minimum

            if denominator == 0:
                scaled = np.zeros(len(X))
            else:
                scaled = (X[feature] - minimum) / denominator

            scaled_features.append(scaled)

        X["NPK_Score"] = np.mean(scaled_features, axis=0)

        # =====================================================
        # Feature 2: Climate Suitability Index
        # =====================================================

        standardized_features = []

        for feature in ["temperature", "humidity", "rainfall"]:

            mean = self.climate_stats_[feature]["mean"]
            std = self.climate_stats_[feature]["std"]

            if std == 0:
                z = np.zeros(len(X))
            else:
                z = (X[feature] - mean) / std

            standardized_features.append(z)

        X["Climate_Suitability_Index"] = np.mean(
            standardized_features,
            axis=0,
        )

        # =====================================================
        # Feature 3: Soil pH Category
        # =====================================================

        conditions = [
            X["ph"] < 6.5,
            X["ph"].between(6.5, 7.5),
        ]

        choices = [0, 1]

        X["ph_category"] = np.select(
            conditions,
            choices,
            default=2,
        ).astype(int)

        return X