# Import Libraries
import os
import sys

# Make src/ imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

from utils.config import FeatureEngineeringConfig
from mylogging.custom_logger import get_logger
from exception.exception import EasyLaborPredictionException

logger = get_logger(__name__)


# 📢 Initialize logger
logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.config = FeatureEngineeringConfig()

    def run(self):
        """Run the full feature engineering process."""
        try:
            logger.info("🔵 Starting Feature Engineering...")

            # Load data
            df = pd.read_csv(self.config.data_path)
            df.drop(columns=['case_id'], inplace=True)
            logger.info(f"✅ Loaded data from {self.config.data_path}")

            target_column = 'case_status'
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Label Encode categorical features
            categorical_cols = [
                'continent', 'education_of_employee', 'has_job_experience',
                'requires_job_training', 'region_of_employment', 'full_time_position', 'unit_of_wage'
            ]

            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le

            joblib.dump(label_encoders, self.config.label_encoder_pkl)
            logger.info("✅ Saved Label Encoders.")

            # Scale numerical features
            numeric_cols = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']
            scaler = MinMaxScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            joblib.dump(scaler, self.config.scaler_pkl)
            logger.info("✅ Saved Scaler.")

            # Encode target
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

            joblib.dump(target_encoder, self.config.target_encoder_pkl)
            logger.info("✅ Saved Target Encoder.")

            # Balance with SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

            # Save full balanced dataset
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df[target_column] = y_resampled
            resampled_df.to_csv(self.config.output_path, index=False)
            logger.info(f"✅ Saved resampled full dataset to {self.config.output_path}")

            # Save X and y separately
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled_df = pd.DataFrame(y_resampled, columns=['target'])

            # Save processed X and y
            X_resampled_df.to_csv('../data/processed/X_processed.csv', index=False)
            y_resampled_df.to_csv('../data/processed/y_processed.csv', index=False)


            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            train_df = pd.DataFrame(X_train, columns=X.columns)
            train_df[target_column] = y_train
            train_df.to_csv(self.config.train_path, index=False)

            test_df = pd.DataFrame(X_test, columns=X.columns)
            test_df[target_column] = y_test
            test_df.to_csv(self.config.test_path, index=False)

            logger.info(f"✅ Saved Train Data to {self.config.train_path}")
            logger.info(f"✅ Saved Test Data to {self.config.test_path}")
            logger.info("✅ Feature Engineering Completed Successfully!")

            return self.config.train_path, self.config.test_path

        except Exception as e:
            raise EasyLaborPredictionException(str(e), sys.exc_info())

# 🚀 Run if executed directly
if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.run()
