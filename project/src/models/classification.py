"""
Module for classification models to identify high-risk areas.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os


class RiskClassifier:
    """Class for classifying counties into risk categories."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize the RiskClassifier.
        
        Args:
            threshold: HIV prevalence threshold for high-risk classification
        """
        self.threshold = threshold
        self.model = None
        self.feature_names = []
        self.scaler = None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'hiv_prevalence') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for classification.
        
        Args:
            df: DataFrame containing features and target
            target_col: Column to use as target for classification
            
        Returns:
            Tuple of (X DataFrame with features, y Series with binary target)
        """
        if df.empty or target_col not in df.columns:
            return pd.DataFrame(), pd.Series()
            
        # Create binary target based on threshold
        y = (df[target_col] > self.threshold).astype(int)
        
        # Select potential features
        # Exclude target, county, year, and any derivatives of the target
        exclude_patterns = [target_col, 'county', 'year', 'ds', 'y_']
        feature_cols = [col for col in df.columns 
                       if not any(pattern in col for pattern in exclude_patterns)]
        
        # Handle categorical variables
        cat_cols = [col for col in feature_cols if df[col].dtype == 'object']
        
        # Simple handling: drop categorical columns for now
        # In a full implementation, you'd use one-hot encoding or other methods
        X = df[feature_cols].drop(columns=cat_cols)
        
        # Store feature names for later
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train a random forest classifier to identify high-risk areas.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Trained RandomForestClassifier
        """
        if X.empty or y.empty:
            return None
            
        # Create a pipeline with preprocessing and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define hyperparameters to tune
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # Set up grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X, y)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Store the scaler separately for prediction
        self.scaler = best_model.named_steps['scaler']
        
        # Store the classifier
        self.model = best_model.named_steps['classifier']
        
        return self.model
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
        """
        Evaluate the trained model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None or X.empty or y.empty:
            return {}
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        class_report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        roc_auc = roc_auc_score(y, y_prob)
        
        # Get feature importances
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'roc_auc': roc_auc,
            'feature_importance': feature_imp
        }
    
    def predict_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk categories for counties.
        
        Args:
            df: DataFrame containing features
            
        Returns:
            DataFrame with original data plus risk predictions
        """
        if self.model is None or df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Prepare features for prediction
        X, _ = self.prepare_data(result_df)
        
        if X.empty:
            return result_df
            
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        result_df['risk_category'] = self.model.predict(X_scaled)
        result_df['risk_probability'] = self.model.predict_proba(X_scaled)[:, 1]
        
        # Add risk level labels
        risk_labels = {0: 'Low Risk', 1: 'High Risk'}
        result_df['risk_level'] = result_df['risk_category'].map(risk_labels)
        
        return result_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model, scaler, and feature names
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'threshold': self.threshold
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RiskClassifier':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Initialized RiskClassifier with loaded model
        """
        # Load the model data
        model_data = joblib.load(filepath)
        
        # Create a new classifier instance
        classifier = cls(threshold=model_data['threshold'])
        
        # Set the model attributes
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.feature_names = model_data['feature_names']
        
        return classifier


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import os
    from src.data.data_loader import DataLoader
    from src.data.data_cleaner import DataCleaner
    from src.features.feature_engineering import FeatureEngineer
    
    # Load and prepare data
    loader = DataLoader()
    cleaner = DataCleaner()
    engineer = FeatureEngineer()
    
    raw_datasets = loader.load_all_datasets()
    cleaned_datasets = cleaner.clean_all_datasets(raw_datasets)
    
    # Get the prevalence dataset and add features
    if 'prevalence' in cleaned_datasets and not cleaned_datasets['prevalence'].empty:
        prevalence_df = cleaned_datasets['prevalence']
        processed_df = engineer.process_features(prevalence_df)
        
        # Initialize the classifier
        classifier = RiskClassifier(threshold=0.05)  # Threshold for high HIV prevalence
        
        # Prepare the data
        X, y = classifier.prepare_data(processed_df)
        
        if not X.empty and not y.empty:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train the model
            classifier.train_model(X_train, y_train)
            
            # Evaluate the model
            metrics = classifier.evaluate_model(X_test, y_test)
            
            # Print evaluation results
            if metrics:
                print("Model Evaluation:")
                print(f"ROC AUC: {metrics['roc_auc']:.4f}")
                print("\nClassification Report:")
                for label, values in metrics['classification_report'].items():
                    if isinstance(values, dict):
                        print(f"  {label}: f1-score={values['f1-score']:.4f}, precision={values['precision']:.4f}, recall={values['recall']:.4f}")
                
                print("\nTop 5 Important Features:")
                print(metrics['feature_importance'].head(5))
            
            # Save the model
            classifier.save_model("data/models/risk_classifier.joblib")
            
            # Make predictions on the full dataset
            risk_predictions = classifier.predict_risk(processed_df)
            
            # Save predictions
            output_path = "data/processed/prevalence_with_risk.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            risk_predictions.to_csv(output_path, index=False)
            print(f"Saved risk predictions to {output_path}")