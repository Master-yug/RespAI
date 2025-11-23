import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CoughClassifier:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
    
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the CSV data"""
        # Load data
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Display class distribution
        print("\nClass distribution:")
        print(df['status'].value_counts())
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variable"""
        # Select numerical features (excluding metadata columns)
        feature_columns = [
            'zcr', 'rms', 'centroid', 'bw', 'rolloff', 'contrast', 'tonnetz', 'chroma',
            'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 
            'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'duration'
        ]
        
        # Additional features that might be useful
        additional_features = ['cough_detected', 'age']
        
        # Combine features
        all_features = feature_columns + [col for col in additional_features if col in df.columns]
        
        # Extract features and target
        X = df[all_features].copy()
        y = df['status']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Encode categorical variables if any in features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = self.label_encoder.fit_transform(X[col])
        
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features shape: {X.shape}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X, y_encoded, all_features
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate multiple models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'SVM':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = None  # SVM doesn't have predict_proba by default
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*50)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*50)
        
        # Create comparison table
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[name]['accuracy'] for name in self.results],
            'Precision': [self.results[name]['precision'] for name in self.results],
            'Recall': [self.results[name]['recall'] for name in self.results],
            'F1-Score': [self.results[name]['f1_score'] for name in self.results],
            'CV Accuracy': [self.results[name]['cv_mean'] for name in self.results]
        })
        
        print("\nModel Comparison:")
        print(results_df.round(4))
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_model = self.results[best_model_name]
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {best_model['f1_score']:.4f})")
        
        return best_model_name, best_model
    
    def plot_confusion_matrix(self, X_test, y_test, model_name):
        """Plot confusion matrix for a specific model"""
        model_info = self.results[model_name]
        y_pred = model_info['predictions']
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' in self.results:
            rf_model = self.results['Random Forest']['model']
            
            importance = rf_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=feature_imp_df.head(15), x='importance', y='feature')
            plt.title('Top 15 Feature Importance - Random Forest')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return feature_imp_df
    
    def generate_detailed_report(self, X_test, y_test, best_model_name):
        """Generate detailed classification report"""
        model_info = self.results[best_model_name]
        y_pred = model_info['predictions']
        
        print("\n" + "="*50)
        print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
        print("="*50)
        
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoder.classes_))
        
        # Additional metrics
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision (Weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"Recall (Weighted): {recall_score(y_test, y_pred, average='weighted'):.4f}")
        print(f"F1-Score (Weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")

def main():
    # Initialize classifier
    classifier = CoughClassifier()
    
    # Load your data (replace with your actual file path)
    file_path = "NO_BLANKS.csv"  # Update this path
    
    try:
        # Load and preprocess data
        df = classifier.load_and_preprocess_data(file_path)
        
        # Prepare features
        X, y, feature_names = classifier.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Train models
        classifier.train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        best_model_name, best_model = classifier.evaluate_models(X_test, y_test)
        
        # Plot confusion matrix for best model
        classifier.plot_confusion_matrix(X_test, y_test, best_model_name)
        
        # Plot feature importance
        feature_importance_df = classifier.plot_feature_importance(feature_names)
        
        # Generate detailed report
        classifier.generate_detailed_report(X_test, y_test, best_model_name)
        
        # Print top features
        if feature_importance_df is not None:
            print("\nTop 10 Most Important Features:")
            print(feature_importance_df.head(10).round(4))
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please update the file_path variable with the correct path to your CSV file.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()