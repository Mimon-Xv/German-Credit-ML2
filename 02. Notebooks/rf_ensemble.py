import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import time
from utils import score, compute_costs, calculate_custom_loss

# Ensure submissions directory exists
os.makedirs('../03. Submissions', exist_ok=True)

def find_optimal_threshold(probas, y_true, X_val, thresholds=np.linspace(0.01, 0.99, 99)):
    """
    Find optimal threshold for classification based on custom loss function
    """
    # Prepare true values DataFrame
    y_true_df = pd.DataFrame(y_true, columns=['Risk'])
    y_true_df['LoanAmount'] = X_val['LoanAmount']
    y_true_df.reset_index(drop=True, inplace=True)

    # Test different thresholds
    losses = []
    for thresh in thresholds:
        # Convert probabilities to predictions using threshold
        y_pred = (probas > thresh).astype(int)
        y_pred = pd.DataFrame({'Risk': np.where(y_pred == 1, 'Risk', 'No Risk')})

        # Calculate loss for this threshold
        current_loss = score(y_true_df, y_pred, 'LoanAmount')
        losses.append(current_loss)

    # Find best threshold
    best_threshold = thresholds[np.argmin(losses)]
    best_loss = min(losses)

    return best_threshold, best_loss, losses

def create_rf_ensemble(df_train, df_test, n_models=10, multiplier=7):
    """
    Create an ensemble of Random Forest models with different random seeds

    Parameters:
    -----------
    df_train : DataFrame
        Training data
    df_test : DataFrame
        Test data
    n_models : int
        Number of models in the ensemble
    multiplier : int
        Multiplier for Risk class weight

    """
    start_time = time.time()
    print(f"Creating RF ensemble with {n_models} models (multiplier={multiplier})")

    # Prepare data
    X_train_eng = df_train.copy()
    X_test_eng = df_test.copy()
    X = X_train_eng.drop(columns=['Risk'])
    y = X_train_eng['Risk']

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define class weights
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}

    rf_class_weights = {
        'Risk': (real_prop['Risk'] / train_prop['Risk']) * multiplier,
        'No Risk': real_prop['No Risk'] / train_prop['No Risk']
    }

    print(f"Class weights: {rf_class_weights}")

    # Define features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Train multiple models with different random seeds
    models = []
    val_probas = []
    test_probas = []

    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...")

        # Create RandomForest classifier with different random seed
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=5,
            class_weight=rf_class_weights,
            random_state=42 + i  # Different seed for each model
        )

        # Create and fit pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', rf_classifier)
        ])

        pipeline.fit(X_train, y_train)
        models.append(pipeline)

        # Get probabilities for validation set
        val_proba = pipeline.predict_proba(X_val)[:, 1]
        val_probas.append(val_proba)

        # Get probabilities for test set
        test_proba = pipeline.predict_proba(X_test_eng)[:, 1]
        test_probas.append(test_proba)

    # Average probabilities
    val_proba_avg = np.mean(val_probas, axis=0)
    test_proba_avg = np.mean(test_probas, axis=0)

    # Find optimal threshold using averaged validation probabilities
    best_threshold, val_loss, _ = find_optimal_threshold(val_proba_avg, y_val, X_val)
    print(f"Best threshold: {best_threshold:.3f}, Validation loss: {val_loss:.3f}")

    # Create final prediction from averaged test probabilities
    test_preds = (test_proba_avg > best_threshold).astype(int)

    # Create submission
    submission = pd.DataFrame({
        'Risk': np.where(test_preds == 1, 'Risk', 'No Risk')
    })

    if 'Id' in X_test_eng.columns:
        submission['Id'] = X_test_eng['Id']
        submission = submission[['Id', 'Risk']]

    # Save submission
    submission_path = f'../03. Submissions/rf_ensemble_n{n_models}_m{multiplier}_t{best_threshold:.3f}.csv'
    submission.to_csv(submission_path, index=False)

    elapsed_time = time.time() - start_time
    print(f"Ensemble training completed in {elapsed_time:.1f} seconds")
    print(f"Submission saved to: {submission_path}")

    return {
        'models': models,
        'best_threshold': best_threshold,
        'validation_loss': val_loss,
        'submission': submission,
        'submission_path': submission_path,
        'elapsed_time': elapsed_time
    }

# Example usage
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df_train = pd.read_csv("../01.Data/german_credit_train.csv")
    df_test = pd.read_csv("../01.Data/german_credit_test.csv")

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Create ensemble with 10 models and multiplier=7
    results = create_rf_ensemble(
        df_train, df_test,
        n_models=10,  # Number of models in the ensemble
        multiplier=7,  # Your best multiplier
        use_feature_engineering=False  # Whether to use enhanced features
    )

    # Uncomment to search for optimal parameters
    # results = search_ensemble_params(
    #     df_train, df_test,
    #     n_models_list=[5, 10, 15],  # Different ensemble sizes to try
    #     multipliers=[6.5, 7, 7.5],  # Different multipliers to try
    #     use_feature_engineering=False
    # )