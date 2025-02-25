import pandas as pd
import numpy as np
from utils import score, compute_costs, calculate_custom_loss
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Ensure only submissions directory exists
os.makedirs('../03. Submissions', exist_ok=True)

def find_optimal_threshold(pipeline, X, y_true, thresholds=np.linspace(0.01, 0.99, 99)):
    """
    Find optimal threshold for classification based on custom loss function
    """
    # Get probabilities
    y_proba = pipeline.predict_proba(X)[:, 1]  # Probability of "Risk"

    # Prepare true values DataFrame
    y_true_df = pd.DataFrame(y_true, columns=['Risk'])
    y_true_df['LoanAmount'] = X['LoanAmount']
    y_true_df.reset_index(drop=True, inplace=True)

    # Test different thresholds
    losses = []
    for thresh in thresholds:
        # Convert probabilities to predictions using threshold
        y_pred = (y_proba > thresh).astype(int)
        y_pred = pd.DataFrame({'Risk': np.where(y_pred == 1, 'Risk', 'No Risk')})

        # Calculate loss for this threshold
        current_loss = score(y_true_df, y_pred, "LoanAmount")
        losses.append(current_loss)

    # Find best threshold
    best_threshold = thresholds[np.argmin(losses)]
    best_loss = min(losses)

    return best_threshold, best_loss, losses

def run_optimized_rf(df_train, df_test, multiplier=7, use_feature_engineering=False):
    """
    Run Random Forest with class weight optimization and optimal threshold

    Parameters:
    -----------
    df_train : DataFrame
        Training data
    df_test : DataFrame
        Test data
    multiplier : int
        Multiplier for Risk class weight
    use_feature_engineering : bool
        Whether to use enhanced features from engineer_features()
    """
    print(f"Running optimized Random Forest (multiplier={multiplier})")

    # Engineer features if requested
    if use_feature_engineering:
        try:
            from enhanced_feature import engineer_features
            X_train_eng = engineer_features(df_train)
            X_test_eng = engineer_features(df_test)
            print("Using enhanced features")
        except (ImportError, ModuleNotFoundError):
            X_train_eng = df_train.copy()
            X_test_eng = df_test.copy()
            print("Enhanced features not found, using original data")
    else:
        X_train_eng = df_train.copy()
        X_test_eng = df_test.copy()
        print("Using original features only")

    # Prepare data
    X = X_train_eng.drop(columns=['Risk'])
    y = X_train_eng['Risk']

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Define real vs training class proportions for weighting
    real_prop = {'Risk': .02, 'No Risk': .98}
    train_prop = {'Risk': 1/3, 'No Risk': 2/3}

    # Calculate weights
    rf_class_weights = {
        'Risk': (real_prop['Risk'] / train_prop['Risk']) * multiplier,
        'No Risk': real_prop['No Risk'] / train_prop['No Risk']
    }

    print(f"Class weights: {rf_class_weights}")

    # Define numerical and categorical features
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=[object]).columns.tolist()

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Create RandomForest classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=5,
        class_weight=rf_class_weights,
        random_state=42
    )

    # Create and fit pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])

    pipeline.fit(X_train, y_train)

    # Find optimal threshold
    best_threshold, val_loss, _ = find_optimal_threshold(pipeline, X_val, y_val)
    print(f"Best threshold: {best_threshold:.3f}, Validation loss: {val_loss:.3f}")

    # Train final model on all data
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_classifier)
    ])

    final_pipeline.fit(X, y)

    # Make predictions with optimal threshold
    test_proba = final_pipeline.predict_proba(X_test_eng)[:, 1]
    test_preds = (test_proba > best_threshold).astype(int)

    # Create submission
    submission = pd.DataFrame({
        'Risk': np.where(test_preds == 1, 'Risk', 'No Risk')
    })

    if 'Id' in X_test_eng.columns:
        submission['Id'] = X_test_eng['Id']
        submission = submission[['Id', 'Risk']]

    # Generate filename with info
    fe_str = "_fe" if use_feature_engineering else ""

    # Save submission only
    submission_path = f'../03. Submissions/rf_submission_m{multiplier}_t{best_threshold:.3f}{fe_str}.csv'
    submission.to_csv(submission_path, index=False)

    print(f"Submission saved to: {submission_path}")

    return {
        'model': final_pipeline,
        'best_threshold': best_threshold,
        'validation_loss': val_loss,
        'submission': submission,
        'submission_path': submission_path
    }

def test_multiple_multipliers(df_train, df_test, multipliers=[5, 6, 7, 8, 9], use_enhanced_features=False):
    """Test multiple multiplier values and find the best one"""
    results = []

    for multiplier in multipliers:
        print(f"\n=== Testing multiplier = {multiplier} ===")
        result = run_optimized_rf(df_train, df_test, multiplier=multiplier, use_feature_engineering=use_enhanced_features)
        results.append({
            'multiplier': multiplier,
            'threshold': result['best_threshold'],
            'validation_loss': result['validation_loss'],
            'submission_path': result['submission_path']
        })

    # Find best multiplier
    best_result = min(results, key=lambda x: x['validation_loss'])

    print("\n=== Results ===")
    for result in results:
        print(f"Multiplier: {result['multiplier']}, Threshold: {result['threshold']:.3f}, "
              f"Loss: {result['validation_loss']:.3f}")

    print(f"\nBest multiplier: {best_result['multiplier']}")
    print(f"Best threshold: {best_result['threshold']:.3f}")
    print(f"Best validation loss: {best_result['validation_loss']:.3f}")
    print(f"Submission: {best_result['submission_path']}")

    return results

# Example usage
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df_train = pd.read_csv("../01.Data/german_credit_train.csv")
    df_test = pd.read_csv("../01.Data/german_credit_test.csv")

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Run with best multiplier (7)
    results = run_optimized_rf(df_train, df_test, multiplier=7,
                              use_feature_engineering=False)

    # Uncomment to test multiple multipliers
    # results = test_multiple_multipliers(df_train, df_test,
    #                                    multipliers=[5, 6, 7, 8, 9],
    #                                    use_enhanced_features=False)