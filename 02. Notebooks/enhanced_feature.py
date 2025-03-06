import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Create enhanced features for German Credit prediction

    Parameters:
    -----------
    df: pandas DataFrame
        The dataframe containing original features

    Returns:
    --------
    df_new: pandas DataFrame
        The dataframe with additional engineered features
    """
    df_new = df.copy()

    # 1. Basic ratio features

    # Loan amount to duration ratio (monthly payment approximation)
    df_new['LoanAmount_per_Duration'] = df['LoanAmount'] / df['LoanDuration']

    # Age-related risk factors
    df_new['Age_to_LoanDuration'] = df['Age'] / df['LoanDuration']

    # Installment percentage to loan amount ratio
    df_new['InstallmentPercent_to_LoanAmount'] = df['InstallmentPercent'] / (df['LoanAmount'] / 1000)

    # 2. Ordinal mappings for categorical features

    # Checking account status risk levels
    checking_risk = {
        'no_checking': 3,  # Highest risk - no checking account
        'less_0': 2,       # Negative balance
        '0_to_200': 1,     # Small positive balance
        'greater_200': 0   # Lowest risk - large positive balance
    }
    df_new['CheckingRisk'] = df['CheckingStatus'].map(checking_risk)

    # Credit history risk levels
    credit_history_risk = {
        'delay_in_paying': 4,          # Highest risk
        'critical_account': 3,
        'existing_credits_paid': 2,
        'prior_payments_delayed': 1,
        'credits_paid_to_date': 0      # Lowest risk
    }

    # Handle any missing mappings for CreditHistory
    if 'CreditHistory' in df.columns:
        df_new['CreditHistoryRisk'] = df['CreditHistory'].map(
            lambda x: credit_history_risk.get(x, 2)  # Default to middle value if missing
        )

    # Savings level - ordinal encoding
    savings_map = {
        'unknown': 0,        # Unknown is risky
        'less_100': 1,       # Very low savings
        '100_to_500': 2,     # Low savings
        '500_to_1000': 3,    # Medium savings
        'greater_1000': 4    # High savings (lowest risk)
    }
    df_new['SavingsLevel'] = df['ExistingSavings'].map(savings_map)

    # Employment duration - ordinal encoding
    employment_map = {
        'unemployed': 0,    # Highest risk
        'less_1': 1,        # Less than 1 year
        '1_to_4': 2,        # 1 to 4 years
        '4_to_7': 3,        # 4 to 7 years
        'greater_7': 4      # More than 7 years (lowest risk)
    }
    df_new['EmploymentLevel'] = df['EmploymentDuration'].map(employment_map)

    # 3. Interaction features - combining risk factors

    # Checking status and credit history interaction
    df_new['Checking_x_Credit'] = df_new['CheckingRisk'] * df_new['CreditHistoryRisk']

    # Checking status and savings interaction
    df_new['Checking_x_Savings'] = df_new['CheckingRisk'] * df_new['SavingsLevel']

    # Employment level and loan duration
    df_new['Employment_x_LoanDuration'] = df_new['EmploymentLevel'] * df['LoanDuration'] / 12  # Normalized to years

    # Age and savings interaction
    df_new['Age_x_Savings'] = df['Age'] * df_new['SavingsLevel']

    # 4. Binary flags for high-risk combinations

    # High-risk flag: No checking & low savings & high loan amount
    df_new['HighRiskCombination'] = (
        (df['CheckingStatus'] == 'no_checking') &
        (df['ExistingSavings'].isin(['unknown', 'less_100'])) &
        (df['LoanAmount'] > df['LoanAmount'].median())
    ).astype(int)

    # Low installment percentage relative to loan size flag
    df_new['LowInstallmentRatio'] = (
        df['InstallmentPercent'] < 3 &
        (df['LoanAmount'] > df['LoanAmount'].median())
    ).astype(int)

    # 5. Loan purpose risk - some purposes are higher risk
    purpose_risk = {
        'car_new': 1,
        'car_used': 2,
        'furniture': 2,
        'radio_tv': 2,
        'domestic_appliance': 2,
        'repairs': 3,
        'education': 3,
        'vacation': 4,
        'retraining': 3,
        'business': 4,
        'other': 4
    }

    df_new['PurposeRisk'] = df['LoanPurpose'].map(purpose_risk)

    # 6. Housing status numeric encoding
    housing_map = {
        'own': 0,     # Lowest risk - owns home
        'rent': 1,    # Medium risk - renting
        'free': 2     # Highest risk - free housing (may indicate financial dependence)
    }
    df_new['HousingRisk'] = df['Housing'].map(housing_map)

    return df_new