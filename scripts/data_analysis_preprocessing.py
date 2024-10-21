import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def clean_data(df_creditcard, df_fraud, df_ipaddress):
    # Remove duplicates
    df_creditcard.drop_duplicates(inplace=True)
    df_fraud.drop_duplicates(inplace=True)
    df_ipaddress.drop_duplicates(inplace=True)

    # Correct data types for credit card dataset
    df_creditcard['Time'] = df_creditcard['Time'].astype('float')
    df_creditcard['Amount'] = df_creditcard['Amount'].astype('float')
    df_creditcard['Class'] = df_creditcard['Class'].astype('int')

    # Correct data types for fraud dataset
    df_fraud['signup_time'] = pd.to_datetime(df_fraud['signup_time'])
    df_fraud['purchase_time'] = pd.to_datetime(df_fraud['purchase_time'])
    df_fraud['age'] = df_fraud['age'].astype('int')

    # Correct data types for IP address dataset
    df_ipaddress['lower_bound_ip_address'] = df_ipaddress['lower_bound_ip_address'].astype('int')
    df_ipaddress['upper_bound_ip_address'] = df_ipaddress['upper_bound_ip_address'].astype('int')

    # Verify the data types after conversion
    print("Credit Card Data Types:\n", df_creditcard.dtypes)
    print("Fraud Data Types:\n", df_fraud.dtypes)

def perform_eda(df_creditcard, df_fraud):
    # Univariate Analysis
    def univariate_analysis():
        print("Univariate Analysis:")
        
        # Credit Card Dataset - Distribution of Amount
        plt.figure(figsize=(10, 5))
        sns.histplot(df_creditcard['Amount'], bins=30, kde=True)
        plt.title('Distribution of Transaction Amount in Credit Card Dataset')
        plt.show()
        
        # Fraud Dataset - Distribution of Age
        plt.figure(figsize=(10, 5))
        sns.histplot(df_fraud['age'], bins=20, kde=True)
        plt.title('Distribution of Age in Fraud Dataset')
        plt.show()

    # Bivariate Analysis
    def bivariate_analysis():
        print("Bivariate Analysis:")
        
        # Credit Card Dataset - Amount vs. Class (Fraud/Non-Fraud)
        plt.figure(figsize=(10, 5))
        sns.boxplot(x='Class', y='Amount', data=df_creditcard)
        plt.title('Transaction Amount by Fraud Class in Credit Card Dataset')
        plt.show()
        
        # Fraud Dataset - Purchase Value vs. Age
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x='age', y='purchase_value', hue='class', data=df_fraud)
        plt.title('Purchase Value by Age and Fraud Class in Fraud Dataset')
        plt.show()

    # Call both analyses
    univariate_analysis()
    bivariate_analysis()

def merge_fraud_data_with_geolocation(fraud_data, ip_address_data):
    # Convert to the necessary types
    fraud_data['ip_address_numeric'] = fraud_data['ip_address'].astype('int32')

    # Set the index for ip_address_data for efficient merging
    ip_address_data.set_index('lower_bound_ip_address', inplace=True)

    # Use merge_asof for efficient range joins
    merged_data = pd.merge_asof(
        fraud_data.sort_values('ip_address_numeric'),
        ip_address_data.sort_index(),
        left_on='ip_address_numeric',
        right_index=True,
        direction='forward',  # Use 'forward' to match to the nearest higher value
        suffixes=('', '_country')
    )
    
    return merged_data




def extract_time_based_features(dataframe):
    """Extract time-based features such as signup duration, hour of purchase, and day of the week."""
    dataframe['signup_duration_days'] = (dataframe['purchase_time'] - dataframe['signup_time']).dt.days
    dataframe['hour_of_purchase'] = dataframe['purchase_time'].dt.hour
    dataframe['day_of_week'] = dataframe['purchase_time'].dt.dayofweek
    return dataframe

def calculate_transaction_frequency(dataframe):
    """Calculate the number of transactions for each user."""
    dataframe['transaction_count'] = dataframe.groupby('user_id')['purchase_time'].transform('count')
    return dataframe

def calculate_transaction_value_features(dataframe):
    """Calculate average and total purchase value per user."""
    dataframe['average_purchase_value'] = dataframe.groupby('user_id')['purchase_value'].transform('mean')
    dataframe['total_purchase_value'] = dataframe.groupby('user_id')['purchase_value'].transform('sum')
    return dataframe

def encode_categorical_features(dataframe):
    """One-hot encode categorical features."""
    return pd.get_dummies(dataframe, columns=['device_id', 'source', 'browser'], drop_first=True)

def encode_sex_and_age(dataframe):
    """Encode sex and normalize age."""
    dataframe['sex_encoded'] = dataframe['sex'].map({'F': 0, 'M': 1})
    dataframe['age_normalized'] = (dataframe['age'] - dataframe['age'].mean()) / dataframe['age'].std()
    return dataframe

def drop_unnecessary_columns(dataframe):
    """Drop columns that are no longer needed."""
    columns_to_drop = ['signup_time', 'purchase_time', 'sex', 'age', 'user_id']
    return dataframe.drop(columns=columns_to_drop, errors='ignore')

def feature_engineering(df):
    """Main function to perform feature engineering."""
    df = extract_time_based_features(df)
    df = calculate_transaction_frequency(df)
    df = calculate_transaction_value_features(df)
    df = encode_categorical_features(df)
    df = encode_sex_and_age(df)
    df = drop_unnecessary_columns(df)
    
    return df


