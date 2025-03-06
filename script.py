import pandas as pd
import numpy as np
import sklearn

training_df = pd.read_csv("train.csv", low_memory=False)
testing_df = pd.read_csv("test.csv", low_memory=False)

def clean_airbnb_data(df):
    
    # 1. Remove columns with > 25% missing values
    # Calculate fraction of NON-missing values in each column
    non_missing_fraction = df.notna().mean()
    # Keep columns that have at least 75% non-null data
    cols_to_keep = non_missing_fraction[non_missing_fraction >= 0.75].index
    df = df[cols_to_keep]
    
    # 2. Remove '$' from 'price' and 'extra_people', convert to float
    for col in ['price', 'extra_people']:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                                .str.replace('$', '', regex=False)
                                .str.replace(',', '', regex=False)
                                .astype(float, errors='ignore'))
    
    # 3. Convert ALL amenities into binary columns
    if 'amenities' in df.columns:
        # a) Clean up amenities string
        df['amenities_clean'] = (df['amenities']
                                 .astype(str)
                                 .str.replace(r'[\{\}"\']', '', regex=True)
                                 .str.lower())
        
        # b) Split into a list of amenities
        df['amenities_list'] = df['amenities_clean'].apply(
            lambda x: [amen.strip() for amen in x.split(',')]
        )
        
        # c) Explode + get_dummies
        df_exploded = df.explode('amenities_list')
        amenities_dummies = pd.get_dummies(df_exploded['amenities_list'], prefix='amen')
        amenities_binary = amenities_dummies.groupby(level=0).sum()
        
        # d) Merge back
        df = pd.concat([df, amenities_binary], axis=1)
        
        # Optionally drop the old columns
        df.drop(columns=['amenities', 'amenities_clean', 'amenities_list'], inplace=True)
    
    # 4. Convert date columns to datetime
    date_cols = ['host_since', 'first_review', 'last_review']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # 5. Remove outliers (e.g., price > 2000)
    if 'price' in df.columns:
        df = df[df['price'] <= 2000]  # Adjust threshold as needed
    
    # 6. Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # 7. Fill missing categorical values with 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna('Unknown', inplace=True)
    
    # 8. Convert 'host_is_superhost' to 0/1
    if 'host_is_superhost' in df.columns:
        df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0, np.nan: 0})
    
    # 9. Print the first few rows of the updated DataFrame
    print(df.head())
    
    return df

cleaned_df = clean_airbnb_data(training_df)  # This will also print the first few rows