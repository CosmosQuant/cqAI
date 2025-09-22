import pandas as pd
import numpy as np

# Load the original data
print("Loading original data...")
data_handler = DataHandler({'ticker': 'btc', 'data_folder': 'data/btc_csv', 'plot': False})
original_df = data_handler.load_data(saving_space=False)

print(f"Original shape: {original_df.shape}")
print(f"Original columns: {original_df.columns.tolist()}")
print(f"Original index type: {type(original_df.index)}")
print(f"Original index sample: {original_df.index[:5]}")

# Load the CSV file
print("\nLoading CSV file...")
csv_df = pd.read_csv('crypto_df.csv.gz', compression='gzip', index_col=0, parse_dates=True)

print(f"CSV shape: {csv_df.shape}")
print(f"CSV columns: {csv_df.columns.tolist()}")
print(f"CSV index type: {type(csv_df.index)}")
print(f"CSV index sample: {csv_df.index[:5]}")

# Check if there's a Date column that should be the index
if 'Date' in csv_df.columns:
    print("\nFound 'Date' column! Let's use it as index...")
    
    # Convert Date column to datetime and set as index
    csv_df['Date'] = pd.to_datetime(csv_df['Date'])
    csv_df_fixed = csv_df.set_index('Date')
    
    print(f"Fixed CSV shape: {csv_df_fixed.shape}")
    print(f"Fixed CSV index type: {type(csv_df_fixed.index)}")
    print(f"Fixed CSV index sample: {csv_df_fixed.index[:5]}")
    
    # Compare with original
    print(f"\nDataframes equal: {csv_df_fixed.equals(original_df)}")
    
    if not csv_df_fixed.equals(original_df):
        print("\nDifferences found:")
        print(f"Shape difference: {original_df.shape} vs {csv_df_fixed.shape}")
        print(f"Column difference: {set(original_df.columns) - set(csv_df_fixed.columns)}")
        
        # Check first few values
        print("\nFirst few values comparison:")
        print("Original:")
        print(original_df.head())
        print("\nCSV:")
        print(csv_df_fixed.head())
else:
    print("\nNo 'Date' column found in CSV")

# Save the correct way
print("\nSaving with correct format...")
original_df.to_csv('crypto_df_fixed.csv.gz', compression='gzip', index=True, date_format='%Y-%m-%d %H:%M:%S')

# Test loading the fixed file
print("\nTesting fixed file...")
test_df = pd.read_csv('crypto_df_fixed.csv.gz', compression='gzip', index_col=0, parse_dates=True)
print(f"Fixed file equals original: {test_df.equals(original_df)}") 