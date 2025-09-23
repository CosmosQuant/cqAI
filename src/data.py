# src/data.py - Data loading and preprocessing
import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

# Timestamp column name constants
TIMESTAMP_COLUMNS = ['datetime', 'time', 'timestamp', 'date', 'Open Time']

# Standard column format
STANDARD_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']

def read_xts(path: str, show_info: bool = True) -> pd.DataFrame:
    """
    Read a single time series file.
    """
    try:
        # Auto-detect file format and read
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.parquet'):
            df = pd.read_parquet(path)
        elif path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(path)
        else:
            # Default to CSV
            df = pd.read_csv(path)
        
        if show_info: print(f"Successfully read {len(df)} rows from {path}")
            
        return df
        
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return pd.DataFrame()

def combine_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple DataFrames with same columns into single DataFrame.
    
    Args:
        dfs: List of DataFrames to combine
        
    Returns:
        pd.DataFrame: Combined DataFrame
    """
    if not dfs:
        return pd.DataFrame()
    
    # Filter out empty DataFrames
    valid_dfs = [df for df in dfs if not df.empty]
    
    if not valid_dfs:
        print("No valid DataFrames to combine")
        return pd.DataFrame()
    
    
    try:
        # Find timestamp column from first DataFrame (avoid searching in large combined data)
        timestamp_col = None
        for col in TIMESTAMP_COLUMNS:
            if col in valid_dfs[0].columns:
                timestamp_col = col
                break
        
        combined_df = pd.concat(valid_dfs, ignore_index=True, sort=False, copy=False)
        original_rows = len(combined_df)
        
        # deduplicate and sort by timestamp if available
        if timestamp_col:
            combined_df = combined_df.drop_duplicates(subset=[timestamp_col], keep='last')
            combined_df = combined_df.sort_values(timestamp_col).reset_index(drop=True)
        else:
            combined_df = combined_df.drop_duplicates().reset_index(drop=True)
            print(f"No timestamp column as one of {TIMESTAMP_COLUMNS} found")
        
        print(f"Combined {len(valid_dfs)} DataFrames into {len(combined_df)} rows, {original_rows - len(combined_df)} rows are dropped")
        return combined_df
        
    except Exception as e:
        print(f"Error combining DataFrames: {e}")
        return pd.DataFrame()

def read_folder_data(
    folder_path: str, file_extension: str = '.csv', market_keyword: str = 'btc',
    exact_match: bool = True, show_info: bool = True
) -> pd.DataFrame:
    """
    Read all matching files in a folder and return combined DataFrame.    
    """
    try:
        # Convert to Path object
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Folder does not exist: {folder_path}")
            return pd.DataFrame()
        
        # Find all files with specified extension
        pattern = f"*{file_extension}"
        all_files = list(folder.glob(pattern))
        
        # Filter files by market keyword
        matching_files = []
        for file_path in all_files:
            filename = file_path.stem.lower()  # Get filename without extension
            keyword_lower = market_keyword.lower()
            
            if exact_match:
                # Check if keyword appears as separate word (e.g., 'btc' but not 'btcusdt')
                if keyword_lower in filename:
                    # Simple check: keyword followed by non-letter or at end
                    idx = filename.find(keyword_lower)
                    if idx != -1:
                        after_keyword = idx + len(keyword_lower)
                        if (after_keyword >= len(filename) or 
                            not filename[after_keyword].isalpha()):
                            matching_files.append(file_path)
            else:
                if keyword_lower in filename:
                    matching_files.append(file_path)
        
        if not matching_files:
            print(f"No files found matching pattern '*{market_keyword}*{file_extension}' in {folder_path}")
            return pd.DataFrame()
        
        print(f"Found {len(matching_files)} matching files")
        
        # Read all matching files (R lapply style)
        dataframes = [read_xts(str(file_path), show_info=show_info) for file_path in sorted(matching_files)]
        # Filter out empty DataFrames
        dataframes = [df for df in dataframes if not df.empty]
        
        # Combine all DataFrames
        return combine_dataframes(dataframes)
        
    except Exception as e:
        print(f"Error reading folder data: {e}")
        return pd.DataFrame()


# OOP Design - Data Source Adapters
class DataSource(ABC):
    """Abstract base class for data sources."""
    
    # Standard column format for all data sources
    STANDARD_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'oi']
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def load_data(
        self,
        folder_path: str,
        file_extension: str = '.csv',
        market_keyword: str = 'btc',
        show_info: bool = True
    ) -> pd.DataFrame:
        """Load and transform data using this data source."""
        # Load raw data
        raw_df = read_folder_data(folder_path, file_extension, market_keyword, True, show_info)
        
        # Transform using this source's configuration
        return self.to_standard(raw_df)
    
    def to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data to standardized format."""
        column_mapping = self.config.get('column_mapping', {})
        
        # Apply column mapping and standardization
        mapped_df = self._apply_column_mapping(df, column_mapping)
        
        # Convert datetime to UTC if datetime column exists
        if 'datetime' in mapped_df.columns:
            mapped_df = self._convert_datetime_to_utc(mapped_df)
        
        return mapped_df
    
    def _apply_column_mapping(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Apply column mapping and ensure standard format."""
        # Rename columns according to mapping
        mapped_df = df.rename(columns=column_mapping)
        
        # Keep only mapped columns (drop unmapped ones)
        mapped_columns = list(column_mapping.values())
        mapped_df = mapped_df[mapped_columns]
        
        # Add missing standard columns with value 0
        for col in self.STANDARD_COLUMNS:
            if col not in mapped_df.columns:
                mapped_df[col] = 0
        
        # Reorder columns to standard format
        return mapped_df[self.STANDARD_COLUMNS]
    
    def _convert_datetime_to_utc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime column to UTC."""
        timestamp_unit = self.config.get('timestamp_unit', 'ms')
        
        # Convert to datetime if not already
        if df['datetime'].dtype == 'object' or 'int' in str(df['datetime'].dtype):
            if timestamp_unit == 'ms':
                df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
            elif timestamp_unit == 's':
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s', utc=True)
            else:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        
        # Ensure it's UTC timezone
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        elif df['datetime'].dt.tz != pd.Timestamp.now(tz='UTC').tz:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        
        return df


class BinanceDataSource(DataSource):
    """Binance data source with specific transformations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default Binance configuration
        default_config = {
            'timestamp_column': 'Open Time',
            'timestamp_unit': 'ms',
            'column_mapping': {
                'Open Time': 'datetime',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
                # 'oi' is missing, will be auto-added as 0
            }
        }
        
        # Merge with user config
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    # Inherit to_standard from parent class - no override needed


class IBDataSource(DataSource):
    """Interactive Brokers data source with specific transformations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default IB configuration (placeholder - to be implemented when data is available)
        default_config = {
            'timestamp_column': 'datetime',  # Placeholder
            'timestamp_unit': 's',  # Placeholder
            'column_mapping': {
                'datetime': 'datetime',  # Placeholder
                'open': 'open',  # Placeholder
                'high': 'high',  # Placeholder
                'low': 'low',  # Placeholder
                'close': 'close',  # Placeholder
                'volume': 'volume',  # Placeholder
                'oi': 'oi'  # Placeholder
            }
        }
        
        # Merge with user config
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    # TODO: Implement IB-specific data transformations when data is available
    # def to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """IB-specific data transformation logic will be implemented here."""
    #     pass


if __name__ == "__main__":
    # # test case of combine_dataframes
    # df1 = read_xts('data/btc_csv/btc_20250120.csv')
    # df2 = read_xts('data/btc_csv/btc_20250121.csv')
    # combine_dataframes([df1, df2])

    # # test case of read_folder_data
    # df = read_folder_data('data/btc_csv/', market_keyword='btc', show_info=False)
    
    # Test case with direct DataSource usage (concise syntax)
    ds_binance = BinanceDataSource() # ib_source = IBDataSource()
    df = ds_binance.load_data(folder_path='data/btc_csv/', market_keyword='btc', show_info=False)
    print(df.head(), df.tail())
    
