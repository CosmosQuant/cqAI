# src/cv.py - Cross-validation strategies for time series
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

def build_splitter(cv_config):
    """
    Build cross-validation splitter based on configuration.
    
    Args:
        cv_config (dict): CV configuration with 'type' and parameters
        
    Returns:
        Cross-validation splitter object
    """
    cv_type = cv_config['type']
    
    if cv_type == 'walk_forward':
        return WalkForwardCV(
            train_days=cv_config['train_days'],
            test_days=cv_config['test_days']
        )
    
    elif cv_type == 'purged_cv':
        return PurgedKFoldCV(
            n_splits=cv_config['n_splits'],
            embargo_days=cv_config['embargo_days']
        )
    
    elif cv_type == 'seasonal_cv':
        return SeasonalCV(
            n_splits=cv_config['n_splits'],
            season_length=cv_config['season_length']
        )
    
    else:
        raise ValueError(f"Unknown CV type: {cv_type}")

class WalkForwardCV(BaseCrossValidator):
    """
    Walk-forward cross-validation for time series.
    
    Uses a rolling window approach where the training window moves forward
    with each split, maintaining fixed train and test periods.
    """
    
    def __init__(self, train_days, test_days):
        """
        Initialize walk-forward CV.
        
        Args:
            train_days (int): Number of days for training
            test_days (int): Number of days for testing
        """
        self.train_days = train_days
        self.test_days = test_days
        
    def split(self, df, y=None, groups=None):
        """
        Generate train/test splits.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(df)
        
        # Convert days to approximate number of samples (assuming daily data)
        # For higher frequency data, this would need adjustment
        if hasattr(df.index, 'freq') and df.index.freq:
            # Try to infer frequency
            freq_str = str(df.index.freq)
            if 'D' in freq_str:
                samples_per_day = 1
            elif 'H' in freq_str:
                samples_per_day = 24
            elif 'T' in freq_str or 'min' in freq_str:
                samples_per_day = 24 * 60
            else:
                samples_per_day = 1
        else:
            # Estimate based on index differences
            time_diffs = df.index[1:] - df.index[:-1]
            avg_diff = time_diffs.mean()
            samples_per_day = int(pd.Timedelta(days=1) / avg_diff)
        
        train_samples = self.train_days * samples_per_day
        test_samples = self.test_days * samples_per_day
        
        # Ensure we have enough data
        if train_samples + test_samples > n_samples:
            raise ValueError(f"Not enough data: need {train_samples + test_samples}, have {n_samples}")
        
        # Generate splits
        start_idx = 0
        while start_idx + train_samples + test_samples <= n_samples:
            train_end = start_idx + train_samples
            test_end = train_end + test_samples
            
            train_indices = np.arange(start_idx, train_end)
            test_indices = np.arange(train_end, test_end)
            
            yield train_indices, test_indices
            
            # Move forward by test period
            start_idx = train_end
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        if X is not None:
            n_samples = len(X)
            train_samples = self.train_days
            test_samples = self.test_days
            return max(0, (n_samples - train_samples) // test_samples)
        return 0

class PurgedKFoldCV(BaseCrossValidator):
    """
    Purged K-Fold cross-validation with embargo period.
    
    Designed for financial time series to prevent data leakage
    by purging observations that overlap with the test set.
    """
    
    def __init__(self, n_splits, embargo_days):
        """
        Initialize purged K-fold CV.
        
        Args:
            n_splits (int): Number of splits
            embargo_days (int): Number of days to embargo around test set
        """
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        
    def split(self, df, y=None, groups=None):
        """
        Generate purged train/test splits.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(df)
        test_size = n_samples // self.n_splits
        
        # Estimate samples per day
        if len(df) > 1:
            time_diffs = df.index[1:] - df.index[:-1]
            avg_diff = time_diffs.mean()
            samples_per_day = int(pd.Timedelta(days=1) / avg_diff)
        else:
            samples_per_day = 1
            
        embargo_samples = self.embargo_days * samples_per_day
        
        for i in range(self.n_splits):
            # Define test set
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = np.arange(test_start, test_end)
            
            # Define purged training set
            train_indices = []
            
            # Training data before test set (with embargo)
            if test_start - embargo_samples > 0:
                train_indices.extend(range(0, test_start - embargo_samples))
            
            # Training data after test set (with embargo)
            if test_end + embargo_samples < n_samples:
                train_indices.extend(range(test_end + embargo_samples, n_samples))
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits

class SeasonalCV(BaseCrossValidator):
    """
    Seasonal cross-validation that respects seasonal patterns.
    
    Splits data into seasonal chunks and uses different seasons
    for training and testing.
    """
    
    def __init__(self, n_splits, season_length):
        """
        Initialize seasonal CV.
        
        Args:
            n_splits (int): Number of splits
            season_length (int): Length of each season in days
        """
        self.n_splits = n_splits
        self.season_length = season_length
        
    def split(self, df, y=None, groups=None):
        """
        Generate seasonal train/test splits.
        
        Args:
            df (pd.DataFrame): DataFrame with datetime index
            
        Yields:
            tuple: (train_indices, test_indices)
        """
        n_samples = len(df)
        
        # Estimate samples per day
        if len(df) > 1:
            time_diffs = df.index[1:] - df.index[:-1]
            avg_diff = time_diffs.mean()
            samples_per_day = int(pd.Timedelta(days=1) / avg_diff)
        else:
            samples_per_day = 1
            
        season_samples = self.season_length * samples_per_day
        n_seasons = n_samples // season_samples
        
        if n_seasons < self.n_splits + 1:
            raise ValueError(f"Not enough seasons: need {self.n_splits + 1}, have {n_seasons}")
        
        for i in range(self.n_splits):
            # Use one season for testing
            test_season = i % n_seasons
            test_start = test_season * season_samples
            test_end = min((test_season + 1) * season_samples, n_samples)
            test_indices = np.arange(test_start, test_end)
            
            # Use all other complete seasons for training
            train_indices = []
            for season in range(n_seasons):
                if season != test_season:
                    season_start = season * season_samples
                    season_end = min((season + 1) * season_samples, n_samples)
                    train_indices.extend(range(season_start, season_end))
            
            train_indices = np.array(train_indices)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits

class TimeSeriesCV(BaseCrossValidator):
    """
    Generic time series cross-validation with customizable parameters.
    """
    
    def __init__(self, n_splits, train_size=None, test_size=None, gap=0):
        """
        Initialize time series CV.
        
        Args:
            n_splits (int): Number of splits
            train_size (int, optional): Size of training set
            test_size (int, optional): Size of test set
            gap (int): Gap between train and test sets
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        
    def split(self, X, y=None, groups=None):
        """Generate time series splits."""
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
            
        if self.train_size is None:
            train_size = n_samples - test_size * self.n_splits - self.gap * self.n_splits
            train_size = train_size // self.n_splits
        else:
            train_size = self.train_size
        
        for i in range(self.n_splits):
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = test_start + test_size
            
            train_end = test_start - self.gap
            train_start = train_end - train_size
            
            if train_start >= 0:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                yield train_indices, test_indices
                
    def get_n_splits(self, X=None, y=None, groups=None):
        """Get number of splits."""
        return self.n_splits



