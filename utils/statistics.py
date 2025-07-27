import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class Statistics:
    """Handles statistical analysis and computations"""
    
    def __init__(self):
        pass
    
    def get_descriptive_stats(self, df):
        """
        Get descriptive statistics for numeric columns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Descriptive statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        
        return df[numeric_cols].describe()
    
    def get_correlation_matrix(self, df):
        """
        Get correlation matrix for numeric columns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        return df[numeric_cols].corr()
    
    def get_categorical_summary(self, df, column):
        """
        Get summary statistics for a categorical column
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Column name
            
        Returns:
            pandas.DataFrame: Categorical summary
        """
        if column not in df.columns:
            return pd.DataFrame()
        
        return df[column].value_counts().to_frame('count')
    
    def generate_comprehensive_stats(self, df):
        """
        Generate comprehensive statistics for all columns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary containing statistical summaries
        """
        results = {
            'numeric_stats': None,
            'categorical_stats': None,
            'general_info': {}
        }
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            numeric_stats = []
            
            for col in numeric_cols:
                series = df[col].dropna()
                
                if len(series) > 0:
                    stats_dict = {
                        'Column': col,
                        'Count': len(series),
                        'Missing': df[col].isnull().sum(),
                        'Mean': series.mean(),
                        'Median': series.median(),
                        'Mode': series.mode().iloc[0] if not series.mode().empty else np.nan,
                        'Std Dev': series.std(),
                        'Min': series.min(),
                        'Max': series.max(),
                        'Q1': series.quantile(0.25),
                        'Q3': series.quantile(0.75),
                        'IQR': series.quantile(0.75) - series.quantile(0.25),
                        'Skewness': float(stats.skew(series)),
                        'Kurtosis': float(stats.kurtosis(series)),
                        'Outliers': self.count_outliers(series)
                    }
                    numeric_stats.append(stats_dict)
            
            if numeric_stats:
                results['numeric_stats'] = pd.DataFrame(numeric_stats).round(3)
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            categorical_stats = []
            
            for col in categorical_cols:
                series = df[col].dropna()
                
                if len(series) > 0:
                    value_counts = series.value_counts()
                    
                    stats_dict = {
                        'Column': col,
                        'Count': len(series),
                        'Missing': df[col].isnull().sum(),
                        'Unique': series.nunique(),
                        'Top Value': value_counts.index[0] if len(value_counts) > 0 else 'N/A',
                        'Top Frequency': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                        'Top Percentage': (value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
                    }
                    categorical_stats.append(stats_dict)
            
            if categorical_stats:
                results['categorical_stats'] = pd.DataFrame(categorical_stats)
        
        # General information
        results['general_info'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'total_missing': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return results
    
    def count_outliers(self, series, method='iqr'):
        """
        Count outliers in a numeric series
        
        Args:
            series (pandas.Series): Input series
            method (str): Method for outlier detection ('iqr' or 'zscore')
            
        Returns:
            int: Number of outliers
        """
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series.values))
            outliers = (z_scores > 3).sum()
        
        else:
            outliers = 0
        
        return outliers
    
    def analyze_missing_values(self, df):
        """
        Analyze missing value patterns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Missing values analysis
        """
        missing_data = []
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_data.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing Percentage': round(missing_percentage, 2),
                'Data Type': str(df[col].dtype)
            })
        
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('Missing Count', ascending=False)
        
        return missing_df
    
    def find_strong_correlations(self, corr_matrix, threshold=0.7):
        """
        Find strong correlations in a correlation matrix
        
        Args:
            corr_matrix (pandas.DataFrame): Correlation matrix
            threshold (float): Correlation threshold
            
        Returns:
            pandas.DataFrame: Strong correlations
        """
        strong_corr = []
        
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with correlation above threshold
        for column in upper.columns:
            for index in upper.index:
                if abs(upper.loc[index, column]) >= threshold:
                    strong_corr.append({
                        'Variable 1': index,
                        'Variable 2': column,
                        'Correlation': round(upper.loc[index, column], 3),
                        'Strength': self.correlation_strength(abs(upper.loc[index, column]))
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            strong_corr_df = strong_corr_df.sort_values('Correlation', key=abs, ascending=False)
            return strong_corr_df
        else:
            return pd.DataFrame()
    
    def correlation_strength(self, corr_value):
        """
        Categorize correlation strength
        
        Args:
            corr_value (float): Absolute correlation value
            
        Returns:
            str: Correlation strength category
        """
        if corr_value >= 0.9:
            return "Very Strong"
        elif corr_value >= 0.7:
            return "Strong"
        elif corr_value >= 0.5:
            return "Moderate"
        elif corr_value >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def detect_data_quality_issues(self, df):
        """
        Detect various data quality issues
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary of detected issues
        """
        issues = {
            'duplicates': len(df) - len(df.drop_duplicates()),
            'missing_values': df.isnull().sum().sum(),
            'columns_all_null': df.columns[df.isnull().all()].tolist(),
            'columns_single_value': [],
            'potential_outliers': {},
            'inconsistent_formats': {}
        }
        
        # Check for columns with single unique value
        for col in df.columns:
            if df[col].nunique() == 1:
                issues['columns_single_value'].append(col)
        
        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            outlier_count = self.count_outliers(df[col])
            if outlier_count > 0:
                issues['potential_outliers'][col] = outlier_count
        
        # Check for inconsistent formats in text columns
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            # Check for mixed case issues
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                has_mixed_case = any(
                    str(val).islower() for val in non_null_values
                ) and any(
                    str(val).isupper() for val in non_null_values
                )
                
                if has_mixed_case:
                    issues['inconsistent_formats'][col] = "Mixed case detected"
        
        return issues
    
    def calculate_data_profiling_metrics(self, df):
        """
        Calculate comprehensive data profiling metrics
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Data profiling metrics
        """
        metrics = {
            'completeness': {},
            'uniqueness': {},
            'validity': {},
            'consistency': {}
        }
        
        total_rows = len(df)
        
        for col in df.columns:
            # Completeness (non-null percentage)
            non_null_count = df[col].count()
            metrics['completeness'][col] = (non_null_count / total_rows) * 100
            
            # Uniqueness (unique values percentage)
            unique_count = df[col].nunique()
            metrics['uniqueness'][col] = (unique_count / non_null_count) * 100 if non_null_count > 0 else 0
            
            # Basic validity checks
            if df[col].dtype in ['object']:
                # For text columns, check for empty strings
                empty_strings = (df[col] == '').sum()
                valid_count = non_null_count - empty_strings
                metrics['validity'][col] = (valid_count / total_rows) * 100
            else:
                # For numeric columns, assume all non-null values are valid
                metrics['validity'][col] = (non_null_count / total_rows) * 100
        
        return metrics
