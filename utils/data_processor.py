import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import re

class DataProcessor:
    """Handles data loading and preprocessing operations"""
    
    def __init__(self):
        pass
    
    def load_file(self, uploaded_file):
        """
        Load CSV or Excel file and return a pandas DataFrame
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    raise ValueError("Could not decode CSV file with any supported encoding")
                    
            elif file_extension in ['xlsx', 'xls']:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_excel(uploaded_file)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            # Basic data cleaning
            df = self.clean_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def clean_data(self, df):
        """
        Perform basic data cleaning operations
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index to ensure clean index
        df = df.reset_index(drop=True)
        
        # Strip whitespace from string columns and handle mixed types
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Handle mixed types safely
            try:
                # Handle complex objects and lists first
                if any(isinstance(x, (list, dict, tuple, set)) for x in df[col].dropna() if x is not None):
                    # Convert complex objects to strings
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else np.nan)
                
                # Convert to string, handling NaN values
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' and 'None' strings with actual NaN
                df[col] = df[col].replace(['nan', 'None', '<NA>', 'null', 'NaT'], np.nan)
                # Convert back empty strings to NaN
                df[col] = df[col].replace('', np.nan)
                # Remove any remaining problematic characters for Arrow compatibility
                df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)
                # Handle any remaining problematic values
                df[col] = df[col].apply(lambda x: str(x)[:1000] if isinstance(x, str) and len(str(x)) > 1000 else x)
            except Exception as e:
                # If conversion fails, create a clean string column
                st.warning(f"Cleaning column '{col}' due to mixed data types: {str(e)}")
                df[col] = df[col].apply(lambda x: str(x) if x is not None else np.nan)
        
        # Clean column names to avoid Arrow serialization issues
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Attempt to convert date columns
        df = self.detect_and_convert_dates(df)
        
        # Optimize data types
        df = self.optimize_dtypes(df)
        
        # Final Arrow compatibility check and fix
        df = self._ensure_arrow_compatibility(df)
        
        return df
    
    def _ensure_arrow_compatibility(self, df):
        """
        Final check to ensure DataFrame is compatible with Arrow serialization
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Arrow-compatible dataframe
        """
        df = df.copy()
        
        # Handle any remaining problematic columns
        for col in df.columns:
            try:
                # Test if column can be converted to Arrow
                import pyarrow as pa
                pa.array(df[col].dropna().head(100))
            except Exception as e:
                # If Arrow conversion fails, convert to safe string type
                st.warning(f"Converting column '{col}' to string for compatibility: {str(e)}")
                df[col] = df[col].astype(str).replace('nan', np.nan)
        
        # Ensure all column names are valid identifiers
        original_columns = df.columns.tolist()
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Check for duplicate column names after cleaning
        if len(set(df.columns)) != len(df.columns):
            # Handle duplicate column names
            seen = {}
            new_columns = []
            for col in df.columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
            df.columns = new_columns
        
        return df
    
    def _clean_column_name(self, col_name):
        """Clean column names to be compatible with Arrow"""
        # Convert to string and clean
        clean_name = str(col_name).strip()
        # Remove or replace problematic characters
        clean_name = re.sub(r'[^\w\s-]', '_', clean_name)
        # Replace spaces with underscores
        clean_name = re.sub(r'\s+', '_', clean_name)
        # Remove multiple underscores
        clean_name = re.sub(r'_+', '_', clean_name)
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        # Ensure it starts with a letter or underscore
        if clean_name and not clean_name[0].isalpha() and clean_name[0] != '_':
            clean_name = 'col_' + clean_name
        # Fallback for empty names
        if not clean_name:
            clean_name = f'column_{hash(col_name) % 10000}'
        return clean_name
    
    def detect_and_convert_dates(self, df):
        """
        Automatically detect and convert date columns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with converted date columns
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    # Sample a few non-null values to test
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        pd.to_datetime(sample_values, errors='raise')
                        # If successful, convert the entire column
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    # Not a date column, continue
                    pass
        
        return df
    
    def optimize_dtypes(self, df):
        """
        Optimize data types to reduce memory usage and ensure Arrow compatibility
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: DataFrame with optimized dtypes
        """
        # Convert object columns that are actually numeric
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                # If more than 50% of values are numeric, convert
                if len(df) > 0 and numeric_col.count() / len(df) > 0.5:
                    df[col] = numeric_col
            except Exception:
                continue
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def get_data_info(self, df):
        """
        Get comprehensive information about the dataset
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            dict: Dictionary containing data information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        
        return info
