import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

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
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Strip whitespace from string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            # Replace 'nan' strings with actual NaN
            df[col] = df[col].replace('nan', np.nan)
        
        # Attempt to convert date columns
        df = self.detect_and_convert_dates(df)
        
        # Optimize data types
        df = self.optimize_dtypes(df)
        
        return df
    
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
        Optimize data types to reduce memory usage
        
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
                if numeric_col.notna().sum() / len(df) > 0.5:
                    df[col] = numeric_col
            except:
                pass
        
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
