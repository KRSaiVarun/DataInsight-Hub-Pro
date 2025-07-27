import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date

class Filters:
    """Handles advanced filtering operations"""
    
    def __init__(self):
        pass
    
    def apply_numeric_filter(self, df, column, operator, value):
        """
        Apply numeric filter to dataframe
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Column name
            operator (str): Comparison operator (>, <, >=, <=, ==, !=)
            value (float): Filter value
            
        Returns:
            pandas.DataFrame: Filtered dataframe
        """
        if column not in df.columns:
            return df
        
        if operator == '>':
            return df[df[column] > value]
        elif operator == '<':
            return df[df[column] < value]
        elif operator == '>=':
            return df[df[column] >= value]
        elif operator == '<=':
            return df[df[column] <= value]
        elif operator == '==':
            return df[df[column] == value]
        elif operator == '!=':
            return df[df[column] != value]
        else:
            return df
    
    def apply_text_filter(self, df, column, operator, value):
        """
        Apply text filter to dataframe
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Column name
            operator (str): Filter type (contains, equals, starts_with, ends_with)
            value (str): Filter value
            
        Returns:
            pandas.DataFrame: Filtered dataframe
        """
        if column not in df.columns:
            return df
        
        if operator == 'contains':
            return df[df[column].str.contains(value, na=False, case=False)]
        elif operator == 'equals':
            return df[df[column].str.lower() == value.lower()]
        elif operator == 'starts_with':
            return df[df[column].str.startswith(value, na=False)]
        elif operator == 'ends_with':
            return df[df[column].str.endswith(value, na=False)]
        else:
            return df
    
    def create_advanced_filters(self, df):
        """
        Create advanced filtering interface
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            pandas.DataFrame: Filtered dataframe
        """
        st.subheader("🔧 Filter Builder")
        
        # Initialize filtered data if not exists
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = df.copy()
        
        # Filter tabs
        filter_tabs = st.tabs(["📊 Numeric Filters", "📝 Text Filters", "📅 Date Filters", "🎯 Custom Filters"])
        
        with filter_tabs[0]:
            self.create_numeric_filters(df)
        
        with filter_tabs[1]:
            self.create_text_filters(df)
        
        with filter_tabs[2]:
            self.create_date_filters(df)
        
        with filter_tabs[3]:
            self.create_custom_filters(df)
        
        return st.session_state.filtered_data
    
    def create_numeric_filters(self, df):
        """Create numeric column filters"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            st.info("No numeric columns available for filtering.")
            return
        
        st.subheader("Numeric Column Filters")
        
        for i, col in enumerate(numeric_cols):
            with st.expander(f"Filter by {col}", expanded=False):
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    st.warning(f"No data available for {col}")
                    continue
                
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                
                # Range slider
                range_key = f"range_{col}_{i}"
                selected_range = st.slider(
                    f"Select range for {col}",
                    min_val, max_val, (min_val, max_val),
                    key=range_key
                )
                
                # Comparison operators
                comparison_key = f"comparison_{col}_{i}"
                comparison_type = st.selectbox(
                    "Comparison type:",
                    ["Between (inclusive)", "Less than", "Greater than", "Equal to", "Not equal to"],
                    key=comparison_key
                )
                
                if comparison_type == "Between (inclusive)":
                    mask = (df[col] >= selected_range[0]) & (df[col] <= selected_range[1])
                elif comparison_type == "Less than":
                    threshold = st.number_input(f"Threshold for {col}:", value=max_val, key=f"threshold_lt_{col}_{i}")
                    mask = df[col] < threshold
                elif comparison_type == "Greater than":
                    threshold = st.number_input(f"Threshold for {col}:", value=min_val, key=f"threshold_gt_{col}_{i}")
                    mask = df[col] > threshold
                elif comparison_type == "Equal to":
                    value = st.number_input(f"Value for {col}:", value=min_val, key=f"value_eq_{col}_{i}")
                    mask = df[col] == value
                else:  # Not equal to
                    value = st.number_input(f"Value for {col}:", value=min_val, key=f"value_neq_{col}_{i}")
                    mask = df[col] != value
                
                # Apply filter button
                if st.button(f"Apply {col} filter", key=f"apply_numeric_{col}_{i}"):
                    st.session_state.filtered_data = st.session_state.filtered_data[mask]
                    st.success(f"Filter applied to {col}")
                    st.rerun()
    
    def create_text_filters(self, df):
        """Create text column filters"""
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_cols:
            st.info("No text columns available for filtering.")
            return
        
        st.subheader("Text Column Filters")
        
        for i, col in enumerate(text_cols):
            with st.expander(f"Filter by {col}", expanded=False):
                unique_values = df[col].dropna().unique()
                
                if len(unique_values) == 0:
                    st.warning(f"No data available for {col}")
                    continue
                
                # Filter type selection
                filter_type = st.selectbox(
                    "Filter type:",
                    ["Select values", "Contains text", "Starts with", "Ends with", "Regex pattern"],
                    key=f"text_filter_type_{col}_{i}"
                )
                
                if filter_type == "Select values":
                    # Multi-select for categorical values
                    if len(unique_values) <= 50:  # Limit for performance
                        selected_values = st.multiselect(
                            f"Select values for {col}:",
                            unique_values,
                            default=unique_values,
                            key=f"multiselect_{col}_{i}"
                        )
                        
                        if st.button(f"Apply {col} selection", key=f"apply_select_{col}_{i}"):
                            mask = df[col].isin(selected_values)
                            st.session_state.filtered_data = st.session_state.filtered_data[mask]
                            st.success(f"Selection filter applied to {col}")
                            st.rerun()
                    else:
                        st.warning(f"Too many unique values ({len(unique_values)}) for multi-select. Use text search instead.")
                
                elif filter_type == "Contains text":
                    search_text = st.text_input(f"Text to search in {col}:", key=f"contains_{col}_{i}")
                    case_sensitive = st.checkbox("Case sensitive", key=f"case_contains_{col}_{i}")
                    
                    if st.button(f"Apply contains filter", key=f"apply_contains_{col}_{i}") and search_text:
                        if case_sensitive:
                            mask = df[col].astype(str).str.contains(search_text, na=False)
                        else:
                            mask = df[col].astype(str).str.contains(search_text, case=False, na=False)
                        
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success(f"Contains filter applied to {col}")
                        st.rerun()
                
                elif filter_type == "Starts with":
                    prefix_text = st.text_input(f"Prefix for {col}:", key=f"startswith_{col}_{i}")
                    case_sensitive = st.checkbox("Case sensitive", key=f"case_starts_{col}_{i}")
                    
                    if st.button(f"Apply starts with filter", key=f"apply_starts_{col}_{i}") and prefix_text:
                        if case_sensitive:
                            mask = df[col].astype(str).str.startswith(prefix_text, na=False)
                        else:
                            mask = df[col].astype(str).str.lower().str.startswith(prefix_text.lower(), na=False)
                        
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success(f"Starts with filter applied to {col}")
                        st.rerun()
                
                elif filter_type == "Ends with":
                    suffix_text = st.text_input(f"Suffix for {col}:", key=f"endswith_{col}_{i}")
                    case_sensitive = st.checkbox("Case sensitive", key=f"case_ends_{col}_{i}")
                    
                    if st.button(f"Apply ends with filter", key=f"apply_ends_{col}_{i}") and suffix_text:
                        if case_sensitive:
                            mask = df[col].astype(str).str.endswith(suffix_text, na=False)
                        else:
                            mask = df[col].astype(str).str.lower().str.endswith(suffix_text.lower(), na=False)
                        
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success(f"Ends with filter applied to {col}")
                        st.rerun()
                
                elif filter_type == "Regex pattern":
                    regex_pattern = st.text_input(f"Regex pattern for {col}:", key=f"regex_{col}_{i}")
                    
                    if st.button(f"Apply regex filter", key=f"apply_regex_{col}_{i}") and regex_pattern:
                        try:
                            mask = df[col].astype(str).str.contains(regex_pattern, regex=True, na=False)
                            st.session_state.filtered_data = st.session_state.filtered_data[mask]
                            st.success(f"Regex filter applied to {col}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Invalid regex pattern: {str(e)}")
    
    def create_date_filters(self, df):
        """Create date column filters"""
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if not date_cols:
            st.info("No date columns available for filtering.")
            return
        
        st.subheader("Date Column Filters")
        
        for i, col in enumerate(date_cols):
            with st.expander(f"Filter by {col}", expanded=False):
                col_data = df[col].dropna()
                
                if len(col_data) == 0:
                    st.warning(f"No data available for {col}")
                    continue
                
                min_date = col_data.min().date()
                max_date = col_data.max().date()
                
                # Date range selector
                date_range = st.date_input(
                    f"Select date range for {col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"date_range_{col}_{i}"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    
                    if st.button(f"Apply date filter", key=f"apply_date_{col}_{i}"):
                        mask = (df[col].dt.date >= start_date) & (df[col].dt.date <= end_date)
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success(f"Date filter applied to {col}")
                        st.rerun()
                
                # Quick date filters
                st.subheader("Quick Date Filters")
                quick_filters = st.columns(3)
                
                with quick_filters[0]:
                    if st.button("Last 30 days", key=f"last30_{col}_{i}"):
                        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=30)
                        mask = df[col] >= cutoff_date
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success("Last 30 days filter applied")
                        st.rerun()
                
                with quick_filters[1]:
                    if st.button("Last 90 days", key=f"last90_{col}_{i}"):
                        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                        mask = df[col] >= cutoff_date
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success("Last 90 days filter applied")
                        st.rerun()
                
                with quick_filters[2]:
                    if st.button("This year", key=f"thisyear_{col}_{i}"):
                        current_year = pd.Timestamp.now().year
                        mask = df[col].dt.year == current_year
                        st.session_state.filtered_data = st.session_state.filtered_data[mask]
                        st.success("This year filter applied")
                        st.rerun()
    
    def create_custom_filters(self, df):
        """Create custom query-based filters"""
        st.subheader("Custom Query Filters")
        st.info("💡 Use pandas query syntax. Example: `age > 25 and income < 50000`")
        
        # Show available columns
        with st.expander("Available Columns", expanded=False):
            col_info = []
            for col in df.columns:
                col_type = str(df[col].dtype)
                sample_values = df[col].dropna().head(3).tolist()
                col_info.append(f"**{col}** ({col_type}): {sample_values}")
            
            for info in col_info:
                st.markdown(info)
        
        # Query input
        query_text = st.text_area(
            "Enter your query:",
            placeholder="Example: age > 25 and category == 'A'",
            height=100
        )
        
        if st.button("Apply Custom Query") and query_text.strip():
            try:
                # Test the query first
                test_result = df.query(query_text)
                
                # Apply to filtered data
                st.session_state.filtered_data = st.session_state.filtered_data.query(query_text)
                st.success(f"Custom query applied! Filtered to {len(st.session_state.filtered_data)} rows.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Query error: {str(e)}")
                st.info("Please check your query syntax and column names.")
        
        # Query examples
        with st.expander("Query Examples", expanded=False):
            st.markdown("""
            **Numeric comparisons:**
            - `price > 100`
            - `age >= 18 and age < 65`
            - `score.between(80, 90)`
            
            **Text filtering:**
            - `name.str.contains('John')`
            - `category == 'Electronics'`
            - `status.isin(['Active', 'Pending'])`
            
            **Date filtering:**
            - `date > '2023-01-01'`
            - `date.dt.year == 2023`
            
            **Complex queries:**
            - `(category == 'A' or category == 'B') and price > 50`
            - `name.str.len() > 5 and score > 80`
            """)
    
    def export_filtered_data(self, df):
        """
        Export filtered data functionality
        
        Args:
            df (pandas.DataFrame): Filtered dataframe
            
        Returns:
            str: CSV data for download
        """
        return df.to_csv(index=False)
    
    def get_filter_summary(self, original_df, filtered_df):
        """
        Get summary of applied filters
        
        Args:
            original_df (pandas.DataFrame): Original dataframe
            filtered_df (pandas.DataFrame): Filtered dataframe
            
        Returns:
            dict: Filter summary statistics
        """
        summary = {
            'original_rows': len(original_df),
            'filtered_rows': len(filtered_df),
            'rows_removed': len(original_df) - len(filtered_df),
            'percentage_remaining': (len(filtered_df) / len(original_df)) * 100 if len(original_df) > 0 else 0,
            'columns_affected': len(original_df.columns)
        }
        
        return summary
