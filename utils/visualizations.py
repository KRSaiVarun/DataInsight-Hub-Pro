import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns

class Visualizations:
    """Handles all visualization creation using Plotly"""
    
    def __init__(self):
        # Color palette matching the design specification
        self.colors = {
            'primary': '#FF4B4B',
            'secondary': '#0E1117',
            'background': '#FFFFFF',
            'sidebar': '#F0F2F6',
            'success': '#00CC88',
            'text': '#262730'
        }
        
        # Default plotly template
        self.template = {
            'layout': {
                'font': {'family': 'Source Sans Pro, sans-serif', 'color': self.colors['text']},
                'plot_bgcolor': self.colors['background'],
                'paper_bgcolor': self.colors['background'],
                'colorway': [self.colors['primary'], self.colors['success'], '#1f77b4', '#ff7f0e', '#2ca02c']
            }
        }
    
    def create_histogram(self, df, column, bins=30):
        """
        Create an interactive histogram
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Column name for histogram
            bins (int): Number of bins
            
        Returns:
            plotly.graph_objects.Figure: Histogram figure
        """
        fig = px.histogram(
            df, 
            x=column,
            nbins=bins,
            title=f'Distribution of {column}',
            labels={column: column, 'count': 'Frequency'},
            color_discrete_sequence=[self.colors['primary']]
        )
        
        fig.update_layout(
            template=self.template,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Add statistics annotations
        mean_val = df[column].mean()
        median_val = df[column].median()
        
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color=self.colors['success'],
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        fig.add_vline(
            x=median_val, 
            line_dash="dot", 
            line_color=self.colors['secondary'],
            annotation_text=f"Median: {median_val:.2f}"
        )
        
        return fig
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None):
        """
        Create an interactive scatter plot
        
        Args:
            df (pandas.DataFrame): Input dataframe
            x_col (str): X-axis column
            y_col (str): Y-axis column
            color_col (str, optional): Column for color coding
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        if color_col:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                color=color_col,
                title=f'{y_col} vs {x_col}',
                hover_data=[color_col]
            )
        else:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                title=f'{y_col} vs {x_col}',
                color_discrete_sequence=[self.colors['primary']]
            )
        
        fig.update_layout(template=self.template)
        
        # Add trend line
        if len(df) > 10:  # Only add trend line if enough points
            fig.add_traces(
                px.scatter(df, x=x_col, y=y_col, trendline="ols").data[1:]
            )
        
        return fig
    
    def create_box_plot(self, df, column, group_col=None):
        """
        Create an interactive box plot
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Column for box plot
            group_col (str, optional): Column for grouping
            
        Returns:
            plotly.graph_objects.Figure: Box plot figure
        """
        if group_col:
            fig = px.box(
                df, 
                x=group_col, 
                y=column,
                title=f'Distribution of {column} by {group_col}',
                color=group_col
            )
        else:
            fig = px.box(
                df, 
                y=column,
                title=f'Distribution of {column}',
                color_discrete_sequence=[self.colors['primary']]
            )
        
        fig.update_layout(template=self.template)
        
        return fig
    
    def create_bar_chart(self, df, column):
        """
        Create an interactive bar chart for categorical data
        
        Args:
            df (pandas.DataFrame): Input dataframe
            column (str): Categorical column
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        value_counts = df[column].value_counts().head(20)  # Top 20 categories
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Count of {column}',
            labels={'x': column, 'y': 'Count'},
            color_discrete_sequence=[self.colors['primary']]
        )
        
        fig.update_layout(
            template=self.template,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_line_chart(self, df, x_col, y_col):
        """
        Create an interactive line chart
        
        Args:
            df (pandas.DataFrame): Input dataframe
            x_col (str): X-axis column (usually time-based)
            y_col (str): Y-axis column
            
        Returns:
            plotly.graph_objects.Figure: Line chart figure
        """
        # Sort by x column if it's not the index
        if x_col != df.index.name and x_col in df.columns:
            df_sorted = df.sort_values(x_col)
        else:
            df_sorted = df.copy()
            x_col = df.index
        
        fig = px.line(
            df_sorted,
            x=x_col,
            y=y_col,
            title=f'{y_col} over {x_col}',
            color_discrete_sequence=[self.colors['primary']]
        )
        
        fig.update_layout(template=self.template)
        
        return fig
    
    def create_correlation_heatmap(self, df):
        """
        Create a correlation heatmap
        
        Args:
            df (pandas.DataFrame): Input dataframe (numeric columns only)
            
        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            template=self.template,
            width=800,
            height=600
        )
        
        return fig
    
    def create_missing_values_heatmap(self, df):
        """
        Create a heatmap showing missing values pattern
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            plotly.graph_objects.Figure: Missing values heatmap
        """
        # Create binary matrix where 1 = missing, 0 = present
        missing_matrix = df.isnull().astype(int)
        
        # Only show columns that have missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            # No missing values
            fig = go.Figure()
            fig.add_annotation(
                text="No missing values found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            fig.update_layout(template=self.template)
            return fig
        
        missing_subset = missing_matrix[missing_cols]
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_subset.T.values,
            x=missing_subset.index,
            y=missing_subset.columns,
            colorscale=[[0, self.colors['success']], [1, self.colors['primary']]],
            showscale=True,
            colorbar=dict(
                title="Missing Values",
                tickvals=[0, 1],
                ticktext=["Present", "Missing"]
            )
        ))
        
        fig.update_layout(
            title='Missing Values Pattern',
            template=self.template,
            xaxis_title='Row Index',
            yaxis_title='Columns',
            height=400
        )
        
        return fig
    
    def create_distribution_comparison(self, df, columns, chart_type='histogram'):
        """
        Create subplots comparing distributions of multiple columns
        
        Args:
            df (pandas.DataFrame): Input dataframe
            columns (list): List of column names
            chart_type (str): Type of chart ('histogram' or 'box')
            
        Returns:
            plotly.graph_objects.Figure: Subplot figure
        """
        n_cols = min(len(columns), 3)
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=columns,
            vertical_spacing=0.1
        )
        
        for i, col in enumerate(columns):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            if chart_type == 'histogram':
                fig.add_trace(
                    go.Histogram(
                        x=df[col],
                        name=col,
                        marker_color=self.colors['primary'],
                        opacity=0.7
                    ),
                    row=row, col=col_pos
                )
            elif chart_type == 'box':
                fig.add_trace(
                    go.Box(
                        y=df[col],
                        name=col,
                        marker_color=self.colors['primary']
                    ),
                    row=row, col=col_pos
                )
        
        fig.update_layout(
            template=self.template,
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig
