import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

class InsightGenerator:
    """Generates automated insights and interpretations from data analysis"""
    
    def __init__(self):
        pass
    
    def generate_insights(self, df):
        """
        Generate insights from dataframe
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            list: List of insights
        """
        results = self.generate_comprehensive_insights(df)
        all_insights = []
        for category_insights in results.values():
            if isinstance(category_insights, list):
                all_insights.extend(category_insights)
        return all_insights
    
    def generate_business_recommendations(self, df):
        """
        Generate business recommendations from dataframe
        
        Args:
            df (pandas.DataFrame): Input dataframe
            
        Returns:
            list: List of business recommendations
        """
        results = self.generate_comprehensive_insights(df)
        return results.get('recommendations', [])
    
    def generate_comprehensive_insights(self, df: pd.DataFrame, analysis_results: Dict = None) -> Dict[str, List[str]]:
        """
        Generate comprehensive insights from dataset analysis
        
        Args:
            df (pandas.DataFrame): Input dataframe
            analysis_results (dict): Optional analysis results from other modules
            
        Returns:
            dict: Categorized insights
        """
        insights = {
            'data_quality': self._analyze_data_quality(df),
            'statistical': self._generate_statistical_insights(df),
            'patterns': self._identify_patterns(df),
            'recommendations': self._generate_recommendations(df),
            'business_insights': self._generate_business_insights(df)
        }
        
        return insights
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Analyze data quality and generate insights"""
        insights = []
        
        # Missing data analysis
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 20:
            insights.append(f"⚠️ High missing data detected ({missing_percentage:.1f}%). Consider data imputation or collection improvement.")
        elif missing_percentage > 5:
            insights.append(f"📊 Moderate missing data present ({missing_percentage:.1f}%). Review data collection processes.")
        elif missing_percentage < 1:
            insights.append("✅ Excellent data completeness with minimal missing values.")
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / len(df)) * 100
            insights.append(f"🔄 Found {duplicate_count} duplicate records ({duplicate_percentage:.1f}%). Consider deduplication.")
        
        # Data type consistency
        object_cols = df.select_dtypes(include=['object']).columns
        mixed_type_cols = []
        for col in object_cols:
            try:
                pd.to_numeric(df[col], errors='raise')
                mixed_type_cols.append(col)
            except:
                pass
        
        if mixed_type_cols:
            insights.append(f"🔧 Potential data type issues in columns: {', '.join(mixed_type_cols)}. Consider type conversion.")
        
        # Dataset size assessment
        if len(df) < 100:
            insights.append("📏 Small dataset detected. Results may have limited statistical significance.")
        elif len(df) > 100000:
            insights.append("📈 Large dataset provides good statistical power for analysis.")
        
        return insights
    
    def _generate_statistical_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate insights based on statistical analysis"""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            insights.append("📝 Dataset contains only categorical data. Focus on frequency and distribution analysis.")
            return insights
        
        # Variability analysis
        high_variability_cols = []
        low_variability_cols = []
        
        for col in numeric_cols:
            if df[col].std() > 0:
                cv = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else float('inf')
                if cv > 1:
                    high_variability_cols.append(col)
                elif cv < 0.1:
                    low_variability_cols.append(col)
        
        if high_variability_cols:
            insights.append(f"📊 High variability detected in: {', '.join(high_variability_cols)}. Consider normalization or outlier analysis.")
        
        if low_variability_cols:
            insights.append(f"📉 Low variability in: {', '.join(low_variability_cols)}. These variables may have limited predictive power.")
        
        # Skewness analysis
        skewed_cols = []
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                skewed_cols.append(f"{col} ({direction})")
        
        if skewed_cols:
            insights.append(f"📐 Significant skewness detected in: {', '.join(skewed_cols)}. Consider transformation for normality.")
        
        # Outlier analysis
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(f"{col} ({outliers} outliers)")
        
        if outlier_cols:
            insights.append(f"🎯 Notable outliers in: {', '.join(outlier_cols)}. Investigate for data quality or special cases.")
        
        return insights
    
    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        """Identify patterns and relationships in the data"""
        insights = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Correlation patterns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Strong positive correlations
            strong_positive = []
            strong_negative = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.7:
                        strong_positive.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} ({corr_val:.2f})")
                    elif corr_val < -0.7:
                        strong_negative.append(f"{corr_matrix.columns[i]} & {corr_matrix.columns[j]} ({corr_val:.2f})")
            
            if strong_positive:
                insights.append(f"🔗 Strong positive correlations found: {', '.join(strong_positive)}")
            
            if strong_negative:
                insights.append(f"🔄 Strong negative correlations found: {', '.join(strong_negative)}")
        
        # Categorical patterns
        for col in categorical_cols:
            if df[col].nunique() > 1:
                value_counts = df[col].value_counts()
                
                # Dominant category
                if len(value_counts) > 0:
                    dominant_pct = (value_counts.iloc[0] / len(df)) * 100
                    if dominant_pct > 80:
                        insights.append(f"📊 {col} is dominated by '{value_counts.index[0]}' ({dominant_pct:.1f}% of data)")
                
                # Category distribution
                if len(value_counts) > 10:
                    insights.append(f"🗂️ {col} has high cardinality ({df[col].nunique()} unique values). Consider grouping rare categories.")
        
        # Time series patterns (if datetime columns exist)
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            insights.append("📅 Time-based data detected. Consider seasonal trends and time series analysis.")
        
        return insights
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Analysis recommendations
        if len(numeric_cols) >= 2:
            recommendations.append("📈 Consider regression analysis to model relationships between numeric variables")
            
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            recommendations.append("📊 Perform group-by analysis to understand how categories affect numeric metrics")
            
        if len(categorical_cols) >= 2:
            recommendations.append("🔍 Use chi-square tests to analyze relationships between categorical variables")
            
        # Data preprocessing recommendations
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage > 5:
            recommendations.append("🛠️ Implement missing data imputation strategies (mean, median, or advanced methods)")
            
        if len(df) > 10000:
            recommendations.append("⚡ Consider sampling techniques for faster exploratory analysis on this large dataset")
            
        # Visualization recommendations
        if len(numeric_cols) > 0:
            recommendations.append("📊 Create distribution plots to understand data spread and identify outliers")
            
        if len(categorical_cols) > 0:
            recommendations.append("📋 Use bar charts and pie charts to visualize categorical distributions")
            
        if len(numeric_cols) >= 2:
            recommendations.append("🎯 Generate scatter plots and correlation heatmaps to explore relationships")
            
        return recommendations
    
    def _generate_business_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate business-oriented insights"""
        insights = []
        
        # Dataset characteristics
        total_records = len(df)
        total_features = len(df.columns)
        
        # Business scale insights
        if total_records < 1000:
            insights.append("🏢 Small-scale dataset suitable for detailed individual analysis and quality review")
        elif total_records < 10000:
            insights.append("🏬 Medium-scale dataset ideal for departmental analysis and tactical decisions")
        else:
            insights.append("🏭 Large-scale dataset suitable for strategic analysis and predictive modeling")
        
        # Feature richness
        if total_features > 20:
            insights.append("📋 Rich feature set provides comprehensive analytical opportunities")
        elif total_features < 5:
            insights.append("📝 Limited feature set may require additional data collection for deeper insights")
        
        # Data completeness business impact
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        if missing_percentage < 2:
            insights.append("✅ High data quality supports reliable business decision-making")
        elif missing_percentage > 15:
            insights.append("⚠️ Data quality issues may impact decision confidence - invest in data collection improvement")
        
        # Actionability insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            insights.append("📊 Quantitative metrics available for KPI tracking and performance measurement")
            
        if len(categorical_cols) > 0:
            insights.append("🏷️ Categorical data enables segmentation and targeted analysis")
        
        return insights
    
    def generate_summary_report(self, df: pd.DataFrame, analysis_results: Dict = None) -> str:
        """
        Generate a comprehensive summary report
        
        Args:
            df (pandas.DataFrame): Input dataframe
            analysis_results (dict): Optional analysis results
            
        Returns:
            str: Formatted summary report
        """
        insights = self.generate_comprehensive_insights(df, analysis_results)
        
        report = f"""
# 📊 DataInsightHub Analysis Report
        
## Dataset Overview
- **Records**: {len(df):,}
- **Features**: {len(df.columns)}
- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
- **Missing Data**: {(df.isnull().sum().sum() / df.size) * 100:.1f}%

## 🔍 Key Insights

### Data Quality
{chr(10).join(['- ' + insight for insight in insights['data_quality']])}

### Statistical Findings
{chr(10).join(['- ' + insight for insight in insights['statistical']])}

### Patterns Identified
{chr(10).join(['- ' + insight for insight in insights['patterns']])}

### Business Implications
{chr(10).join(['- ' + insight for insight in insights['business_insights']])}

## 💡 Recommendations
{chr(10).join(['- ' + rec for rec in insights['recommendations']])}

---
*Report generated automatically by DataInsightHub Insight Generator*
        """
        
        return report.strip()
    
    def create_insight_summary_data(self, insights: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create structured data from insights for visualization
        
        Args:
            insights (dict): Generated insights by category
            
        Returns:
            pandas.DataFrame: Structured insights data
        """
        insight_data = []
        
        for category, insight_list in insights.items():
            for i, insight in enumerate(insight_list):
                # Extract priority/sentiment from emoji/text
                priority = "High" if any(emoji in insight for emoji in ["⚠️", "🚨"]) else "Medium"
                if any(emoji in insight for emoji in ["✅", "📈", "💡"]) :
                    priority = "Low"  # Positive insights are low priority for action
                
                insight_data.append({
                    'Category': category.replace('_', ' ').title(),
                    'Insight': insight,
                    'Priority': priority,
                    'Index': i + 1
                })
        
        return pd.DataFrame(insight_data)