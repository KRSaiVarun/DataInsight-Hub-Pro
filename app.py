import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.visualizations import Visualizations
from utils.statistics import Statistics
from utils.filters import Filters
from utils.resume_analyzer import ResumeAnalyzer
from utils.insight_generator import InsightGenerator
from utils.sample_data import SampleDataGenerator

# Page configuration
st.set_page_config(
    page_title="DataInsightHub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("styles/custom.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

class DataInsightHub:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.visualizations = Visualizations()
        self.statistics = Statistics()
        self.filters = Filters()
        self.resume_analyzer = ResumeAnalyzer()
        self.insight_generator = InsightGenerator()
        self.sample_data_generator = SampleDataGenerator()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filtered_data' not in st.session_state:
            st.session_state.filtered_data = None
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = "Overview"

    def render_sidebar(self):
        """Render the sidebar navigation and controls"""
        st.sidebar.title("📊 DataInsightHub")
        st.sidebar.markdown("---")
        
        # File upload section
        st.sidebar.subheader("📁 Data Upload")
        
        # Add sample data option
        use_sample_data = st.sidebar.checkbox("🎲 Use Sample Data", help="Load sample dataset for demonstration")
        
        if use_sample_data:
            sample_datasets = self.sample_data_generator.get_available_datasets()
            selected_dataset = st.sidebar.selectbox(
                "Choose sample dataset:",
                list(sample_datasets.keys()),
                help="Select a sample dataset to explore the platform features"
            )
            
            if st.sidebar.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    if selected_dataset == "Sales Performance Data":
                        st.session_state.data = self.sample_data_generator.generate_sales_data()
                    elif selected_dataset == "Employee Analytics Data":
                        st.session_state.data = self.sample_data_generator.generate_employee_data()
                    elif selected_dataset == "Customer Behavior Data":
                        st.session_state.data = self.sample_data_generator.generate_customer_data()
                    elif selected_dataset == "Financial Performance Data":
                        st.session_state.data = self.sample_data_generator.generate_financial_data()
                    
                    st.session_state.filtered_data = st.session_state.data.copy()
                
                st.sidebar.success("✅ Sample data loaded successfully!")
                st.sidebar.info(f"**Rows:** {len(st.session_state.data)}\n**Columns:** {len(st.session_state.data.columns)}")
        
        else:
            uploaded_file = st.sidebar.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel files for analysis"
            )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Loading data..."):
                    st.session_state.data = self.data_processor.load_file(uploaded_file)
                    st.session_state.filtered_data = st.session_state.data.copy()
                st.sidebar.success("✅ Data loaded successfully!")
                
                # Show basic data info
                st.sidebar.info(f"**Rows:** {len(st.session_state.data)}\n**Columns:** {len(st.session_state.data.columns)}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
                return
        
        if st.session_state.data is not None:
            st.sidebar.markdown("---")
            
            # Navigation menu
            st.sidebar.subheader("🧭 Navigation")
            analysis_options = [
                "Overview",
                "Summary Statistics", 
                "Data Visualization",
                "Correlation Analysis",
                "Data Filtering",
                "Resume Analyzer",
                "AI Insights"
            ]
            
            st.session_state.analysis_type = st.sidebar.selectbox(
                "Select Analysis Type",
                analysis_options,
                index=analysis_options.index(st.session_state.analysis_type)
            )
            
            # Quick filters in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎛️ Quick Filters")
            self.render_quick_filters()

    def render_quick_filters(self):
        """Render quick filtering options in sidebar"""
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Numeric columns for range filtering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_numeric = st.sidebar.selectbox(
                    "Filter by numeric column:",
                    ["None"] + numeric_cols
                )
                
                if selected_numeric != "None":
                    min_val = float(df[selected_numeric].min())
                    max_val = float(df[selected_numeric].max())
                    
                    range_values = st.sidebar.slider(
                        f"Range for {selected_numeric}",
                        min_val, max_val, (min_val, max_val)
                    )
                    
                    # Apply filter
                    mask = (df[selected_numeric] >= range_values[0]) & (df[selected_numeric] <= range_values[1])
                    st.session_state.filtered_data = df[mask]
            
            # Categorical columns for selection filtering
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                selected_categorical = st.sidebar.selectbox(
                    "Filter by categorical column:",
                    ["None"] + categorical_cols
                )
                
                if selected_categorical != "None":
                    unique_values = df[selected_categorical].unique()
                    selected_values = st.sidebar.multiselect(
                        f"Select {selected_categorical} values:",
                        unique_values,
                        default=unique_values
                    )
                    
                    if selected_values:
                        mask = df[selected_categorical].isin(selected_values)
                        if 'filtered_data' in st.session_state:
                            st.session_state.filtered_data = st.session_state.filtered_data[
                                st.session_state.filtered_data[selected_categorical].isin(selected_values)
                            ]
                        else:
                            st.session_state.filtered_data = df[mask]

    def render_overview(self):
        """Render the overview page"""
        st.title("📊 DataInsightHub Overview")
        st.markdown("Welcome to your comprehensive data analysis platform!")
        
        if st.session_state.data is not None:
            df = st.session_state.filtered_data
            
            # Key metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📊 Total Rows",
                    value=f"{len(df):,}",
                    delta=f"{len(df) - len(st.session_state.data):,}" if len(df) != len(st.session_state.data) else None
                )
            
            with col2:
                st.metric(
                    label="📋 Columns",
                    value=len(df.columns)
                )
            
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric(
                    label="❓ Missing Values",
                    value=f"{missing_values:,}",
                    delta=f"{(missing_values/df.size)*100:.1f}%" if missing_values > 0 else "0%"
                )
            
            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.metric(
                    label="💾 Memory Usage",
                    value=f"{memory_usage:.1f} MB"
                )
            
            st.markdown("---")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=300
            )
            
            # Column information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Column Types")
                dtype_counts = df.dtypes.value_counts()
                st.bar_chart(dtype_counts)
            
            with col2:
                st.subheader("📈 Data Quality")
                quality_data = {
                    'Complete': len(df.columns) - df.isnull().any().sum(),
                    'Has Missing': df.isnull().any().sum(),
                    'All Missing': df.isnull().all().sum()
                }
                st.bar_chart(quality_data)
        
        else:
            st.info("👆 Please upload a file using the sidebar to begin your analysis.")

    def render_summary_statistics(self):
        """Render summary statistics page"""
        st.title("📈 Summary Statistics")
        
        if st.session_state.filtered_data is not None:
            df = st.session_state.filtered_data
            
            # Generate comprehensive statistics
            stats_data = self.statistics.generate_comprehensive_stats(df)
            
            # Display statistics
            if stats_data['numeric_stats'] is not None:
                st.subheader("🔢 Numeric Variables")
                st.dataframe(
                    stats_data['numeric_stats'],
                    use_container_width=True
                )
            
            if stats_data['categorical_stats'] is not None:
                st.subheader("📝 Categorical Variables")
                st.dataframe(
                    stats_data['categorical_stats'],
                    use_container_width=True
                )
            
            # Missing values analysis
            st.subheader("❓ Missing Values Analysis")
            missing_analysis = self.statistics.analyze_missing_values(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(missing_analysis, use_container_width=True)
            
            with col2:
                if missing_analysis['Missing Count'].sum() > 0:
                    fig = self.visualizations.create_missing_values_heatmap(df)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Please upload a file to view summary statistics.")

    def render_visualizations(self):
        """Render data visualization page"""
        st.title("📊 Data Visualization")
        
        if st.session_state.filtered_data is not None:
            df = st.session_state.filtered_data
            
            # Visualization controls
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Histogram", "Scatter Plot", "Box Plot", "Bar Chart", "Line Chart"]
            )
            
            if viz_type == "Histogram":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column:", numeric_cols)
                    bins = st.slider("Number of bins:", 10, 100, 30)
                    
                    fig = self.visualizations.create_histogram(df, selected_col, bins)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis:", numeric_cols)
                    with col2:
                        y_col = st.selectbox("Y-axis:", numeric_cols, index=1)
                    
                    # Optional color coding
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                    color_col = None if color_col == "None" else color_col
                    
                    fig = self.visualizations.create_scatter_plot(df, x_col, y_col, color_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column:", numeric_cols)
                    
                    # Optional grouping
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    group_col = st.selectbox("Group by (optional):", ["None"] + categorical_cols)
                    group_col = None if group_col == "None" else group_col
                    
                    fig = self.visualizations.create_box_plot(df, selected_col, group_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Bar Chart":
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    selected_col = st.selectbox("Select column:", categorical_cols)
                    
                    fig = self.visualizations.create_bar_chart(df, selected_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Line Chart":
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column:", numeric_cols)
                    
                    # Check if there's a datetime column for x-axis
                    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                    if datetime_cols:
                        x_col = st.selectbox("X-axis (time):", datetime_cols)
                        fig = self.visualizations.create_line_chart(df, x_col, selected_col)
                    else:
                        fig = self.visualizations.create_line_chart(df, df.index, selected_col)
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Please upload a file to create visualizations.")

    def render_correlation_analysis(self):
        """Render correlation analysis page"""
        st.title("🔗 Correlation Analysis")
        
        if st.session_state.filtered_data is not None:
            df = st.session_state.filtered_data
            numeric_df = df.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) >= 2:
                # Correlation heatmap
                st.subheader("🔥 Correlation Heatmap")
                fig = self.visualizations.create_correlation_heatmap(numeric_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation table
                st.subheader("📊 Correlation Matrix")
                corr_matrix = numeric_df.corr()
                st.dataframe(
                    corr_matrix.round(3),
                    use_container_width=True
                )
                
                # Strong correlations
                st.subheader("💪 Strong Correlations")
                strong_corr = self.statistics.find_strong_correlations(corr_matrix)
                if not strong_corr.empty:
                    st.dataframe(strong_corr, use_container_width=True)
                else:
                    st.info("No strong correlations found (threshold: 0.7)")
            
            else:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        else:
            st.info("👆 Please upload a file to perform correlation analysis.")

    def render_data_filtering(self):
        """Render advanced data filtering page"""
        st.title("🎛️ Advanced Data Filtering")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            
            st.subheader("🔍 Filter Configuration")
            
            # Show current filter status
            filtered_rows = len(st.session_state.filtered_data)
            total_rows = len(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{total_rows:,}")
            with col2:
                st.metric("Filtered Rows", f"{filtered_rows:,}")
            with col3:
                percentage = (filtered_rows / total_rows) * 100
                st.metric("Remaining", f"{percentage:.1f}%")
            
            st.markdown("---")
            
            # Advanced filtering interface
            filter_data = self.filters.create_advanced_filters(df)
            
            if st.button("🔄 Reset All Filters"):
                st.session_state.filtered_data = st.session_state.data.copy()
                st.rerun()
            
            # Show filtered data preview
            st.subheader("📋 Filtered Data Preview")
            st.dataframe(
                st.session_state.filtered_data.head(20),
                use_container_width=True,
                height=400
            )
            
            # Download filtered data
            if st.button("💾 Download Filtered Data"):
                csv = st.session_state.filtered_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("👆 Please upload a file to access filtering options.")

    def render_resume_analyzer(self):
        """Render the resume analyzer page"""
        st.title("📄 Resume Analyzer")
        st.markdown("Upload and analyze resumes to extract key information and skills")
        
        # File upload for resume
        uploaded_resume = st.file_uploader(
            "Upload Resume",
            type=['txt', 'pdf'],
            help="Upload a text file or PDF containing the resume content"
        )
        
        if uploaded_resume is not None:
            try:
                # Extract text from resume
                resume_text = self.resume_analyzer.extract_text_from_upload(uploaded_resume)
                
                if resume_text.strip():
                    # Analyze resume
                    with st.spinner("Analyzing resume..."):
                        analysis = self.resume_analyzer.analyze_resume(resume_text)
                    
                    # Display results in tabs
                    analysis_tabs = st.tabs(["📊 Summary", "🔧 Technical Skills", "💼 Soft Skills", "🎓 Education", "📞 Contact Info"])
                    
                    with analysis_tabs[0]:
                        st.subheader("📊 Resume Summary")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Technical Skills", analysis['summary_stats']['total_technical_skills'])
                        
                        with col2:
                            st.metric("Soft Skills", analysis['summary_stats']['total_soft_skills'])
                        
                        with col3:
                            st.metric("Education Level", analysis['summary_stats']['education_level'])
                        
                        with col4:
                            st.metric("Estimated Experience", f"{analysis['summary_stats']['estimated_experience']} years")
                    
                    with analysis_tabs[1]:
                        st.subheader("🔧 Technical Skills")
                        
                        for category, skills in analysis['technical_skills'].items():
                            if skills:
                                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                                for skill in skills:
                                    st.markdown(f"- {skill}")
                                st.markdown("---")
                        
                        if not any(analysis['technical_skills'].values()):
                            st.info("No technical skills detected in the resume.")
                    
                    with analysis_tabs[2]:
                        st.subheader("💼 Soft Skills")
                        
                        if analysis['soft_skills']:
                            for skill in analysis['soft_skills']:
                                st.markdown(f"- {skill}")
                        else:
                            st.info("No soft skills detected in the resume.")
                    
                    with analysis_tabs[3]:
                        st.subheader("🎓 Education")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Highest Education Level:**")
                            st.write(analysis['education']['highest_level'])
                            
                            if analysis['education']['degrees']:
                                st.markdown("**Degrees Found:**")
                                for degree in set(analysis['education']['degrees']):
                                    st.markdown(f"- {degree}")
                        
                        with col2:
                            if analysis['education']['institutions']:
                                st.markdown("**Institutions:**")
                                for institution in analysis['education']['institutions']:
                                    st.markdown(f"- {institution}")
                    
                    with analysis_tabs[4]:
                        st.subheader("📞 Contact Information")
                        
                        contact = analysis['contact_info']
                        contact_found = False
                        
                        if contact['email']:
                            st.markdown(f"**Email:** {contact['email']}")
                            contact_found = True
                        
                        if contact['phone']:
                            st.markdown(f"**Phone:** {contact['phone']}")
                            contact_found = True
                        
                        if contact['linkedin']:
                            st.markdown(f"**LinkedIn:** {contact['linkedin']}")
                            contact_found = True
                        
                        if contact['github']:
                            st.markdown(f"**GitHub:** {contact['github']}")
                            contact_found = True
                        
                        if not contact_found:
                            st.info("No contact information detected.")
                
                else:
                    st.warning("No text content found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error analyzing resume: {str(e)}")
        
        else:
            st.info("👆 Please upload a resume file (text or PDF format) to begin analysis.")

    def render_ai_insights(self):
        """Render the AI insights page"""
        st.title("🤖 AI-Powered Insights")
        st.markdown("Get automated insights and recommendations from your data")
        
        if st.session_state.filtered_data is not None:
            df = st.session_state.filtered_data
            
            # Generate insights
            with st.spinner("Generating AI insights..."):
                insights = self.insight_generator.generate_comprehensive_insights(df)
                summary_report = self.insight_generator.generate_summary_report(df)
            
            # Display insights in tabs
            insight_tabs = st.tabs(["📋 Executive Summary", "🔍 Detailed Insights", "💡 Recommendations"])
            
            with insight_tabs[0]:
                st.markdown(summary_report)
            
            with insight_tabs[1]:
                # Detailed insights by category
                for category, insight_list in insights.items():
                    if insight_list:
                        st.subheader(f"{category.replace('_', ' ').title()}")
                        for insight in insight_list:
                            st.markdown(f"- {insight}")
                        st.markdown("---")
            
            with insight_tabs[2]:
                st.subheader("💡 Action Items")
                
                # High priority insights
                high_priority = [insight for insight_list in insights.values() for insight in insight_list if "⚠️" in insight or "🚨" in insight]
                
                if high_priority:
                    st.markdown("**🔴 High Priority Actions:**")
                    for insight in high_priority:
                        st.markdown(f"- {insight}")
                
                # Recommendations
                if insights['recommendations']:
                    st.markdown("**📋 Recommended Next Steps:**")
                    for i, recommendation in enumerate(insights['recommendations'], 1):
                        st.markdown(f"{i}. {recommendation}")
                
                # Export insights
                st.markdown("---")
                if st.button("📄 Export Insights Report"):
                    report_data = summary_report.encode('utf-8')
                    st.download_button(
                        label="Download Report",
                        data=report_data,
                        file_name=f"insights_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        else:
            st.info("👆 Please upload and process data first to generate AI insights.")

    def run(self):
        """Main application runner"""
        self.render_sidebar()
        
        # Main content area
        if st.session_state.analysis_type == "Overview":
            self.render_overview()
        elif st.session_state.analysis_type == "Summary Statistics":
            self.render_summary_statistics()
        elif st.session_state.analysis_type == "Data Visualization":
            self.render_visualizations()
        elif st.session_state.analysis_type == "Correlation Analysis":
            self.render_correlation_analysis()
        elif st.session_state.analysis_type == "Data Filtering":
            self.render_data_filtering()
        elif st.session_state.analysis_type == "Resume Analyzer":
            self.render_resume_analyzer()
        elif st.session_state.analysis_type == "AI Insights":
            self.render_ai_insights()
        else:
            st.error("Invalid analysis type selected.")

# Run the application
if __name__ == "__main__":
    app = DataInsightHub()
    app.run()
