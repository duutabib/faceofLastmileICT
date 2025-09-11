import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add the project root to the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Import project modules
try:
    from src.data_manager import Manager
    from src.data_analyzer import Analyzer
    from src.data_visualizer import Visualizer
except ImportError as e:
    st.error(f"Error importing project modules: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="LabFit - Inclusive Energy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("📊 LabFit - Inclusive Energy Dashboard")
st.markdown("""
This dashboard provides an interactive interface to explore and analyze energy data.
Upload your data file or use the sample data to get started.
""")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Data Input")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)",
        type=['csv', 'xlsx'],
        help="Upload your data file to get started"
    )
    
    # Sample data option
    use_sample = st.checkbox("Use sample data", value=True)
    
    # Analysis options
    st.header("Analysis Options")
    show_summary = st.checkbox("Show data summary", value=True)
    show_visualizations = st.checkbox("Show visualizations", value=True)

# Load data
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Data loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    elif use_sample:
        try:
            # Create sample data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            data = {
                'date': dates,
                'energy_consumption': [100 + i*2 + (i % 7)*5 for i in range(100)],
                'temperature': [20 + 10 * (i % 24)/24 + (i % 7)*2 for i in range(100)],
                'carbon_emissions': [50 + i*1.5 + (i % 7)*3 for i in range(100)]
            }
            df = pd.DataFrame(data)
            st.session_state.df = df
            return df
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            return None
    return None

# Main app logic
def main():
    # Load data
    df = load_data(uploaded_file, use_sample)
    
    if df is None and not use_sample and uploaded_file is None:
        st.info("Please upload a file or use the sample data to get started.")
        return
    
    # Show data summary
    if show_summary and df is not None:
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.subheader("Data Information")
            buffer = []
            buffer.append(f"**Number of Rows:** {len(df)}")
            buffer.append(f"**Number of Columns:** {len(df.columns)}")
            buffer.append("\n**Column Types:**")
            for col in df.columns:
                buffer.append(f"- {col}: {df[col].dtype}")
            st.markdown("\n".join(buffer))
    
    # Data Analysis
    if df is not None and show_visualizations:
        st.header("Data Analysis")
        
        # Initialize analyzer and visualizer
        analyzer = DataAnalyzer(df)
        visualizer = DataVisualizer(df)
        
        # Time series visualization
        if 'date' in df.columns:
            st.subheader("Time Series Analysis")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X-axis", ['date'], index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=0)
            
            try:
                fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over Time")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating time series plot: {e}")
        
        # Summary statistics
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        try:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1
                )
                fig.update_layout(title="Feature Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {e}")

if __name__ == "__main__":
    main()
