import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# LabFit Constants
LABFIT_CONSTANTS = {
    'A': 0.21334,
    'B': 0.14933,
    'C': 0.033616,
    'D': 0.023546,
    'output_lpm': 100000,  # atm pressure in Pa
    'Tsb': 30.66,
    'KCorrection': 273.15,  # Kelvin temp correction
    'Tn': 293.15,  # standard temperature in Kelvin (20°C)
    'A1': 0.0005366622509, # diameter 1
    'A2': 0.0002010619298, # diameter 2
    'r0': 1.184,
    'epsilon': 0.1
}

# Add the project root to the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Import project modules
try:
    from src.data_reader import Reader
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
        "Upload your CSV/Excel file",
        type=["csv", "xlsx"],
        help="Upload your data file to get started"
    )
    
    # Sample data option
    use_sample = st.checkbox("Use sample data", value=True)
    
    # Display options
    st.header("Display Options")
    show_summary = st.checkbox("Show Data Summary", value=True)
    show_visualizations = st.checkbox("Show Visualizations", value=True)

# Load data
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = Reader(uploaded_file).read_csv()
            else:
                df = Reader(uploaded_file).read_excel()
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
        analyzer = Analyzer(df)
        visualizer = Visualizer(df)
        
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
        
        # LabFit Analysis
        st.header("LabFit Analysis")
        
        # Display LabFit Constants
        with st.expander("LabFit Constants"):
            st.write(LABFIT_CONSTANTS)
        
        # Model Fitting Section
        st.subheader("Flow Analysis")
        
        # Select features for regression
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X variable", numeric_cols, index=0)
            with col2:
                y_col = st.selectbox("Select Y variable", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            # Add analysis options
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Polynomial Regression", "Linear Regression", "Decision Tree"],
                index=0
            )
            
            if analysis_type == "Polynomial Regression":
                degree = st.slider("Polynomial Degree", 1, 5, 2)
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model_name = f"Polynomial (deg={degree})"
            elif analysis_type == "Linear Regression":
                model = LinearRegression()
                model_name = "Linear"
            else:  # Decision Tree
                max_depth = st.slider("Max Depth", 1, 10, 3)
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                model_name = f"Decision Tree (max_depth={max_depth})"
            
            try:
                # Use common LabFit variables if available
                x_col_default = 'Differential_Pa' if 'Differential_Pa' in df.columns else x_col
                y_col_default = 'Flow_lph' if 'Flow_lph' in df.columns else y_col
                
                # Prepare data
                X = df[[x_col_default]]
                y = df[y_col_default]
                
                # Handle potential NaN values
                if X.isna().any().any() or y.isna().any():
                    st.warning("Warning: NaN values found in the selected columns. Rows with NaN values will be dropped.")
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X = X[mask]
                    y = y[mask]
                
                # Convert to numpy arrays
                X = X.values.reshape(-1, 1)
                y = y.values
                
                # Fit model
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Calculate residuals
                residuals = y - y_pred
                
                # Create figure with subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(
                        f"{model_name} Fit: {y_col_default} vs {x_col_default}",
                        "Residual Plot"
                    ),
                    column_widths=[0.7, 0.3]
                )
                
                # Add scatter plot with actual data
                fig.add_trace(
                    go.Scatter(
                        x=X.flatten(),
                        y=y,
                        mode='markers',
                        name='Actual',
                        marker=dict(
                            color='blue',
                            opacity=0.7,
                            size=8,
                            line=dict(width=1, color='DarkSlateGrey')
                        )
                    ),
                    row=1, col=1
                )
                
                # Add regression line
                x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                if hasattr(model, 'predict'):
                    y_range = model.predict(x_range)
                else:
                    y_range = model.predict(x_range.reshape(-1, 1))
                    
                fig.add_trace(
                    go.Scatter(
                        x=x_range.flatten(),
                        y=y_range,
                        mode='lines',
                        name=f'{model_name} Fit',
                        line=dict(
                            color='red',
                            width=3,
                            dash='solid'
                        )
                    ),
                    row=1, col=1
                )
                
                # Add residuals with hover information
                fig.add_trace(
                    go.Scatter(
                        x=y_pred,
                        y=residuals,
                        mode='markers',
                        name='Residuals',
                        marker=dict(
                            color='green',
                            size=8,
                            opacity=0.7,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        customdata=np.stack((X.flatten(), y), axis=-1),
                        hovertemplate=
                            'Predicted: %{x:.2f}<br>' +
                            'Residual: %{y:.2f}<br>' +
                            f'{x_col_default}: %{{customdata[0]:.2f}}<br>' +
                            f'{y_col_default}: %{{customdata[1]:.2f}}' +
                            '<extra></extra>'
                    ),
                    row=1, col=2
                )
                
                # Add zero line for residuals
                fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
                
                # Update layout
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    xaxis_title=x_col_default,
                    yaxis_title=y_col_default,
                    xaxis2_title="Predicted Values",
                    yaxis2_title="Residuals",
                    margin=dict(l=50, r=50, t=80, b=80),
                    hovermode='closest',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate model metrics
                r2 = model.score(X, y)
                mse = np.mean(residuals**2)
                mae = np.mean(np.abs(residuals))
                rmse = np.sqrt(mse)
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}", 
                            help="Coefficient of determination. Closer to 1 is better.")
                with col2:
                    st.metric("Mean Squared Error", f"{mse:.4f}",
                            help="Average squared difference between actual and predicted values.")
                with col3:
                    st.metric("Root Mean Squared Error", f"{rmse:.4f}",
                            help="Square root of MSE. In the same units as the response variable.")
                with col4:
                    st.metric("Mean Absolute Error", f"{mae:.4f}",
                            help="Average absolute difference between actual and predicted values.")
                
                # Residual analysis
                st.subheader("Residual Analysis")
                
                # Normality plot
                fig_qq = go.Figure()
                qq = stats.probplot(residuals, dist="norm")
                x = np.array([qq[0][0][0], qq[0][0][-1]])
                
                fig_qq.add_trace(go.Scatter(
                    x=qq[0][0],
                    y=qq[0][1],
                    mode='markers',
                    name='Residuals'
                ))
                
                fig_qq.add_trace(go.Scatter(
                    x=x,
                    y=qq[1][1] + qq[1][0] * x,
                    mode='lines',
                    name='Normal'
                ))
                
                fig_qq.update_layout(
                    title='Q-Q Plot of Residuals',
                    xaxis_title='Theoretical Quantiles',
                    yaxis_title='Sample Quantiles'
                )
                
                st.plotly_chart(fig_qq, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in model fitting: {str(e)}")
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        
        # Add correlation options
        st.markdown("### Variable Relationships")
        
        # Select variables for correlation analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_var1 = st.selectbox("First Variable", numeric_cols, index=0)
            corr_var2 = st.selectbox("Second Variable", numeric_cols, 
                                   index=1 if len(numeric_cols) > 1 else 0)
            
            # Calculate correlation
            corr_value = df[corr_var1].corr(df[corr_var2])
            
            # Display correlation metric
            st.metric(
                f"Correlation between {corr_var1} and {corr_var2}",
                f"{corr_value:.4f}",
                help=f"Correlation ranges from -1 to 1. "
                     f"Values close to 1 or -1 indicate a strong relationship."
            )
            
            # Scatter plot of the two variables
            fig_scatter = px.scatter(
                df, 
                x=corr_var1, 
                y=corr_var2,
                title=f"{corr_var2} vs {corr_var1}",
                trendline="ols",
                trendline_color_override="red"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### Correlation Heatmap")
        try:
            # Get numeric columns only
            numeric_df = df.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                # Calculate correlation matrix
                corr = numeric_df.corr()
                
                # Create the heatmap
                fig = px.imshow(
                    corr,
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    aspect="auto"
                )
                
                # Update layout for better readability
                fig.update_layout(
                    title="Feature Correlation Heatmap",
                    xaxis=dict(tickangle=45),
                    width=800,
                    height=600
                )
                
                # Show correlation values on hover
                fig.update_traces(
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 14}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No numeric columns found for correlation analysis.")
                
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")

if __name__ == "__main__":
    main()
