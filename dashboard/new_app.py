import os
import sys
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inclusivePackage.src.data_transformer import FlowTransformer
from inclusivePackage.src.data_reader import Reader
from inclusivePackage.src.data_analyzer import Analyzer
from inclusivePackage.src.data_visualizer import Visualizer
from inclusivePackage.src.config import LABFIT_CONSTANTS


# Simple transformer class to avoid import errors
class FlowTransformer:
    def __init__(self, df):
        self.df = df
        
    def convert_flow_lph_to_lpm(self):
        """Convert flow from LPH to LPM (divide by 60)"""
        if 'Flow_lph' in self.df.columns:
            return self.df['Flow_lph'] / 60
        else:
            # Return first numeric column if Flow_lph doesn't exist
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            return self.df[numeric_cols[0]] if len(numeric_cols) > 0 else pd.Series([0])

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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
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

@st.cache_data
def create_sample_data():
    """Create sample data with caching for better performance"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible results
    
    data = {
        'date': dates,
        'energy_consumption': 100 + np.cumsum(np.random.normal(2, 5, 100)),
        'temperature': 20 + 10 * np.sin(np.arange(100) * 2 * np.pi / 365) + np.random.normal(0, 2, 100),
        'carbon_emissions': 50 + np.cumsum(np.random.normal(1.5, 3, 100)),
        'Flow_lph': 240 + np.random.normal(0, 20, 100),  # Added Flow_lph for testing
        'Differential_Pa': 50 + np.random.normal(0, 10, 100),  # Added differential pressure
        'Static_Pressure': 1000 + np.random.normal(0, 50, 100)  # Added static pressure
    }
    return pd.DataFrame(data)

@st.cache_data
def load_file_data(file_path, file_name):
    """Load file data with caching using tempfile for better memory management"""
    try:
        # Create a temporary file to store the uploaded content
        file_ext = Path(file_name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(file_path)
            tmp_file_path = tmp_file.name
        
        try:
            # Use the temporary file path with the Reader
            if file_name.endswith('.csv'):
                df = Reader(tmp_file_path).read_csv()
            else:  # Excel file
                df = Reader(tmp_file_path).read_excel()
            return df, None
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"Warning: Could not delete temporary file: {e}")
    except Exception as e:
        return None, str(e)

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

# Load data function
def load_data():
    """Load data from file or create sample data"""
    if uploaded_file is not None:
        try:
            df, error = load_file_data(uploaded_file.getvalue(), uploaded_file.name)
            if error:
                st.error(f"Error loading file: {error}")
                return None
            st.session_state.df = df
            st.success("Data loaded successfully!")
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    elif use_sample:
        try:
            df = create_sample_data()
            st.session_state.df = df
            return df
        except Exception as e:
            st.error(f"Error generating sample data: {e}")
            return None
    return None


def safe_model_fitting(X, y, model, model_name):
    """Safely fit model and return results"""
    try:
        # Ensure we have valid data
        if len(X) == 0 or len(y) == 0:
            return None, None, "No valid data points found"
            
        # Fit model
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = model.score(X, y)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(mse)
        
        # Create predicted vs actual plot
        fig = go.Figure()
        
        # Add scatter plot of predicted vs actual
        fig.add_trace(go.Scatter(
            x=y,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6, size=8),
            hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
        ))
        
        # Add perfect prediction line
        min_val = min(np.min(y), np.min(y_pred))
        max_val = max(np.max(y), np.max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Predicted vs Actual Values - {model_name}',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            showlegend=True,
            height=600
        )
        
        metrics = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'residuals': residuals,
            'y_pred': y_pred,
            'prediction_plot': fig  # Include the figure in metrics
        }
        
        return model, metrics, None
        
    except Exception as e:
        return None, None, str(e)




def create_model_visualization(X, y, model, metrics, x_cols, y_col_display, model_name):
    """Create appropriate visualization based on number of features"""
    try:
        if len(x_cols) == 1:
            # 2D plot for single feature
            fig = make_subplots(
                rows=1, cols=2,
                column_widths=[0.7, 0.3],
                subplot_titles=('Model Fit', 'Residuals')
            )
            
            # Add actual data points
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=y,
                    mode='markers',
                    name='Actual',
                    marker=dict(color='blue', opacity=0.6, size=6),
                    hovertemplate=f'{x_cols[0]}: %{{x:.2f}}<br>' +
                                f'{y_col_display}: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Generate prediction line
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            X_range = x_range.reshape(-1, 1)
            y_range = model.predict(X_range)
            
            # Add regression line
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=f'{model_name} Fit',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Add residuals plot
            fig.add_trace(
                go.Scatter(
                    x=metrics['y_pred'],
                    y=metrics['residuals'],
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='green', opacity=0.6, size=6),
                    hovertemplate='Predicted: %{x:.2f}<br>' +
                                'Residual: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
            
            # Update layout
            fig.update_layout(
                height=500,
                xaxis_title=x_cols[0],
                yaxis_title=y_col_display,
                xaxis2_title='Predicted Values',
                yaxis2_title='Residuals',
                showlegend=True,
                margin=dict(l=50, r=50, t=80, b=80)
            )
            
            return fig
            
        elif len(x_cols) == 2:
            # 3D plot for two features
            fig = go.Figure()
            
            # Create surface
            x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
            x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
            x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
            
            X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
            y_grid = model.predict(X_grid).reshape(x1_grid.shape)
            
            # Add surface
            fig.add_trace(
                go.Surface(
                    x=x1_grid,
                    y=x2_grid,
                    z=y_grid,
                    name=f'{model_name} Fit',
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=False
                )
            )
            
            # Add scatter points
            fig.add_trace(
                go.Scatter3d(
                    x=X[:, 0],
                    y=X[:, 1],
                    z=y,
                    mode='markers',
                    name='Actual',
                    marker=dict(size=4, color='blue', opacity=0.8)
                )
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_cols[0],
                    yaxis_title=x_cols[1],
                    zaxis_title=y_col_display
                ),
                margin=dict(l=50, r=50, t=80, b=80)
            )
            
            return fig
            
        else:
            # Multiple features - show predicted vs actual
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=metrics['y_pred'],
                    y=y,
                    mode='markers',
                    name='Predicted vs Actual',
                    marker=dict(color='blue', opacity=0.6, size=6),
                    hovertemplate='Predicted: %{x:.2f}<br>' +
                                f'Actual: %{{y:.2f}}<extra></extra>'
                )
            )
            
            # Add perfect prediction line
            min_val = min(metrics['y_pred'].min(), y.min())
            max_val = max(metrics['y_pred'].max(), y.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                title='Predicted vs Actual Values',
                xaxis_title='Predicted Values',
                yaxis_title=f'Actual {y_col_display}',
                showlegend=True
            )
            
            return fig
            
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


# display metrics...
def display_metrics(metrics):
    """Display model performance metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "R² Score", 
            f"{metrics['r2']:.4f}",
            help="Coefficient of determination. Closer to 1 is better."
        )
    with col2:
        st.metric(
            "RMSE", 
            f"{metrics['rmse']:.4f}",
            help="Root Mean Squared Error. Lower is better."
        )
    with col3:
        st.metric(
            "MAE", 
            f"{metrics['mae']:.4f}",
            help="Mean Absolute Error. Lower is better."
        )
    with col4:
        st.metric(
            "MSE", 
            f"{metrics['mse']:.4f}",
            help="Mean Squared Error. Lower is better."
        )


# Main app logic
def main():
    # Load data
    df = load_data()
    
    if df is None:
        st.info("Please upload a file or use the sample data to get started.")
        return
    
    # Show data summary
    if show_summary and df is not None:
        st.header("📋 Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Data Information")
            info_data = {
                "Metric": ["Rows", "Columns", "Memory Usage"],
                "Value": [
                    f"{len(df):,}",
                    len(df.columns),
                    f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
                ]
            }
            st.table(pd.DataFrame(info_data))
            
            # Data types
            st.subheader("Column Types")
            type_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes],
                'Non-Null': [df[col].count() for col in df.columns]
            })
            st.dataframe(type_info, use_container_width=True)
    
    # Data Analysis
    if df is not None and show_visualizations:
        st.header("📊 Data Analysis")
        
        # Time series visualization
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if 'date' in df.columns and numeric_cols:
            st.subheader("Time Series Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", ['date'], index=0)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols, index=0)
            
            try:
                fig = px.line(
                    df, 
                    x=x_axis, 
                    y=y_axis, 
                    title=f"{y_axis} over Time",
                    markers=True
                )
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating time series plot: {e}")
        
        # Summary statistics
        if numeric_cols:
            st.subheader("Statistical Summary")
            summary_stats = df[numeric_cols].describe()
            st.dataframe(summary_stats, use_container_width=True)
        
        # LabFit Analysis
        st.header("🔬 LabFit Analysis")
        
        # Display LabFit Constants
        with st.expander("LabFit Constants", expanded=False):
            const_df = pd.DataFrame(list(LABFIT_CONSTANTS.items()), 
                                  columns=['Parameter', 'Value'])
            st.table(const_df)
        
        # Model Fitting Section
        if len(numeric_cols) >= 2:
            st.subheader("Model Fitting & Analysis")
            
            # Feature selection
            col1, col2 = st.columns(2)
            
            with col1:
                # Y variable selection
                default_y = 'Flow_lph' if 'Flow_lph' in df.columns else numeric_cols[0]
                y_col = st.selectbox("Target Variable (Y)", numeric_cols, 
                                   index=numeric_cols.index(default_y) if default_y in numeric_cols else 0)
            
            with col2:
                # X variables selection
                available_x = [col for col in numeric_cols if col != y_col]
                default_x = [col for col in ['Differential_Pa', 'Static_Pressure'] if col in available_x]
                if not default_x:
                    default_x = available_x[:1] if available_x else []
                
                x_cols = st.multiselect(
                    "Features (X)", 
                    available_x,
                    default=default_x
                )
            
            if not x_cols:
                st.warning("Please select at least one feature for X.")
                return
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                analysis_type = st.selectbox(
                    "Model Type",
                    ["Linear Regression", "Polynomial Regression", "Decision Tree"],
                    index=0
                )
            
            with col2:
                if analysis_type == "Polynomial Regression":
                    degree = st.slider("Polynomial Degree", 1, 5, 2)
                elif analysis_type == "Decision Tree":
                    max_depth = st.slider("Max Depth", 1, 10, 3)
            
            # Prepare data
            X = df[x_cols].values
            
            # Handle y variable transformation
            if y_col == 'Flow_lph':
                transformer = FlowTransformer(df)
                y = transformer.convert_flow_lph_to_lpm().values
                y_col_display = 'Flow_lpm'
            else:
                y = df[y_col].values
                y_col_display = y_col
            
            # Remove NaN values
            mask = ~(pd.DataFrame(X).isna().any(axis=1) | pd.Series(y).isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 2:
                st.error("Not enough valid data points for analysis.")
                return
            
            # Create model
            if analysis_type == "Polynomial Regression":
                model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                model_name = f"Polynomial (deg={degree})"
            elif analysis_type == "Linear Regression":
                model = LinearRegression()
                model_name = "Linear"
            else:  # Decision Tree
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
                model_name = f"Decision Tree (max_depth={max_depth})"
            
            # Fit model
            fitted_model, metrics, error = safe_model_fitting(X, y, model, model_name)
            
            if error:
                st.error(f"Error fitting model: {error}")
                return
            
            # Display results
            st.subheader("Model Results")
            
            # Show metrics
            display_metrics(metrics)
            
            # Show predicted vs actual plot
            st.subheader("Predicted vs Actual Values")
            st.plotly_chart(metrics['prediction_plot'], use_container_width=True)
            
            # Show model visualization
            fig = create_model_visualization(X, y, fitted_model, metrics, x_cols, y_col_display, model_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Residual analysis
            if len(metrics['residuals']) > 3:
                st.subheader("Residual Analysis")
                
                try:
                    # Q-Q plot
                    qq = stats.probplot(metrics['residuals'], dist="norm")
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(
                        x=qq[0][0],
                        y=qq[0][1],
                        mode='markers',
                        name='Residuals',
                        marker=dict(color='blue', size=6)
                    ))
                    
                    # Add theoretical line
                    x_line = np.array([qq[0][0][0], qq[0][0][-1]])
                    y_line = qq[1][1] + qq[1][0] * x_line
                    
                    fig_qq.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        name='Theoretical Normal',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_qq.update_layout(
                        title='Q-Q Plot of Residuals (Normality Check)',
                        xaxis_title='Theoretical Quantiles',
                        yaxis_title='Sample Quantiles',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_qq, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Could not create Q-Q plot: {str(e)}")
        
        # Correlation Analysis
        if len(numeric_cols) >= 2:
            st.subheader("📈 Correlation Analysis")
            
            # Individual correlation
            col1, col2 = st.columns(2)
            with col1:
                corr_var1 = st.selectbox("First Variable", numeric_cols, index=0, key="corr1")
            with col2:
                corr_var2 = st.selectbox("Second Variable", numeric_cols, 
                                       index=1 if len(numeric_cols) > 1 else 0, key="corr2")
            
            if corr_var1 != corr_var2:
                try:
                    # Calculate correlation
                    corr_value = df[corr_var1].corr(df[corr_var2])
                    
                    # Display correlation
                    st.metric(
                        f"Correlation: {corr_var1} vs {corr_var2}",
                        f"{corr_value:.4f}",
                        help="Correlation ranges from -1 to 1. Values close to ±1 indicate strong relationships."
                    )
                    
                    # Scatter plot
                    fig_scatter = px.scatter(
                        df, 
                        x=corr_var1, 
                        y=corr_var2,
                        title=f"{corr_var2} vs {corr_var1}",
                        trendline="ols"
                    )
                    fig_scatter.update_layout(showlegend=True)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error in correlation analysis: {str(e)}")
            
            # Correlation heatmap
            try:
                st.markdown("### Correlation Heatmap")
                numeric_df = df[numeric_cols]
                
                if len(numeric_df.columns) >= 2:
                    corr_matrix = numeric_df.corr()
                    
                    fig_heatmap = px.imshow(
                        corr_matrix,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        color_continuous_scale='RdBu_r',
                        zmin=-1,
                        zmax=1,
                        aspect="auto",
                        text_auto='.2f'
                    )
                    
                    fig_heatmap.update_layout(
                        title="Feature Correlation Matrix",
                        xaxis=dict(tickangle=45),
                        height=600
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating correlation heatmap: {str(e)}")

if __name__ == "__main__":
    main()
