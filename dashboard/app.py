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
    from src.data_transformer import Transformer, FlowTransformer
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
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv' if uploaded_file.name.endswith('.csv') else '.xlsx') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Read the file using Reader
            if uploaded_file.name.endswith('.csv'):
                df = Reader(tmp_file_path).read_csv()
            else:
                df = Reader(tmp_file_path).read_excel()
            
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
                
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
                # Let user select multiple features for X
                st.sidebar.subheader("Feature Selection")
                available_features = [col for col in df.select_dtypes(include=['number']).columns 
                                   if col != y_col and df[col].count() > 0]
                
                # Default features if they exist
                default_features = [col for col in ['Differential_Pa', 'Static Pressure'] 
                                 if col in available_features]
                
                # If no default features, use the first numeric column
                if not default_features and available_features:
                    default_features = [available_features[0]]
                
                # Multi-select widget for features
                x_cols = st.sidebar.multiselect(
                    "Select features for X (independent variables):",
                    options=available_features,
                    default=default_features,
                    format_func=lambda x: f"{x}"
                )
                
                # If no features selected, use defaults
                if not x_cols and default_features:
                    x_cols = default_features
                elif not x_cols and available_features:
                    x_cols = [available_features[0]]
                
                # Set default y column
                y_col_default = 'Flow_lph' if 'Flow_lph' in df.columns else y_col
                
                # Convert flow from LPH to LPM if needed
                if y_col_default == 'Flow_lph':
                    transformer = FlowTransformer(df)
                    y = transformer.convert_flow_lph_to_lpm()
                    y_col_display = 'Flow_lpm'  # Update column name for display
                else:
                    y = df[y_col_default]
                    y_col_display = y_col_default
                
                # Prepare features
                X = df[x_cols]
                
                # Handle potential NaN values
                if X.isna().any().any() or y.isna().any():
                    st.warning("Warning: NaN values found in the selected columns. Rows with NaN values will be dropped.")
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X = X[mask]
                    y = y[mask]
                
                # Convert to numpy arrays
                X_values = X.values
                y_values = y.values
                
                # Fit model
                model.fit(X_values, y_values)
                y_pred = model.predict(X_values)
                
                # Calculate residuals
                residuals = y_values - y_pred
                
                # Create subplots
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(
                        f"{model_name} Fit: {y_col_default} vs {x_cols[0]}",
                        "Residual Plot"
                    ),
                    column_widths=[0.7, 0.3]
                )

                # Plot actual vs predicted (using first feature for x-axis)
                fig.add_trace(
                    go.Scatter(
                        x=X_values[:, 0],  # Use first feature for x-axis
                        y=y_values,
                        mode='markers',
                        name='Actual',
                        marker=dict(color='blue', opacity=0.6),
                        customdata=np.column_stack((X_values, y_values)),
                        hovertemplate=
                            f'Actual {y_col_display}: %{{y:.2f}}<br>' +
                            '<br>'.join([f'{col}: %{{customdata[0][' + str(i) + ']:.2f}' for i, col in enumerate(x_cols)]) +
                            '<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add regression surface or line
                if len(x_cols) == 1:
                    # For single feature, show regression line
                    x1_range = np.linspace(X_values[:, 0].min(), X_values[:, 0].max(), 100)
                    X_range = x1_range.reshape(-1, 1)
                    y_range = model.predict(X_range)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x1_range,
                            y=y_range,
                            mode='lines',
                            name=f'{model_name} Fit',
                            line=dict(color='red', width=2)
                        ),
                        row=1, col=1
                    )
                else:
                    # For multiple features, show a 3D surface or multiple lines
                    if len(x_cols) == 2:
                        # For 2 features, show a 3D surface
                        x1_range = np.linspace(X_values[:, 0].min(), X_values[:, 0].max(), 20)
                        x2_range = np.linspace(X_values[:, 1].min(), X_values[:, 1].max(), 20)
                        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
                        
                        # Create grid of points for prediction
                        X_grid = np.column_stack([
                            x1_grid.ravel(),
                            x2_grid.ravel(),
                            # Add means for any additional features
                            *[np.full_like(x1_grid.ravel(), X_values[:, i].mean()) 
                              for i in range(2, len(x_cols))]
                        ])
                        
                        # Make predictions and reshape for surface
                        y_grid = model.predict(X_grid).reshape(x1_grid.shape)
                        
                        # Add 3D surface
                        fig.add_trace(
                            go.Surface(
                                x=x1_grid,
                                y=x2_grid,
                                z=y_grid,
                                name=f'{model_name} Fit',
                                colorscale='Viridis',
                                opacity=0.7,
                                showscale=False
                            ),
                            row=1, col=1
                        )
                        
                        # Add scatter points on top
                        fig.add_trace(
                            go.Scatter3d(
                                x=X_values[:, 0],
                                y=X_values[:, 1],
                                z=y_values,
                                mode='markers',
                                name='Actual',
                                marker=dict(size=4, color='blue', opacity=0.8),
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        # Update layout for 3D
                        fig.update_layout(
                            scene=dict(
                                xaxis_title=x_cols[0],
                                yaxis_title=x_cols[1],
                                zaxis_title=y_col_display
                            )
                        )
                    else:
                        # For more than 2 features, show multiple lines for the first two features
                        # while holding other features at their mean
                        x1_range = np.linspace(X_values[:, 0].min(), X_values[:, 0].max(), 100)
                        
                        # Create grid with first feature varying, second feature at 3 different levels,
                        # and other features at their mean
                        for x2_val in np.percentile(X_values[:, 1], [25, 50, 75]):
                            X_range = np.column_stack([
                                x1_range,
                                np.full_like(x1_range, x2_val),
                                *[np.full_like(x1_range, X_values[:, i].mean()) 
                                  for i in range(2, len(x_cols))]
                            ])
                            
                            y_range = model.predict(X_range)
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=x1_range,
                                    y=y_range,
                                    mode='lines',
                                    name=f'{model_name} ({x_cols[1]} = {x2_val:.2f})',
                                    line=dict(width=2)
                                ),
                                row=1, col=1
                            )
                
                # Update layout based on number of features
                if len(x_cols) == 1:
                    fig.update_layout(
                        xaxis_title=x_cols[0],  # Use first feature name for x-axis
                        yaxis_title=y_col_display,  # Use the correct unit (LPM if converted)
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
                        customdata=np.column_stack((X_values, y_values)),
                        hovertemplate=
                            'Predicted: %{x:.2f}<br>' +
                            'Residual: %{y:.2f}<br>' +
                            f'{x_cols[0]}: %{{customdata[0]:.2f}}<br>' +
                            (f'{x_cols[1]}: %{{customdata[1]:.2f}}<br>' if len(x_cols) > 1 else '') +
                            f'{y_col_default}: %{{customdata[{len(x_cols)}]:.2f}}' +
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
                    xaxis_title=x_cols[0],  # Use first feature name for x-axis
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
