# Inclusive Energy Dashboard

An interactive Streamlit dashboard for analyzing and visualizing energy data from the Inclusive Energy project.

## Features

- **Data Upload**: Upload your own CSV or Excel files, or use sample data
- **Data Exploration**: View and explore your dataset with an interactive table
- **Time Series Analysis**: Visualize trends over time with interactive plots
- **Statistical Summary**: Get key statistics about your data
- **Correlation Analysis**: Explore relationships between numerical features

## Installation

1. Make sure you have Python 3.8+ installed
2. Clone the repository if you haven't already:
   ```bash
   git clone git@github.com:lastmileICT/venturi-testing.git
   cd venturi-testing
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements-dashboard.txt
   ```

## Running the Dashboard

1. Navigate to the project root directory
2. Run the Streamlit app:
   ```bash
   streamlit run dashboard/app.py
   ```
3. The dashboard will open automatically in your default web browser

## Usage

1. **Data Input**
   - Use the sidebar to upload your data file (CSV or Excel)
   - Or check "Use sample data" to work with example data

2. **Data Exploration**
   - View the first few rows of your data
   - See data types and basic statistics

3. **Visualizations**
   - Create time series plots by selecting X and Y axes
   - View correlation heatmaps between numerical features
   - Toggle different visualizations using the sidebar options

## Customization

You can customize the dashboard by modifying `app.py`. Some ideas:
- Add new visualizations
- Include additional data processing steps
- Modify the layout and styling
- Add more interactive components

## Dependencies

- streamlit
- pandas
- plotly
- numpy
- matplotlib

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
