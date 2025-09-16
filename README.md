# Inclusive Energy Analytics Platform

This repository contains a comprehensive data processing, analysis, and visualization platform for Inclusive Energy. The system provides an interactive web interface for exploring and analyzing energy data, along with robust backend processing capabilities.

## Features

- **Interactive Dashboard**: Streamlit-based web interface for data exploration
- **Advanced Analytics**: Regression analysis, correlation studies, and statistical modeling
- **LabFit Integration**: Built-in support for LabFit data analysis
- **Data Visualization**: Interactive plots and charts using Plotly
- **Machine Learning**: Support for various regression models (Linear, Polynomial, Decision Trees)
- **Data Management**: Tools for data cleaning, transformation, and analysis

## Project Structure

```
inclusiveEnergy/
├── dashboard/              # Streamlit dashboard application
│   └── app.py             # Main dashboard application
├── inclusivePackage/       # Core Python package
│   ├── src/               # Source code
│   │   ├── data_manager.py     # Data management utilities
│   │   ├── data_transformer.py # Data transformation logic
│   │   └── ...
│   └── tests/             # Unit and integration tests
├── experimental/          # Experimental code and notebooks
├── data/                  # Data storage directory (not versioned)
├── requirements-dashboard.txt  # Dashboard dependencies
└── setup.py               # Package installation script
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lastmileICT/venturi-testing.git
   cd venturi-testing
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the core package in development mode:
   ```bash
   pip install -e .
   ```

4. Install dashboard dependencies:
   ```bash
   pip install -r requirements-dashboard.txt
   ```

## Running the Dashboard

To start the interactive dashboard:

```bash
cd dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## Usage

### Using the Dashboard

The dashboard provides a user-friendly interface for data analysis:

1. **Data Upload**: Upload your CSV/Excel file or use the sample data
2. **Data Exploration**: View data summary, statistics, and visualizations
3. **Model Fitting**: Fit various regression models to your data
4. **Correlation Analysis**: Explore relationships between variables

### Programmatic Usage

You can also use the core package programmatically:

```python
import pandas as pd
from inclusivePackage.src import data_transformer

# Load and transform data
df = pd.read_csv('path/to/your/data.csv')
transformer = data_transformer.FlowTransformer(df)

# Convert flow from LPH to LPM
flow_lpm = transformer.convert_flow_lph_to_lpm()

# Perform data analysis
# ...
```

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### Running Tests

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please open an issue in the repository.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact the repository maintainers.
