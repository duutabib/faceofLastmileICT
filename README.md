# Inclusive Energy Data Pipeline

This repository contains a comprehensive data processing and analysis pipeline developed for Inclusive Energy. The system streamlines data acquisition, processing, and analysis, eliminating the need for manual data movement between Excel and analysis tools like LabFit.

## Features

- **Data Management**: Robust data handling and manipulation utilities
- **Data Analysis**: Advanced analytics and modeling capabilities
- **Visualization**: Tools for generating insightful visualizations
- **LabFit Integration**: Seamless integration with LabFit for data fitting
- **Venting Algorithm**: Specialized algorithm for venting analysis
- **Carbon Credit Monitoring**: Tools for tracking and analyzing carbon credits

## Project Structure

```
inclusivePackage/
├── src/
│   ├── data_manager.py    # Core data management utilities
│   ├── data_analyzer.py   # Data analysis and modeling
│   ├── data_visualizer.py # Data visualization tools
│   ├── data_retriever.py  # Data retrieval from various sources
│   ├── data_transformer.py # Data transformation utilities
│   └── config.py          # Configuration settings
├── tests/                 # Unit and integration tests
└── data/                  # Data storage directory (not versioned)
```

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:lastmileICT/venturi-testing.git
   cd venturi-testing
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

### Basic Data Operations

```python
import pandas as pd
from inclusivePackage.src import data_manager

# Load and manage data
df = pd.read_csv('path/to/your/data.csv')

# Use the custom manager accessor
df.manager.clean_data()
df.manager.deduplicate()

# Access processed data
processed_df = df.manager.get_processed_data()
```

### Data Analysis

```python
from inclusivePackage.src import data_analyzer

# Initialize analyzer
analyzer = data_analyzer.DataAnalyzer(processed_df)

# Run analysis
results = analyzer.analyze()
summary = analyzer.get_summary_statistics()
```

### Visualization

```python
from inclusivePackage.src import data_visualizer

# Create visualizations
visualizer = data_visualizer.DataVisualizer(processed_df)
visualizer.plot_timeseries(x='date', y='value', title='Time Series Analysis')
visualizer.save_plot('timeseries_plot.png')
```

### Configuration

Modify `config.py` to set up:
- Data source paths
- Output directories
- Analysis parameters
- Visualization settings

## Running Tests

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact the repository maintainers.
