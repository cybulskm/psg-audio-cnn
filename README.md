# PSG-Audio Project

## Overview
The PSG-Audio project is designed for the analysis of polysomnography (PSG) data using machine learning techniques. The project implements a Random Forest model to determine feature importance and a Convolutional Neural Network (CNN) for classification tasks based on the selected features.

## Project Structure
```
PSG-Audio
├── src
│   ├── __init__.py
│   ├── main.py          # Entry point of the application
│   ├── random_forest.py # Contains Random Forest functionality
│   ├── cnn.py          # Defines CNN architecture and training procedures
│   ├── data_loader.py   # Responsible for loading and preprocessing data
│   └── utils.py         # Utility functions used across the project
├── bin                  # Directory for output files and checkpoints
├── config
│   └── config.py       # Configuration settings for the project
├── requirements.txt     # Lists project dependencies
└── README.md            # Documentation for the project
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd PSG-Audio
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**: Ensure that your PSG data is preprocessed and saved in the expected format.
2. **Run the Application**: Execute the main script to start the analysis.

```bash
python src/main.py
```

## Functionality
- **Random Forest**: Computes feature importance and selects the top 25% and top 50% of features for further analysis.
- **CNN**: Trains a Convolutional Neural Network on the selected features and evaluates its performance.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.