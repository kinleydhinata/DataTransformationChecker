# Data Transformation Checker

## Overview

The **Data Transformation Checker** is a Python utility designed to analyze, transform, and visualize datasets for normality. It offers a range of transformation methods, conducts normality tests (Shapiro-Wilk and Lilliefors Kolmogorov-Smirnov), and provides insightful visualizations such as histograms and Q-Q plots.

This tool is ideal for data scientists, statisticians, and anyone working with numerical datasets who need to preprocess data for statistical analysis or machine learning models.

---

## Features

- **Transformation Methods**:
  - Logarithmic (Log10, Natural Log)
  - Square Root
  - Reciprocal
  - Box-Cox (only for positive values)
  - Yeo-Johnson (handles negative values)
  - Quantile Transform
  - Z-Score Standardization
  - Rank Inverse Transformation
  - Cubic Root
  - Sigmoid
  - Exponential

- **Normality Tests**:
  - Shapiro-Wilk Test
  - Lilliefors Test (Kolmogorov-Smirnov with estimated parameters)

- **Visualizations**:
  - Histograms with KDE
  - Q-Q Plots

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DataTransformationChecker.git
   ```

2. Navigate to the project directory:
   ```bash
   cd DataTransformationChecker
   ```

3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Input Data
The tool allows users to input numerical data directly via the console. Enter one number per line, and press Enter twice to finish.

### 2. Analyze Transformations
The script automatically applies multiple transformations to the data and tests each for normality using Shapiro-Wilk and Lilliefors tests.

### 3. Visualize Results
The results are displayed in a tabular format, showing:
- Transformation name
- Whether the data passed the normality test
- Shapiro-Wilk statistic and p-value
- Lilliefors statistic and p-value

Additionally, histograms and Q-Q plots are generated for the original and transformed datasets.

### Run the Script
```bash
python transform.py
```

---

## Example Output

### Console Output
```
Normality Test Results:
--------------------------------------------------
Transformation           Normal?   Shapiro Stat   Shapiro p      KS Stat        KS p           
--------------------------------------------------
Original Data            no        0.9377         0.0000         0.0856         0.0000         
Log10                    no        0.7697         0.0000         0.1509         0.0000         
Square Root              no        0.8815         0.0000         0.1147         0.0000         
Reciprocal               no        0.4061         0.0000         0.2729         0.0000         
Log (Natural)            no        0.6949         0.0000         0.1705         0.0000         
Box-Cox (λ=1.486)        no        0.9453         0.0000         0.0597         0.0000         
Yeo-Johnson (λ=1.493)    no        0.9451         0.0000         0.0596         0.0000         
Quantile                 yes       0.9973         0.0034         0.0053         0.2000         
Z-Score                  no        0.9377         0.0000         0.0856         0.0000         
Rank Inverse             yes       0.9999         1.0000         0.0049         0.2000         
Cubic Root               no        0.8509         0.0000         0.1263         0.0000         
Sigmoid                  no        0.0146         0.0000         0.5049         0.0000         
Exponential              no        0.0073         0.0000         0.5089         0.0000     
```

### Visualization
The script generates the following plots:
- **Original Data**: Histogram and Q-Q plot
- **Transformed Data**: Histogram and Q-Q plot for each transformation that passes normality

---

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Scikit-learn
- Seaborn
- Matplotlib

---
