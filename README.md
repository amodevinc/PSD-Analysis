# PSD Failure Analysis Dashboard

This Streamlit application provides an interactive dashboard for visualizing and analyzing the failure probability of Platform Screen Door (PSD) components based on historical data and survival analysis models.

## Purpose

The main goal of this dashboard is to provide insights into the expected lifespan and failure patterns of various PSD components. It allows users to:

*   Visualize how failure probability accumulates over time for different components.
*   Compare the typical lifespan (Median Time To Failure) across components and location types.
*   Generate custom failure predictions for specific scenarios based on component type, location, and station usage (daily train runs).

## Features

*   **Interactive Plots:** Failure probability curves and Median TTF comparison bar charts using Plotly.
*   **Filtering:** Filter data displayed by component type and location type (Overall, Above Ground, Underground).
*   **Custom Prediction Tool:** Generate specific failure predictions by selecting a component, location, and inputting average daily train runs.
*   **Station Search:** Look up stations by Korean or English name to see their average daily runs.
*   **Bilingual Support:** User interface available in both English and Korean.
*   **Methodology Descriptions:** Explanations of the underlying survival analysis techniques (Weibull AFT models) are provided within the app.

## Data Requirements

The dashboard relies on the following files being present in the same directory or specified paths:

1.  `psd_failures_cleaned_filtered.csv`: The main dataset containing historical failure records, component names (Korean and English), location types, station names, and daily run data.
2.  `survival_insights_summary.csv`: Pre-calculated survival probabilities and median TTF values for various component/location groups.
3.  `survival_analysis/component_regression_params.json`: Parameters (coefficients, shape, scale) of the fitted Weibull AFT models for each component.

## Setup

1.  **Clone Repository (if applicable):** Ensure you have the project code.
2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:** Create a `requirements.txt` file with the following content (or use `pip freeze > requirements.txt` in your environment and clean it up):
    ```txt
    streamlit
    pandas
    numpy
    scipy
    plotly
    # Add other specific libraries if used
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place Data Files:** Ensure the required CSV and JSON data files are located correctly relative to `failure_dashboard.py` as defined by the paths in the script (e.g., in the same directory and a `survival_analysis` subdirectory).

## Running the Application

Navigate to the directory containing `failure_dashboard.py` in your terminal (ensure your virtual environment is activated) and run:

```bash
streamlit run failure_dashboard.py
```

This will start the Streamlit server and open the dashboard in your default web browser.

## Methodology Overview

The dashboard utilizes **Survival Analysis**, specifically **Weibull Accelerated Failure Time (AFT) models**, fitted to historical component failure data. These models estimate the time until an event (failure) occurs and how covariates (like location type and station usage) influence this time.

*   **Failure Probability Curves:** Show the cumulative probability of failure up to a given point in time.
*   **Median Time To Failure (Median TTF):** Represents the time by which 50% of components in a group are expected to fail.
*   **Custom Predictions:** Adjust the base model parameters based on user-defined inputs for location and daily runs.

## Limitations

*   **Daily Runs Anomaly:** The current underlying models show an association between higher average daily train runs and *longer* time-to-failure. This is counter-intuitive and likely due to unmeasured **confounding factors** such as maintenance frequency/quality or component/station age. Predictions using the custom tool that rely heavily on the 'Daily Runs' input should be interpreted with this limitation in mind.
*   **Data Quality:** The accuracy of the predictions depends heavily on the quality and completeness of the input data. 