# Machine Learning Based Weather Prediction and Dynamic Weather Sensation Index (DWSI)

## Overview
This project is a real-time weather prediction and analysis system developed using Machine Learning techniques and the OpenWeatherMap API. The system predicts temperature for upcoming hours and calculates a Dynamic Weather Sensation Index (DWSI) to estimate user comfort levels based on weather conditions.

The project combines historical weather data, real-time API data, and machine learning models to provide accurate weather insights through an interactive graphical user interface (GUI).

## Features
- Real-time weather data collection using OpenWeatherMap API
- Temperature prediction using XGBoost Machine Learning algorithm
- Dynamic Weather Sensation Index (DWSI) calculation
- Real-time weather condition analysis
- Graphical User Interface (GUI) using Tkinter
- Visualization of historical and predicted weather data
- Intelligent comfort level assessment

## Tech Stack
- Python
- XGBoost
- Tkinter
- OpenWeatherMap API
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Workflow
1. Collect historical and real-time weather data
2. Preprocess and clean the dataset
3. Train the XGBoost machine learning model
4. Predict temperature for upcoming hours
5. Calculate Dynamic Weather Sensation Index (DWSI)
6. Display predictions and analysis through GUI

## Dynamic Weather Sensation Index (DWSI)
The DWSI is a rule-based model that evaluates weather comfort levels using:
- Temperature
- Humidity
- Wind Speed
- Cloudiness
The system applies predefined seasonal weight calculations to determine the comfort index.

## Results
- Temperature Prediction Accuracy: 86.74%
- DWSI Prediction Accuracy: 85.48%
The model performed effectively on historical weather datasets and real-time weather conditions.

