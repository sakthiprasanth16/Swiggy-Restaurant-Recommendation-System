Swiggy Restaurant Recommendation System using Streamlit

Overview

This project implements a restaurant recommendation system for Swiggy using Python and Streamlit. It leverages K-Nearest Neighbors (KNN) with cosine similarity to suggest restaurants to users based on their preferences, such as city, cuisine, rating, and cost. The system combines data preprocessing, feature encoding, and similarity-based recommendation to provide personalized results.

The project uses three key data files and pre-trained models:

for_encode&cleaned.csv → Original dataset cleaned for preprocessing.
encoded_data.csv → Dataset with numerical features after one-hot encoding categorical variables.
cleaned_data.csv → Fully cleaned dataset for displaying results.
Encoders (city_encoder.pkl, cuisine_encoder.pkl) and scaler (scaler.pkl) to transform user input.

Features

User Input Options: City, Cuisine, Cost, and Rating.

Recommendation Engine: Uses KNN with cosine similarity to find restaurants similar to the user’s preferences.

Streamlit Interface: Easy-to-use web application for input and interactive results.

Data Preprocessing: Handles missing values, duplicate removal, and feature encoding.

Skills & Technologies

Python

Streamlit for interactive application

Data Preprocessing & Cleaning

One-Hot Encoding of categorical variables

Scaling numerical features

KNN-based similarity recommendation

Data Analytics & Visualization

How It Works

Data Loading
Preloaded CSV files (for_encode&cleaned.csv, encoded_data.csv, cleaned_data.csv) provide the basis for preprocessing and recommendation. Encoders and scaler are loaded to transform user input.

User Input Handling
Users select City, Cuisine, Cost, and Rating. The system handles input encoding using the saved encoders and scales numerical features.

Feature Combination
User input features are combined into a single numerical vector:

[scaled rating & cost] + [encoded city] + [encoded cuisine]

Recommendation Generation
The KNN model computes cosine similarity between the user input vector and all restaurants in encoded_data.csv. The nearest neighbors are returned as recommendations.

Display Results
Recommendations are mapped to cleaned_data.csv to show human-readable information including:

Restaurant Name
City
Cost
Cuisine
Rating
Rating Count
Address

Business Use Cases

Personalized Recommendations: Suggest restaurants matching user preferences.

Customer Experience: Improve decision-making by providing tailored results.

Market Insights: Identify popular cuisines, cities, and price preferences.

Operational Efficiency: Optimize offerings based on customer preference patterns.

Results

Cleaned Dataset: Duplicates removed, missing values handled.

Encoded Dataset: One-hot encoded categorical features, scaled numerical features.

Recommendation Engine: KNN with cosine similarity returns personalized restaurant suggestions.

Streamlit UI: Interactive interface for smooth user experience.
