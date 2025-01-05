# AI-Projects-Beginner
This repository showcases a collection of beginner-level AI projects developed during my AI module in my bachelor's degree. Each project is implemented in Python and explores fundamental concepts in artificial intelligence, including machine learning, data preprocessing, and predictive modelling. The projects aim to demonstrate practical applications of AI techniques while highlighting my early journey into the field.

## Table of contents

### 1. Calories Burnt Prediction 
- CaloriesBurntPrediction.ipynb
- This project involves developing a machine learning model to predict calories burnt based on user-specific features, such as Gender, Age, Height, Weight, Duration of activity, Heart Rate, and Body Temperature. Using Python, the data is analysed and visualised with pandas, Matplotlib, and Seaborn, followed by splitting the dataset into training and testing sets. An XGBoost Regressor is implemented for regression, and model performance is evaluated using Mean Absolute Error.

### 2. Chatbot ROBO
- Chatbot_ROBO.ipynb
- This project is a simple rule-based chatbot named ROBO, designed to answer user queries about chatbots. It uses TF-IDF vectorisation and cosine similarity to generate context-based responses from a provided text corpus. The chatbot can recognise greetings, respond interactively, and handle basic conversations. Built with Python, it leverages libraries like NLTK for preprocessing (tokenisation, lemmatisation) and scikit-learn for text vectorisation.

### 3. Diabetes Prediction
- DiabetesPrediction.ipynb
- This project focuses on predicting the likelihood of diabetes based on various health features, including Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age. Using Python, the data is preprocessed by standardising the features with StandardScaler and splitting the dataset into training and testing sets. A Support Vector Machine (SVM) model is trained and evaluated for accuracy using the accuracy_score metric.

### 4. Diamond Price Forecasting 
- DiamondPriceForecasting.ipynb
- This project aims to forecast diamond prices based on features such as carat, cut, color, clarity, depth, table, and x, y, and z measurements. The dataset is analysed through Exploratory Data Analysis (EDA) and visualised using libraries like Seaborn and Matplotlib. Categorical data is transformed into numerical form using Label Encoding. A Random Forest Regressor model is trained to predict diamond prices, with performance evaluated using metrics like Mean Absolute Error and Mean Squared Error.

### 5. Insurance Charge Prediction 
- InsuranceChargePrediction.ipynb
- This project focuses on predicting insurance charges based on features such as age, sex, BMI, children, smoker, and region. Through Exploratory Data Analysis (EDA) and visualisations with Seaborn and Matplotlib, the dataset is thoroughly examined. A Linear Regression model is applied for prediction, with performance assessed using the R-squared value, which indicates the proportion of variance explained by the model. This project demonstrates the use of regression techniques and data visualisation to predict insurance costs.

### 6. Movie Recommendation System
- MovieRecommendationSystem.ipynb
- This project involves developing a simple movie recommendation system based on key features such as genres, keywords, tagline, cast, and director. The system uses TF-IDF Vectorization to convert these textual features into numerical form, then calculates cosine similarity to assess how similar movies are to one another. By leveraging difflib for string matching, the recommendation engine suggests movies based on user input. This project demonstrates how text-based features can be used to build a simple and effective movie recommendation system.
