# AI-Projects-Beginner
This repository contains a collection of beginner-level AI projects developed during my final year AI module in my bachelor's degree in Computer Systems Engineering. The projects focus on exploring key concepts in artificial intelligence, including machine learning, data preprocessing, and predictive modelling. Each project is implemented in Python and demonstrates practical applications of AI techniques in real-world scenarios. These projects not only highlight the use of AI methods but also reflect my early journey into the field.

The following machine learning models were applied across the projects:

- XGBoost Regressor: Utilised for predicting calories burnt through regression.
- Support Vector Machine (SVM): Applied for classifying the likelihood of diabetes.
- Random Forest Regressor: Used to forecast diamond prices based on multiple features.
- Linear Regression: Implemented to predict insurance charges based on various factors.
- TF-IDF Vectorisation and Cosine Similarity: Applied in the development of the Chatbot ROBO and Movie Recommendation System for text-based feature extraction and similarity measurement.

## Table of contents

### 1. Calories Burnt Prediction 
- CaloriesBurntPrediction.ipynb
- This project involves developing a machine learning model to predict calories burnt based on user-specific features, such as Gender, Age, Height, Weight, Duration of activity, Heart Rate, and Body Temperature. Using Python, the data is analysed and visualised with pandas, Matplotlib, and Seaborn, followed by splitting the dataset into training and testing sets. An XGBoost Regressor is implemented for regression, and model performance is evaluated using Mean Absolute Error. This project demonstrates the application of regression techniques to predict a continuous variable, showcasing the power of machine learning in health and fitness-related analytics.

### 2. Chatbot ROBO
- Chatbot_ROBO.ipynb
- This project is a simple rule-based chatbot named ROBO, designed to answer user queries about chatbots. It uses TF-IDF vectorisation and cosine similarity to generate context-based responses from a provided text corpus. The chatbot can recognise greetings, respond interactively, and handle basic conversations. Built with Python, it leverages libraries like NLTK for preprocessing (tokenisation, lemmatisation) and scikit-learn for text vectorisation. This project highlights the use of natural language processing techniques in building conversational agents, providing a foundation for developing more advanced AI-driven chatbots.

### 3. Diabetes Prediction
- DiabetesPrediction.ipynb
- This project focuses on predicting the likelihood of diabetes based on various health features, including Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age. Using Python, the data is preprocessed by standardising the features with StandardScaler and splitting the dataset into training and testing sets. A Support Vector Machine (SVM) model is trained and evaluated for accuracy using the accuracy_score metric. This project showcases how machine learning can aid in healthcare by predicting medical conditions, demonstrating the potential for AI to support early diagnosis and treatment planning.

### 4. Diamond Price Forecasting 
- DiamondPriceForecasting.ipynb
- This project aims to forecast diamond prices based on features such as carat, cut, color, clarity, depth, table, and x, y, and z measurements. The dataset is analysed and visualised using libraries like Seaborn and Matplotlib. Categorical data is transformed into numerical form using Label Encoding. A Random Forest Regressor model is trained to predict diamond prices, with performance evaluated using metrics like Mean Absolute Error and Mean Squared Error. This project illustrates the ability of machine learning models to handle complex datasets for price prediction, emphasising their usefulness in the e-commerce and luxury goods sectors.

### 5. Insurance Charge Prediction 
- InsuranceChargePrediction.ipynb
- This project focuses on predicting insurance charges based on features such as age, sex, BMI, children, smoker, and region. Through Exploratory Data Analysis (EDA) and visualisations with Seaborn and Matplotlib, the dataset is thoroughly examined. A Linear Regression model is applied for prediction, with performance assessed using the R-squared value, which indicates the proportion of variance explained by the model. This project demonstrates how regression techniques can effectively predict financial variables, offering insights for decision-making in the insurance industry.

### 6. Movie Recommendation System
- MovieRecommendationSystem.ipynb
- This project involves developing a simple movie recommendation system based on key features such as genres, keywords, tagline, cast, and director. The system uses TF-IDF Vectorization to convert these textual features into numerical form, then calculates cosine similarity to assess how similar movies are to one another. By leveraging difflib for string matching, the recommendation engine suggests movies based on user input. This project highlights the role of recommendation systems in enhancing user experience, showcasing a practical application of machine learning in personalising content delivery.
