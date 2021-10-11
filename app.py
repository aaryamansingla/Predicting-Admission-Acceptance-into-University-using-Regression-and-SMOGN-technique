# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:58:16 2021

@author: aaryaman
"""
import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge

df = pd.read_csv('Admission_Predict.csv')
df = df.rename(columns={'GRE Score': 'GRE Score','TOEFL Score': 'TOEFL Score', 
                                                                'LOR ': 'LOR', 
                                                                'Chance of Admit ': 'Admit Possibilty'})
st.markdown("### Predicting Admission Acceptance into University with features like CGPA, GRE Score, etc. using Bayesian and Random Forest Regressor along with SMOGN balancing technique")
ml_model = pickle.load(open("model.pkl", "rb"))

def get_user_input():
    GRE = st.sidebar.slider("1. Enter your GRE Score (Out of 340)",
                          int(df["GRE Score"].min()),
                          int(df["GRE Score"].max()),
                          step = 1)
    TOEFL = st.sidebar.slider("2. Enter your TOEFL Score (Out of 120)",
                          int(df["TOEFL Score"].min()),
                          int(df["TOEFL Score"].max()),
                          step = 1)
    Rating = st.sidebar.slider("3. Enter University Rating",
                           int(df["University Rating"].min()),
                           int(df["University Rating"].max()),
                           step = 1)
    SOP = st.sidebar.slider("4. Enter your SOP Score",
                           df["SOP"].min(),
                           df["SOP"].max(),
                           float(df["SOP"].mean()))
    LOR = st.sidebar.slider("5. Enter your LOR Score",
                           df["LOR"].min(),
                           df["LOR"].max(),
                           float(df["LOR"].mean()))
    CGPA = st.sidebar.slider("6. Enter your CGPA Score",
                           df["CGPA"].min(),
                           df["CGPA"].max(),
                           float(df["CGPA"].mean()))
    Research = st.sidebar.slider("7. Enter if you have any research work (1 for yes and 0 for No)",
                           int(df["Research"].min()),
                           int(df["Research"].max()),
                           step = 1)
    features = pd.DataFrame({"GRE Score":GRE,
                             "TOEFL Score":TOEFL,
                             "University Rating":Rating,
                             "SOP":SOP,
                             "LOR":LOR,
                             "CGPA":CGPA,
                             "Research":Research}, 
                            index = [0])
    return features

input_df = get_user_input() #get user input from sidebar
prediction = ml_model.predict(input_df) #get predicitions

#display predictions
st.markdown("###### Enter Seven Features: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA")

st.subheader("Adjust the seek in the left to get a value for chance of admit")
st.write("**The chance of getting accepted is: **",str(round(prediction[0],2)))
st.write("\n\n")

st.image("heatmap.png", caption='Correlation between the feature variables and Chance of Admit')