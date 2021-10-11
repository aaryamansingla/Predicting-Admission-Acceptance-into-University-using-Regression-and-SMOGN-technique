# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:58:16 2021

@author: aaryaman
"""
import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv('Admission_Predict.csv')
df = df.rename(columns={'GRE Score': 'GRE Score','TOEFL Score': 'TOEFL Score', 
                                                                'LOR ': 'LOR', 
                                                                'Chance of Admit ': 'Admit Possibilty'})
st.markdown("### Predicting Admission Acceptance into University with features like CGPA, GRE Score, etc. using Bayesian and Random Forest Regressor along with SMOGN balancing technique")
ml_model = pickle.load(open("model.pkl", "rb"))

def get_user_input():
    GRE = st.sidebar.slider("Enter your GRE Score (Out of 340)",
                          df["GRE Score"].min(),
                          df["GRE Score"].max(),
                          step = 1)
    TOEFL = st.sidebar.slider("Enter your TOEFL Score (Out of 120)",
                          df["TOEFL Score"].min(),
                          df["TOEFL Score"].max(),
                          step = 1)
    Rating = st.sidebar.slider("Enter University Rating",
                           df["University Rating"].min(),
                           df["University Rating"].max(),
                           step = 1)
    SOP = st.sidebar.slider("Enter your SOP Score",
                           df["SOP"].min(),
                           df["SOP"].max(),
                           df["SOP"].mean())
    LOR = st.sidebar.slider("Enter your LOR Score",
                           df["LOR"].min(),
                           df["LOR"].max(),
                           df["LOR"].mean())
    CGPA = st.sidebar.slider("Enter your CGPA Score",
                           df["CGPA"].min(),
                           df["CGPA"].max(),
                           df["CGPA"].mean())
    Research = st.sidebar.slider("Enter if you have any research work (1 for yes and 0 for No)",
                           df["Research"].min(),
                           df["Research"].max(),
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
st.markdown("###### Features: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA")

st.subheader("Adjust the seek in the left to get a value for chance of admit")
st.write("**The chance of getting accepted is: **",str(round(prediction[0],2)))
st.write("\n\n")

st.image("heatmap.png", caption='Correlation between the feature variables and Chance of Admit')