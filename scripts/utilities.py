import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
import streamlit as st


def load_data(path):
    df = pd.read_csv(path)
    return df


def streamlit_app(model_name, model, model_params, features, target): 
    st.header(f"Welcom to the {model_name} App")
    st.write("In this app you can use the model to make predictions")
    
    # Load the data
    
    for key, value in model_params.items():
        st.write(f"{key}: {value}")
        
    st.write("The features are:")
    
    for feature in features:
        feature = st.text_input(feature)
        
    new_featues = pd.DataFrame(features)
    
    model.predict(features)