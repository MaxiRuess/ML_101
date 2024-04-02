import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
import yaml
import logging
import streamlit as st
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
    
#df = load_data('./data/car_prices.csv')

class CustomRandomForestRegressor: 
    def __init__(self): 
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.TARGET = config['RANDOMFOREST_REG']['MODEL_PARAMS']['TARGET']
        self.FEATURES = config['RANDOMFOREST_REG']['MODEL_PARAMS']['FEATURES']
        self.N_ITER = config['RANDOMFOREST_REG']['MODEL_PARAMS']['N_ITER']
        self.VERBOSE = config['RANDOMFOREST_REG']['MODEL_PARAMS']['VERBOSE']
        
    
    def preprocessing(self, df):
        """
        This function preprocesses the data by dropping unnecessary columns, encoding the target variable, scaling the features and splitting the data into training and testing sets
        
        Parameters
        ----------
        df: pd.DataFrame
        
        Returns
        -------
        X_scaled: np.array
        y: np.array
        """
        
        self.df = df
        self.df = self.df.dropna()
        
        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET]
        
        X_num = self.X.select_dtypes(['int64', 'float64'])
        X_cat = self.X.select_dtypes('object')
        
        self.X_scaler = StandardScaler()
        self.X_scaled = self.X_scaler.fit_transform(X_num)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=X_num.columns)
        
        self.y_scaler = StandardScaler()
        self.y_ = self.y_scaler.fit_transform(self.y.values.reshape(-1, 1))
        
        
        if X_cat.shape[1] > 0:
            X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
            self.X_scaled = pd.concat((self.X_scaled, X_cat_encoded), axis=1)
        logger.info(self.X_scaled.shape, self.y_.shape)
        return self.X_scaled, self.y_
    
    def model_training(self, X, y): 
        """
        This function trains the Random Forest Regressor model
        
        Parameters
        ----------
        X: np.array
        y: np.array
        
        Returns
        -------
        model: RandomForestRegressor
        """
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor()
        random_search = RandomizedSearchCV(estimator=model, 
                                           param_distributions=config['RANDOMFOREST_REG']['PARAM_GRID'], 
                                           n_iter=self.N_ITER, cv=KFold(n_splits=5), 
                                           verbose=self.VERBOSE)
        
        random_search.fit(self.X_train, self.y_train.ravel())
        
        self.best_model = random_search.best_estimator_
        self.best_score = random_search.best_score_
        self.best_params = random_search.best_params_
        logger.info(f"Best score: {self.best_score}")
        logger.info(f"Best parameters: {random_search.best_params_}")
        
        
        return self.best_model, self.best_params
    
    
    def predict_all_data(self, X): 
        """
        This function predicts the target variable for all the data
        
        Parameters
        ----------
        X: np.array
        
        Returns
        -------
        y_pred: np.array
        """
        
        self.y_pred_all = self.best_model.predict(X)
        self.df_new = self.df.copy()
        self.df_new['predicted_price'] = self.y_scaler.inverse_transform(self.y_pred_all.reshape(-1, 1))
        self.df_new['predicted_price'] = self.df_new['predicted_price'].round()
        self.df_new['residuals'] = self.df_new[self.TARGET] - self.df_new['predicted_price']
        
        
        return self.df_new

    
    def evaluate_model(self):
        """
        This function evaluates the model using the R2 score, RMSE and MAE
        
        Parameters
        ----------
        y_test: np.array
        y_pred: np.array
        
        Returns
        -------
        r2_score: float
        rmse: float
        mae: float
        """
        # Reversing the scaling for MAE and RMSE
        self.y_test = self.y_scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        self.y_pred = self.best_model.predict(self.X_test)
        # Reversing the scaling for MAE and RMSE
        self.y_pred = self.y_scaler.inverse_transform(self.y_pred.reshape(-1, 1))
        self.r2_score = self.best_score
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        
        return self.r2_score, self.rmse, self.mae
    
    def visualize_tree(self):
        """
        This function visualizes the tree of the Random Forest Regressor model
        """
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.best_model.estimators_[0], feature_names=self.FEATURES, filled=True)
        plt.show()
        
    def feature_importance(self):
        """
        This function returns the top 5 most important features
        
        Returns
        -------
        feature_importance: pd.DataFrame
        """
        feature_importance = self.best_model.feature_importances_
        features = self.X_scaled.columns
        
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        return feature_importance.head(5)
    
    def visualize_feature_importance(self):
        """
        This function visualizes the feature importance
        """
        
        feature_importance = self.feature_importance()
        
        fig = px.bar(feature_importance, x='Feature', y='Importance')

        
        return fig
    
    def residual_plot(self):
        """
        This function plots the residuals
        """
        
        fig = px.scatter(self.df_new, x=self.TARGET, y='residuals')
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        
        return fig 
        
    
    
def run(df_test): 
    #df = pd.read_csv(uploaded_file)
    rf = CustomRandomForestRegressor()
    df = df_test
    X, y = rf.preprocessing(df)
    best_model, best_params = rf.model_training(X, y)
    st.error(f"Best model parameters: {best_params}")
    df_new = rf.predict_all_data(X)
    r2_score, rmse, mae = rf.evaluate_model()
    feature_importance = rf.feature_importance()
    st.info(f"R2 Score: {r2_score}, RMSE: {rmse}, MAE: {mae}")
    st.subheader('Predictions')
    st.write(df_new)
    st.subheader('Feature Importance')
    fig = rf.visualize_feature_importance()
    st.plotly_chart(fig)
    st.subheader('Residual Plot')
    fig = rf.residual_plot()
    st.plotly_chart(fig)
    st.subheader('Tree Visualization')
    #fig = rf.visualize_tree()
    #st.plotly_chart(fig)

       

st.header('Random Forest Regressor')
st.write('This is a simple Random Forest Regressor model that predicts the price of a car based on the features provided')

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    run(df)

else: 
    uploaded_file = load_data('./data/car_prices.csv')
    run(uploaded_file)
    
    
    

    
    
    



    
    
    
            
        
            
        
        
        
        
        