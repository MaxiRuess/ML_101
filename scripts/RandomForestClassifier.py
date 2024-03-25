import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
import yaml
import logging
from sklearn.tree import plot_tree
from utilities import load_model
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)
    

df = load_model('./data/Cancer_Data.csv')


class RandomForestClassifier: 
    def __init__(self): 
        super().__init__()
        
        self.TARGET = config['RANDOMFOREST_CLF']['TARGET']
        self.FEATURES = config['RANDOMFOREST_CLF']['FEATURES']
        self.N_ITER = config['RANDOMFOREST_CLF']['N_ITER']
        self.CV = config['RANDOMFOREST_CLF']['CV']
        self.VERBOSE = config['RANDOMFOREST_CLF']['VERBOSE']
    
    
    def preprocess(self, df):
        
        self.df = df.drop(columns=['Unnamed: 32', 'id'])
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})
        
        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X.values.reshape(-1, self.X.shape[-1]))
        
        return X_scaled, self.y
    
    def model_training(self, X, y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = RandomForestClassifier()
        
        param_grid = config['RANDOMFOREST_CLF']['PARAM_GRID']
        
        kfc = KFold(n_splits=5, shuffle=True, random_state=42, )
        
        self.random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=self.N_ITER, cv=kfc, verbose=self.VERBOSE, random_state=42)
        self.random_search.fit(X_train, y_train)
        
        return self.random_search.best_params_
    
    
    def predict_new_data(self, data):
        
        data = data[self.FEATURES]
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, data.shape[-1]))
        
        return self.random_search.predict(data_scaled)
    
    
    def model_tree(self):
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(self.random_search.best_estimator_, ax=ax)
        plt.show()
        
        return fig
    
    def feature_importance(self):
        
        feature_importance = self.random_search.best_estimator_.feature_importances_
        features = self.FEATURES
        
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        
        
        fig = px.bar(x=self.FEATURES, y=feature_importance, labels={'x': 'Feature', 'y': 'Importance'})
        fig.show()
        
        return feature_importance-