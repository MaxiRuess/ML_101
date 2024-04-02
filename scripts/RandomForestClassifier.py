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
from utilities import load_data
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    

df = load_data('./data/Cancer_Data.csv')


class CustomRandomForestClassifier: 
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.TARGET = config['RANDOMFOREST_CLF']['MODEL_PARAMS']['TARGET']
        self.FEATURES = config['RANDOMFOREST_CLF']['MODEL_PARAMS']['FEATURES']
        self.N_ITER = config['RANDOMFOREST_CLF']['MODEL_PARAMS']['N_ITER']
        self.VERBOSE = config['RANDOMFOREST_CLF']['MODEL_PARAMS']['VERBOSE']
    
    
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
        
        self.df = df.drop(columns=['Unnamed: 32', 'id'])
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})
        
        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET]
        
        X_num = self.X.select_dtypes(['int64', 'float64'])
        X_cat = self.X.select_dtypes('object')
        
        self.X_scaler = StandardScaler()
        
        X_scaled = self.X_scaler.fit_transform(X_num.values.reshape(-1, self.X.shape[-1]))
        
        if X_cat.shape[1] > 0:
            X_cat = pd.get_dummies(X_cat, drop_first=True)
            logger.info(f"Shape of X_cat: {X_cat.shape}")
            
            X_scaled = pd.concat([X_scaled, X_cat], axis=1)
            
        
        return X_scaled, self.y
    
    def model_training(self, X, y):
        """
        This function trains the model using RandomizedSearchCV
        
        Parameters
        ----------
        X: np.array
        y: np.array
        
        Returns
        -------
        best_model: RandomForestClassifier
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.model = RandomForestClassifier()
        
        param_grid = config['RANDOMFOREST_CLF']['PARAM_GRID']
        
        kfc = KFold(n_splits=5, shuffle=True, random_state=42, )
        
        self.random_search = RandomizedSearchCV(estimator=self.model, param_distributions=param_grid, n_iter=self.N_ITER, cv=kfc, verbose=self.VERBOSE, random_state=42)
        self.random_search.fit(X_train, y_train)
        
        self.best_score = self.random_search.best_score_
        
        logger.info(f'Best Score: {self.best_score}')
        logger.info(f'Best Parameters: {self.random_search.best_params_}')
        
        self.best_model = self.random_search.best_estimator_
        
        return self.best_model
    
    def predict_new_data(self, X):
        """
        This function predicts the target variable for new data
        
        Parameters
        ----------
        data: pd.DataFrame
        
        Returns
        -------
        predictions: np.array
        """
        predictions = self.best_model.predict(X)
        
        df_new = df.copy()
        df_new['predictions'] = predictions
        
        return df_new
    
    
    def model_tree(self):
        """
        This function plots the first decision tree in the Random Forest Classifier
        
        Returns
        -------
        fig: matplotlib.figure.Figure
        """
        
        plt.figure(figsize=(20, 10))
        fig = plot_tree(self.best_model.estimators_[0], filled=True)
        fig.show()
        
        return fig
    
    def feature_importance(self):
        """
        This function returns the top 5 most important features
        
        Returns
        -------
        feature_importance: pd.DataFrame
        """
        
        feature_importance = self.best_model.feature_importances_
        features = self.FEATURES
        
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        
        
        return feature_importance.head(5)
    

    
    def confusion_matrix(self, X, y):
        """
        This function returns the confusion matrix
        
        Parameters
        ----------
        X: np.array
        y: np.array
        
        Returns
        -------
        cm: np.array (Confusion Matrix)
        """
        
        y_pred = self.best_model.predict(X)
        
        cm = confusion_matrix(y, y_pred)
        
        return cm
    
    def classification_report(self, X, y):
        """
        This function returns the classification report
        
        Parameters
        ----------
        X: np.array
        y: np.array
        
        Returns
        -------
        cr: str (Classification Report)
        """
        
        y_pred = self.best_model.predict(X)
        
        cr = classification_report(y, y_pred)
        
        return cr
    
    def feature_importance_plot(self): 
        """
        This function plots the feature importance
        
        Returns
        -------
        fig: plotly.graph_objs._figure.Figure"""
        
        feature_importance = self.best_model.feature_importances_
        features = self.FEATURES
        
        fig = px.bar(x=features, y=feature_importance, labels={'x': 'Feature', 'y': 'Importance'})
        fig.show()
        
        return fig 
    
    def plot_forecasts(self, df): 
        """
        This function plots the forecasts
        Args:
        df: pd.DataFrame
        
        Returns
        -------
        fig: plotly.graph_objs._figure.Figure
        """
        
        fig = px.scatter(df, x = config['RANDOMFOREST_CLF']['MODEL_PARAMS']['TARGET'], y = 'predictions', title = 'Predictions vs Actuals')
        fig.add_shape(type='line' , x0 = df[self.TARGET].min(), x1 = df[self.TARGET].max(), y0 = df[self.TARGET].min(), y1 = df[self.TARGET].max())
        fig.show()
        
        return fig
    
    def plot_residuals(self, df):
        """
        This function plots the residuals
        
        Args:
        df: pd.DataFrame
        
        Returns
        -------
        fig: plotly.graph_objs._figure.Figure
        """
        
        df_copy = df.copy()
        df_copy['residuals'] = df[self.TARGET] - df['predictions']
        fig = px.scatter(x = df[self.TARGET], y = 'residuals', title = 'Residuals vs Actuals')
        fig.show()
        
        return fig
    
    
# Initialize the model
model_rfc = CustomRandomForestClassifier()

    
    
# I'm using try except block to catch any error that might occur during the training process
# This is to ensure that the error is logged and the pipeline does not break
# This also makes it easier to debug the error
# This could also be summarised ina single function that calls all the other functions 
# This function could be called in the main function

try: 
    X,y = model_rfc.preprocessing(df)
except Exception as e:
    logger.error(f'Error in preprocessing: {e}')
        
try:
    best_model = model_rfc.model_training(X, y)
except Exception as e:
    logger.error(f'Error in training: {e}')
        
try: 
    predict_new_data = model_rfc.predict_new_data(X)
except Exception as e:
    logger.error(f'Error in prediction: {e}')
        
try:
    feature_importance_df = model_rfc.feature_importance()
except Exception as e:
    logger.error(f'Error in feature importance: {e}')
    
try: 
    cm = model_rfc.confusion_matrix(X, y)
except Exception as e:
    logger.error(f'Error in confusion matrix: {e}')

try:
    cr = model_rfc.classification_report(X, y)
except Exception as e:
    logger.error(f'Error in classification report: {e}')
    
if config['RANDOMFOREST_CLF']['PLOT']:
    try:
        fig = model_rfc.feature_importance_plot()
    except Exception as e:
        logger.error(f'Error in feature importance plot: {e}')
        
    try:
        fig = model_rfc.plot_forecasts(predict_new_data)
    except Exception as e:
        logger.error(f'Error in forecast plot: {e}')
        
    try:
        fig = model_rfc.plot_residuals(predict_new_data)
    except Exception as e:
        logger.error(f'Error in residual plot: {e}')
        
    try:
        fig = model_rfc.model_tree()
    except Exception as e:
        logger.error(f'Error in model tree plot: {e}')
    
logger.info(f'Confusion Matrix: {cm}')
logger.info(f'Classification Report: {cr}')
logger.info(f'Feature Importance: {feature_importance_df}')
logger.info(f'Predictions: {predict_new_data.head()}')
