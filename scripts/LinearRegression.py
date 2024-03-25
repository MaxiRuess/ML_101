import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
from utilities import load_data
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the config file
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load the data
df = load_data('./data/medical_insurance.csv')


# Keeping all code in a class is not necessary, but it's a good practice to keep the code organized
# This way, you can easily see which functions are related to each other
# This is also usefull to get more familiar with OOP


class LinearReg:
    def __init__(self):
        super().__init__()
        self.TARGET = config['LINEAR_REG']['MODEL_PARAMS']['TARGET']
        self.FEATURES = config['LINEAR_REG']['MODEL_PARAMS']['FEATURES']
        self.logger = logging.getLogger(__name__)
    
    # Preprocessing
    # These functions would normally be in a separate file, but I'm including them here for simplicity and readability
    # I'm also using the StandardScaler from scikit-learn, which is a bit more convenient than the statsmodels version

    def preprocess_data(self, df):
        """
        Preprocess the data for linear regression
        Args:
            param df: DataFrame
        
        Returns: 
            X, y, X_scaler, y_scaler
        """
        # One-hot encode    

        
        df = pd.concat([df[self.FEATURES], df[self.TARGET]], axis=1)
        
        df_categorical = pd.get_dummies(df.select_dtypes('object'))
        df_dummies = pd.get_dummies(df_categorical, drop_first=True)
            

        df_numerical = df.select_dtypes(['int64', 'float64'])
        X_num = df_numerical.drop(self.TARGET,  axis=1)
        # Scale data 
        # I'm using different scalers for X and y, because I want to be able to inverse transform the predictions later
        X_scaler = StandardScaler()
        X_num_scaled = X_scaler.fit_transform(X_num.values.reshape(-1, X_num.shape[1]))
        X_num_scaled = pd.DataFrame(X_num_scaled, columns=X_num.columns)
        
        y_scaler = StandardScaler()
        y = df_numerical[self.TARGET]
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))
        
        # Combine X
        X = pd.concat([X_num_scaled, df_dummies], axis=1)
        y = y_scaled
        
        return X, y, X_scaler, y_scaler

    # Model training
    # This function would normally be in a separate file, but I'm including it here for simplicity and readability

    def model_training(self, X, y):
        """
        Train the linear regression model
        Args:
            param X: DataFrame
            param y: DataFrame
        
        Returns: 
            model
        """
        X = sm.add_constant(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = sm.OLS(y_train, X_train.astype(float)).fit()
        y_pred = model.predict(X_test)
        y_test_inv = y_scaler.inverse_transform(y_test)
        pred_scaled = pd.DataFrame(y_pred, columns=[self.TARGET])
        y_pred_original = y_scaler.inverse_transform(pred_scaled)
        
        logger.info(f'R2: {r2_score(y_test, y_pred)}')
        logger.info(f'MAE: {mean_absolute_error(y_test_inv, y_pred_original)}')
        logger.info(f"Model summary: {model.summary()}")
        
        return model

    def predict_new_data(self, model, df, X, y_scaler): 
        """
        Predict new data
        Args:
            param model: model
            param df: DataFrame
            param X: DataFrame
        
        Returns: 
            predictions
        """
        X = sm.add_constant(X)
        predictions = model.predict(X)
        pred_scaled = pd.DataFrame(predictions, columns=[self.TARGET])
        y_pred_original = y_scaler.inverse_transform(pred_scaled)
        y_pred_original = pd.DataFrame(y_pred_original, columns=[self.TARGET])
        
        df_new = df.copy()
        df_new['predictions'] = y_pred_original
        
        return df_new

    def plot_forecasts(self, df):
        """
        Plot the forecasts
        Args:
            param df: DataFrame
        """
        fig = px.scatter(df, x = config['LINEAR_REG']['MODEL_PARAMS']['TARGET'], y = 'predictions', title='Predictions vs. Actuals')
        fig.add_shape(type='line', line=dict(dash='dash'), x0=df[config['LINEAR_REG']['MODEL_PARAMS']['TARGET']].min(), y0=df[config['LINEAR_REG']['MODEL_PARAMS']['TARGET']].min(), x1=df[config['LINEAR_REG']['MODEL_PARAMS']['TARGET']].max(), y1=df[config['LINEAR_REG']['MODEL_PARAMS']['TARGET']].max())
        fig.show()
        
        return fig

    def plot_residual(self, df):
        """
        Plot the residuals
        Args:
            param df: DataFrame
        """
        df_copy = df.copy()
        df_copy['residuals'] = df[config['LINEAR_REG']['MODEL_PARAMS']['TARGET']] - df['predictions']
        fig = px.scatter(df_copy, x = 'predictions', y = 'residuals', title='Residuals vs. Predictions')
        fig.add_hline(y=0, line_dash="dash")
        fig.show()
        
        return fig

linear_reg  = LinearReg()

try: 
    X, y, X_scaler, y_scaler = linear_reg.preprocess_data(df)
except Exception as e:
    logger.error(f'Error: {e}')
    
try:
    model = linear_reg.model_training(X, y)
except Exception as e:
    logger.error(f'Error: {e}')
    
try:
    df_new = linear_reg.predict_new_data(model, df, X, y_scaler)
    logger.info(f'Predictions: {df_new}')
except Exception as e:
    logger.error(f'Error: {e}')

  
# Flag include plots 
if config['LINEAR_REG']['PLOTS']:
    try:
        model.plot_forecasts(df_new)
    except Exception as e:
        logger.error(f'Error: {e}')
    
    try:
        model.plot_residual(df_new)
    except Exception as e:
        logger.error(f'Error: {e}')
        
        

if __name__ == '__main__':
    pass


    
    

    
