import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold
from utilities import load_data, streamlit_app
import yaml
import logging
import mlflow
import mlflow.sklearn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
    
df = load_data('./data/perth_house_prices.csv')

# Keeping all code in a class is not necessary, but it's a good practice to keep the code organized
# This way, you can easily see which functions are related to each other
# This is also usefull to get more familiar with OOP

class XGBoost:
    def __init__(self):
        super().__init__()
        self.TARGET = config['XGBOOST_REG']['MODEL_PARAMS']['TARGET']
        self.FEATURES = config['XGBOOST_REG']['MODEL_PARAMS']['FEATURES']
        self.logger = logging.getLogger(__name__)
        self.n_iter = config['XGBOOST_REG']['MODEL_PARAMS']['N_ITER']
        self.verbose = config['XGBOOST_REG']['MODEL_PARAMS']['VERBOSE']
        self.n_splits = config['XGBOOST_REG']['MODEL_PARAMS']['N_SPLITS']
        
    # Preprocessing
    # These functions would normally be in a separate file, but I'm including them here for simplicity and readability
    # I've tried to make this as modular but there are limits to this approach, like with the Null Value impuation
    # You need to look at the data first to determine the best way to impute the missing values
    
    def preprocessing(self, df): 
        """
        Preprocess the data for XGBoost
        Args:
            param df: DataFrame
            
        Returns:
            X, y, X_scaler, y_scaler
        """
        
        df['GARAGE'] = df['GARAGE'].fillna(0)
        df['BUILD_YEAR'] = df['BUILD_YEAR'].fillna(df['BUILD_YEAR'].median())
        df['NEAREST_SCH_RANK'] = df['NEAREST_SCH_RANK'].fillna(df['NEAREST_SCH_RANK'].median())
        
        df_new = pd.concat([df[self.FEATURES], df[self.TARGET]], axis=1)
        df_numerical = df_new.select_dtypes(['int64', 'float64'])
        df_categorical = pd.get_dummies(df_new.select_dtypes('object'))
        
        df_dummies = pd.get_dummies(df_categorical, drop_first=True)
        
        X_num = df_numerical.drop(self.TARGET, axis=1)
        self.X_scaler = StandardScaler()
        X_num_scaled = self.X_scaler.fit_transform(X_num.values.reshape(-1, X_num.shape[1]))
        X_num_scaled = pd.DataFrame(X_num_scaled, columns=X_num.columns)
        
        self.y_scaler = StandardScaler()
        y = df_numerical[self.TARGET]
        y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1))
        
        X = pd.concat([X_num_scaled, df_dummies], axis=1)
        y = y_scaled
        
        return X, y
    
    
    # Train the model 
    # I'm using RandomizedSearchCV to find the best hyperparameters
    
    def model_training(self, X, y): 
        """
        Train the XGBoost model
        Args:
            param X: DataFrame
            param y: DataFrame
            
        Returns:
            model
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb = XGBRegressor()
        
        param_grid = config['XGBOOST_REG']['PARAM_GRID']
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=self.n_iter, scoring='r2', n_jobs=-1, cv=kf, random_state=42)
        random_search.fit(X_train, y_train)
                
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        self.logger.info(f'Best parameters: {self.best_params}')
        self.logger.info(f'Best score: {self.best_score}')
        
        y_pred = random_search.best_estimator_.predict(X_test)
        y_test_inv = self.y_scaler.inverse_transform(y_test)
        pred_scaled = pd.DataFrame(y_pred, columns=[self.TARGET])
        y_pred_original = self.y_scaler.inverse_transform(pred_scaled)
        
        logger.info(f'R2: {r2_score(y_test, y_pred)}')
        logger.info(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        
        self.model = random_search.best_estimator_
        
        return self.model
    
    # This is the same as the previous function, but with MLflow
    # MLflow is a great tool to track your experiments and models
    # To visit the MLflow UI, run mlflow ui in the terminal
    
    def model_training_mlflow(self, X, y):
        """
        Train the XGBoost model with MLflow
        Args:
            param X: DataFrame
            param y: DataFrame
            
        Returns:
            model
        """
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb = XGBRegressor()
        param_grid = config['XGBOOST_REG']['PARAM_GRID']
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        random_search = RandomizedSearchCV(xgb, param_distributions=param_grid, n_iter=self.n_iter, scoring='r2', n_jobs=-1, cv=kf, random_state=42)
        
        with mlflow.start_run():
            xgb = random_search.fit(X_train, y_train)
            
            self.best_params = random_search.best_params_
            self.best_score = random_search.best_score_
            
            mlflow.log_params(self.best_params)
            mlflow.log_metric('r2', self.best_score)
            
            y_pred = random_search.best_estimator_.predict(X_test)
            y_test_inv = self.y_scaler.inverse_transform(y_test)
            pred_scaled = pd.DataFrame(y_pred, columns=[self.TARGET])
            y_pred_original = self.y_scaler.inverse_transform(pred_scaled)
            mae = mean_absolute_error(y_test_inv, y_pred_original)
            
            mlflow.log_metric('mae', mae)
            mlflow.sklearn.log_model(random_search.best_estimator_, 'model')
            
        self.model = random_search.best_estimator_
        
        return self.model
            
    
    def predict_new_data(self, model, df, X): 
        """
        Predict new data
        Args:
            param model: model
            param df: DataFrame
            param X: DataFrame
            
        Returns:
            predictions
        """
        
        predictions = model.predict(X)
        pred_scaled = pd.DataFrame(predictions, columns=[self.TARGET])
        y_pred_original = self.y_scaler.inverse_transform(pred_scaled)
        y_pred_original = pd.DataFrame(y_pred_original, columns=[self.TARGET])
        
        df_new = df.copy()
        df_new['prediction'] = y_pred_original
        
        return df_new
    
    def feature_importance(self, model, X):
        """
        Get the feature importance
        Args:
            param model: model
            param X: DataFrame
        """
        
        feature_importance = model.feature_importances_
        features = X.columns
        feature_importance_df = pd.DataFrame({'features': features, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        
        
        #fig = px.bar(feature_importance_df, x='features', y='importance', title='Feature Importance')
        #fig.show()
        
        return feature_importance_df.head()
    
    
    def plot_forecasts(self, df):
        """
        Plot the forecasts
        Args:
            param df: DataFrame
        """
        
        fig = px.scatter(df, x = config['XGBOOST_REG']['MODEL_PARAMS']['TARGET'], y = 'predictions', title='Predictions vs. Actuals')
        fig.add_shape(type='line', x0=df[self.TARGET].min(), y0=df[self.TARGET].min(), x1=df[self.TARGET].max(), y1=df[self.TARGET].max())
        fig.show()
        
        return fig
    
    def plot_residuals(self, df):
        """
        Plot the residuals
        Args:
            param df: DataFrame
        """
        
        df_copy = df.copy()
        df_copy['residuals'] = df[self.TARGET] - df['predictions']
        fig = px.scatter(df_copy, x = 'predictions', y = 'residuals', title='Residuals vs. Predictions')
        fig.add_hline(y=0, line_dash="dash")
        fig.show()
        
        return fig
    
    
xgboost = XGBoost()

try: 
    X, y = xgboost.preprocessing(df)
    logger.info(f'Preprocessing completed, X shape: {X.shape}, y shape: {y.shape}')
except Exception as e:
    logger.error(f'Error: {e}')

# Optinal MLflow
if config['XGBOOST_REG']['MLFLOW']:
    try: 
        model_xgb = xgboost.model_training_mlflow(X, y)
        logger.info(f'Model training completed')
    except Exception as e:
        logger.error(f'Error: {e}')
else:
    try: 
        model_xgb = xgboost.model_training(X, y)
        logger.info(f'Model training completed')
    except Exception as e:
        logger.error(f'Error: {e}')
    
try:
    features_imp = xgboost.feature_importance(model_xgb, X)
    logger.info(f'Feature importance completed, Top 5 features: {features_imp.head()}') 
except Exception as e:
    logger.error(f'Error: {e}')
    
try:
    df_new = xgboost.predict_new_data(model_xgb, df, X)
    logger.info(f'Predictions: {df_new.head()}')
except Exception as e:
    logger.error(f'Error: {e}')
    

# Flag include plots
if config['XGBOOST_REG']['PLOTS']:
    try:
        xgboost.plot_forecasts(df_new)
    except Exception as e:
        logger.error(f'Error: {e}')
    
    try:
        xgboost.plot_residuals(df_new)
    except Exception as e:
        logger.error(f'Error: {e}')
        
        
    

    
        
        