import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import yaml 
import logging
from utilities import load_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    

df = load_data('./data/Diabetes.csv')


class XGBClassifer: 
    def __init__(self): 
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.TARGET = config['XGB_CLF']['MODEL_PARAMS']['TARGET']
        self.FEATURES = config['XGB_CLF']['MODEL_PARAMS']['FEATURES']
        self.N_ITER = config['XGB_CLF']['MODEL_PARAMS']['N_ITER']
        self.VERBOSE = config['XGB_CLF']['MODEL_PARAMS']['VERBOSE']
        
    
    def preprocessing(self, df):
        """
        This function preprocesses the data by dropping unnecessary columns, encoding the target variable, scaling the features and splitting the data into training and testing sets
        
        Parameters
        ----------
        df: pd.DataFrame
        
        Returns
        -------
        X_scaled: np.array
        y : np.array
        
        """
        self.df = df
        self.X = self.df[self.FEATURES]
        self.y = self.df[self.TARGET]
        
        
        X_num = self.X.select_dtypes(['int64', 'float64'])
        X_cat = self.X.select_dtypes('object')
        
        self.X_scaled = StandardScaler().fit_transform(X_num)
        
        if X_cat.shape[1] > 0:
            self.logger.info('Categorical columns found')
            self.X_encoded = pd.get_dummies(X_cat, drop_first=True)
            self.X_scaled = np.concatenate((self.X_scaled, self.X_encoded), axis=1)
            
        return self.X_scaled, self.y
    
    
    def model_training(self, X, y): 
        """
        This function trains the model on the training data
        
        Parameters
        ----------
        X: np.array
        y: np.array
        
        Returns
        -------
        model: trained model
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier()
        self.random_search = RandomizedSearchCV(estimator=model, 
                                                scoring = 'roc_auc',
                                                param_distributions=config['XGB_CLF']['PARAM_GRID'], 
                                                n_iter=self.N_ITER, 
                                                cv=StratifiedKFold(n_splits=5), 
                                                verbose=self.VERBOSE, 
                                                random_state=42)
        
        self.random_search.fit(self.X_train, self.y_train)
        self.best_model = self.random_search.best_estimator_
        self.best_score = self.random_search.best_score_
        
        logger.info(f"Best model score: {self.best_score}")
        logger.info(f"Best model params: {self.best_model.get_params()}")
        
        return self.best_model
    

    def predict_new_data(self, X):
        """
        This function predicts the target variable for new data
        
        Parameters
        ----------
        X: np.array
        
        Returns
        -------
        y_pred: np.array
        """
        self.y_pred = self.best_model.predict(X)
        
        return self.y_pred
    
    
    def confusion_matrix(self, y_test):
        """
        This function generates the confusion matrix
        
        Parameters
        ----------
        y_test: np.array
        """
        cm = confusion_matrix(y_test, self.y_pred)
        sns.heatmap(cm, annot=True)
        
        return cm 
    
    
    def classification_report(self, y_test):
        """
        This function generates the classification report
        
        Parameters
        ----------
        y_test: np.array
        """
        cr = classification_report(y_test, self.y_pred)
        
        return cr
    
    def feature_importance(self):
        """
        This function generates the feature importance plot
        """
        feature_importance = self.best_model.feature_importances_
        feature_importance_df = pd.DataFrame(feature_importance, index=self.FEATURES, columns=['importance'])
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        
        
        return feature_importance_df.head(5)
    
    
    
    def run(self):
        """
        This function runs all the other functions
        """
        X, y = self.preprocessing(df)
        self.model_training(X, y)
        self.predict_new_data(self.X_test)
        cm = self.confusion_matrix(self.y_test)
        logger.info(f"Confusion Matrix: {cm}")
        cr = self.classification_report(self.y_test)
        logger.info(f"Classification Report: {cr}")
        fm = self.feature_importance()
        logger.info(f"Feature Importance: {fm}")
        
        return self.best_model
    
    
# Instantiate the class
model = XGBClassifer()
    
    
if __name__ == '__main__':
    model.run()

