import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.express as px
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
