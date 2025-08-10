import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

def train_models():
    # Load data
    
    df = pd.read_csv("/home/sidd/Desktop/Used-Car-Price-Prediction/cardekho_dataset.csv", index_col=[0])
    
    # Remove unnecessary columns
    df.drop(['car_name', 'brand'], axis=1, inplace=True)
    
    # Prepare features and target
    X = df.drop(['selling_price'], axis=1)
    y = df['selling_price']
    
    # Encode model column
    le = LabelEncoder()
    X['model'] = le.fit_transform(X['model'])
    
    # Save label encoder
    with open('model_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Column transformer
    num_features = X.select_dtypes(exclude="object").columns
    onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
    
    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder(drop='first')
    
    preprocessor = ColumnTransformer([
        ("OneHotEncoder", oh_transformer, onehot_columns),
        ("StandardScaler", numeric_transformer, num_features)
    ], remainder='passthrough')
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Save preprocessor
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    
    # Define all models
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Adaboost Regressor": AdaBoostRegressor(),
        "Gradient Boost Regressor": GradientBoostingRegressor(),
        "XGBoost Regressor": XGBRegressor()
    }
    
    # Train all models
    trained_models = {}
    results = {}
    
    print("Training all models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Evaluate
        train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
        
        trained_models[name] = model
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        print(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.0f}")
    
    # Hyperparameter tuning for best models
    print("\nPerforming hyperparameter tuning...")
    
    # Random Forest tuning
    rf_params = {
        "max_depth": [5, 8, 15, None, 10],
        "max_features": [5, 7, "auto", 8],
        "min_samples_split": [2, 8, 15, 20],
        "n_estimators": [100, 200, 500, 1000]
    }
    
    rf_random = RandomizedSearchCV(
        estimator=RandomForestRegressor(),
        param_distributions=rf_params,
        n_iter=50,
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    rf_random.fit(X_train, y_train)
    
    # XGBoost tuning
    xgb_params = {
        "learning_rate": [0.1, 0.01],
        "max_depth": [5, 8, 12, 20, 30],
        "n_estimators": [100, 200, 300],
        "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]
    }
    
    xgb_random = RandomizedSearchCV(
        estimator=XGBRegressor(),
        param_distributions=xgb_params,
        n_iter=50,
        cv=3,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )
    xgb_random.fit(X_train, y_train)
    
    # Update models with tuned versions
    trained_models["Random Forest Regressor (Tuned)"] = rf_random.best_estimator_
    trained_models["XGBoost Regressor (Tuned)"] = xgb_random.best_estimator_
    
    # Evaluate tuned models
    for name, model in [("Random Forest Regressor (Tuned)", rf_random.best_estimator_), 
                       ("XGBoost Regressor (Tuned)", xgb_random.best_estimator_)]:
        y_test_pred = model.predict(X_test)
        test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
        results[name] = {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        print(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {test_rmse:.0f}")
    
    # Save all trained models
    with open('all_models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)
    
    # Save results
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nAll models trained and saved!")
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY (Test Set)")
    print("="*60)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:<30} | R2: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.0f}")

if __name__ == "__main__":
    train_models()