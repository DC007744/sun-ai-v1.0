# Description: This script contains the logic to retrain the ML model based on the customer ID and the model to train input parameters.
# It fetches the data from the database, preprocesses it, performs hyperparameter tuning, trains the model, and saves the best parameters and model.

# --------------------------------------------------------------------------------------------------------------------
# Section-I: Importing required libraries
# --------------------------------------------------------------------------------------------------------------------

import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, OneHotEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
import matplotlib.pyplot as plt
import time
from datetime import datetime

from skopt.space.space import Integer, Real
from db import get_db_connection
from skopt import BayesSearchCV

# --------------------------------------------------------------------------------------------------------------------
# Section-II: Defining functions to load and preprocess data, and train ML models
# --------------------------------------------------------------------------------------------------------------------

def determine_dataset_size_change(Cust_id):
    # Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # SQL query to fetch counts based on whether the rows have been used in retraining, filtered by customer ID
    if Cust_id == "Generic":
        query = """
            SELECT 
                SUM(CASE WHEN Used_in_Retraining = 0 THEN 1 ELSE 0 END) AS Not_Used_Count,
                SUM(CASE WHEN Used_in_Retraining = 1 THEN 1 ELSE 0 END) AS Used_Count
            FROM Dataset_table;
        """
        cursor.execute(query)
        result = cursor.fetchone()

    else:
        query = """
            SELECT 
                SUM(CASE WHEN Used_in_Retraining = 0 THEN 1 ELSE 0 END) AS Not_Used_Count,
                SUM(CASE WHEN Used_in_Retraining = 1 THEN 1 ELSE 0 END) AS Used_Count
            FROM Dataset_table
            WHERE Cust_id = ?;
        """
        cursor.execute(query, (Cust_id,))
        result = cursor.fetchone()

    # Extract the counts for not used and used rows
    count_of_not_used = result[0] if result[0] is not None else 0
    count_of_used = result[1] if result[1] is not None else 0

    # Total size of the dataset
    total_size = count_of_used + count_of_not_used

    # Initialize default dataset size change as marginal
    dataset_size_changed_scale = 'marginal'

    # Handle edge case where no data was previously used
    if count_of_used == 0:
        dataset_size_changed_scale = 'significant'
    else:
        # Calculate the percentage of new data
        size_change_percentage = (count_of_not_used / count_of_used) * 100

        # Apply dynamic thresholds based on dataset size
        if total_size <= 2000:  # For small datasets, use percentage-based logic
            if size_change_percentage < 10:
                dataset_size_changed_scale = 'minimal'
            elif 10 <= size_change_percentage < 30:
                dataset_size_changed_scale = 'marginal'
            else:
                dataset_size_changed_scale = 'significant'
        else:  # For larger datasets, use absolute threshold logic
            if count_of_not_used < 500:  # Less than 500 new rows is marginal
                dataset_size_changed_scale = 'minimal'
            elif 500 <= count_of_not_used < 1000:  # Between 500 and 1500 new rows is significant
                dataset_size_changed_scale = 'marginal'
            else:  # More than 1500 new rows is significant
                dataset_size_changed_scale = 'significant'

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return dataset_size_changed_scale

def preprocess_data(data):
    data_cleaned = data.copy()
    data_cleaned['Design_ref_category'] = data_cleaned['Design_ref'].str[:-3]  # Slicing to remove the last 3 characters
    if (data_cleaned['Prod_surface_area_mm2'] != 0).all():
        data_cleaned['Volume_Area_Ratio'] = data_cleaned['Prod_volume_mm3'] / data_cleaned['Prod_surface_area_mm2']
        data_cleaned['Weight_Area_Ratio'] = data_cleaned['Prod_weight_gms'] / data_cleaned['Prod_surface_area_mm2']
    else:
        data_cleaned['Volume_Area_Ratio'] = 0
        data_cleaned['Weight_Area_Ratio'] = 0

    if (data_cleaned['Prod_volume_mm3'] != 0).all():
        data_cleaned['Weight_Volume_Ratio'] = data_cleaned['Prod_weight_gms'] / data_cleaned['Prod_volume_mm3']
    else:
        data_cleaned['Weight_Volume_Ratio'] = 0

    logging.info(f"Calculated ratios: Volume_Area_Ratio, Weight_Area_Ratio, and Weight_Volume_Ratio...")
    print(f"Calculated ratios: Volume_Area_Ratio, Weight_Area_Ratio, and Weight_Volume_Ratio...")
    return data_cleaned

def remove_outliers(X, y):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (y >= lower_bound) & (y <= upper_bound)
    return X[mask], y[mask]

def fetch_best_params_from_db(model_type, model_for, cust_id):
    # Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    max_attempts = 3

    # Retry logic to fetch the latest model ID
    for attempt in range(max_attempts):
        try:
            if cust_id == "Generic":
                # SQL query to find the largest Model_ID for the specified model type and target
                latest_model_id_query = """
                SELECT MAX(Model_ID) FROM Model_Parameters
                WHERE MODEL_TYPE = ? AND Model_For = ?
                """
                cursor.execute(latest_model_id_query, (model_type, model_for))
                model_id = cursor.fetchone()[0]
                
                logging.info(f"Fetched the latest Model_ID: {model_id} for model type: {model_type} and model for: {model_for}")
                print(f"Fetched the latest Model_ID: {model_id} for model type: {model_type} and model for: {model_for}")
                
                break  # Exit the loop if the query is successful
            else:
                # SQL query to find the largest Model_ID for the specified model type and target
                latest_model_id_query = """
                SELECT MAX(Model_ID) FROM Model_Parameters
                WHERE MODEL_TYPE = ? AND Model_For = ? AND Cust_id = ?
                """
                cursor.execute(latest_model_id_query, (model_type, model_for, cust_id))
                model_id = cursor.fetchone()[0]
                
                logging.info(f"Fetched the latest Model_ID: {model_id} for model type: {model_type} and model for: {model_for}")
                print(f"Fetched the latest Model_ID: {model_id} for model type: {model_type} and model for: {model_for}")
                
                break  # Exit the loop if the query is successful
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1}: Fetching latest Model_ID failed due to a database connection issue. Retrying...")

            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to fetch the latest Model_ID after multiple attempts.")
                print("Failed to fetch the latest Model_ID after multiple attempts.")
                raise e
            
            # Wait before retrying
            time.sleep(5)

    if model_id is None:
        raise ValueError("No model parameters found in the database.")

    # Retry logic to fetch best parameters for the latest Model_ID
    for attempt in range(max_attempts):
        try:
            # SQL query to fetch the best parameters for the latest Model_ID
            query = """
            SELECT Param_Key, Param_Value
            FROM Model_Parameters
            WHERE Model_ID = ?
            """
            cursor.execute(query, (model_id,))
            param_rows = cursor.fetchall()
            
            logging.info(f"Fetched best parameters for Model_ID: {model_id}")
            print(f"Fetched best parameters for Model_ID: {model_id}")
            
            break  # Exit the loop if the query is successful
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1}: Fetching best parameters failed due to a database connection issue. Retrying...")

            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to fetch the best parameters after multiple attempts.")
                print("Failed to fetch the best parameters after multiple attempts.")
                raise e
            
            # Wait before retrying
            time.sleep(5)

    # Convert the results into a dictionary of parameters
    best_params = {row[0]: float(row[1]) for row in param_rows}

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return best_params

def perform_grid_search(model, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_

def save_training_details_summary(user, r2_fit_score, mae, mse, rmse, folder_name, main_save_path, cust_id, target, train_start_time, train_end_time):
    
    try:
        # Ensure `train_start_time` and `train_end_time` are datetime objects
        if not isinstance(train_start_time, datetime):
            train_start_time = datetime.fromtimestamp(train_start_time)
        if not isinstance(train_end_time, datetime):
            train_end_time = datetime.fromtimestamp(train_end_time)
        
        # Calculate total training duration in minutes
        total_training_duration = (train_end_time - train_start_time).total_seconds() / 60

        training_details_time = {
            'Training date': train_start_time.strftime('%b %d %Y'),
            'Training initiated by': user,
            'For customer ID': cust_id,
            'Training start time': train_start_time.strftime('%I:%M:%S %p'),
            'Training end time': train_end_time.strftime('%I:%M:%S %p'),
            'Total training duration': f"{total_training_duration:.2f} minutes",
            'New model accuracy (R2 Score)': f"{r2_fit_score*100:.2f} %",
            'MAE (error margin +/-)': mae,
            'MSE': mse,
            'RMSE': rmse,
        }
        
        with open(f'{main_save_path}/{folder_name}/{cust_id}_{target}_training_details.txt', 'w') as f:
            for key, value in training_details_time.items():
                f.write(f"{key}: {value}\n")

    except Exception as e:
            logging.exception("Error fetching dataset from database: ")
            print("Error fetching dataset from database: ")
            set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)

def set_is_being_used_for_retraining_flag(toggle: str, cust_id):

    conn = get_db_connection()
    cursor = conn.cursor()

    # If retraining is for generic customer with no historical data, use all available data of all customers
    if cust_id == "Generic":
        update_retraining_flag_query = """
            UPDATE Dataset_table
            SET is_being_used_for_retraining = ?
            WHERE (feedback_provided = 'Yes' OR feedback_provided = 'NA')
            """
        if toggle.lower() == "yes":
            cursor.execute(update_retraining_flag_query, ('Yes',))
            conn.commit()
        elif toggle.lower() == "no":
            cursor.execute(update_retraining_flag_query, ('No',))
            conn.commit()

    # If retraining is for a specific customer having historical data, use all data available for that customer only
    else:
        update_retraining_flag_query = """
            UPDATE Dataset_table
            SET is_being_used_for_retraining = ?
            WHERE Cust_id = ?
            AND (feedback_provided = 'Yes' OR feedback_provided = 'NA')
            """
        if toggle.lower() == "yes":
            cursor.execute(update_retraining_flag_query, ('Yes', cust_id))
            conn.commit()
        elif toggle.lower() == "no":
            cursor.execute(update_retraining_flag_query, ('No', cust_id))
            conn.commit()

    logging.info(f"Setting 'is_being_used_for_retraining' to {toggle} for customer {cust_id}.")
    print(f"Setting 'is_being_used_for_retraining' to {toggle} for customer {cust_id}.")

    conn.close()

# --------------------------------------------------------------------------------------------------------------------
# Section-III: Main function to start training the ML model
# --------------------------------------------------------------------------------------------------------------------

def start_model_training(cust_id: str, target, model_type, user, save_to_location):

    # Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # Set `is_being_used_for_retraining` to 'Yes' for the loaded data when training begins with the first model (i.e., the CFP_time)
    if target == 'CFP_time':
        set_is_being_used_for_retraining_flag(toggle="Yes", cust_id=cust_id)

    # Determine dataset size change
    logging.info(f"Determining dataset size change for customer {cust_id} since last training...")
    print(f"Determining dataset size change for customer {cust_id} since last training...")
    dataset_size_changed_scale = determine_dataset_size_change(Cust_id=cust_id)

    if dataset_size_changed_scale == 'minimal':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Random Search will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Random Search will be used for hyperparameter tuning...")

    elif dataset_size_changed_scale == 'marginal':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Grid Search will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Grid Search will be used for hyperparameter tuning...")

    elif dataset_size_changed_scale == 'significant':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Bayesian Optimization will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Bayesian Optimization will be used for hyperparameter tuning...")

    # Note the training start time
    start_time = time.time()

    # Create a new ML model folder
    folder_name = f"{cust_id}_Trained_Model_Version-{datetime.now().strftime('%b-%d-%Y_%HH.%MM_%p')}"
    base_path = save_to_location
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Set up logging
    formatted_time = datetime.fromtimestamp(time.time()).strftime("%b %d %Y (%I:%M:%S %p)")
    log_filename = os.path.join(folder_path, f'{cust_id}_ML_models_retraining_{formatted_time}_log.txt')

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("Started retraining for customer {} at {}...".format(cust_id, formatted_time))
    print("Started retraining for customer {} at {}...".format(cust_id, formatted_time))

    # Load the dataset based on the model to train
    if target == 'CFP_time':
        query = """
                SELECT  Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time
                FROM Dataset_table
                WHERE Cust_id = ? AND is_being_used_for_retraining = 'Yes'
                """
    elif target == 'Total_time':
        query = """
                SELECT Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time, Actual_production_labour_time_mins
                FROM Dataset_table 
                WHERE Cust_id = ? AND is_being_used_for_retraining = 'Yes'
                """
    elif target == 'CFP_cost':
        query = """
                SELECT  Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time, Actual_production_labour_time_mins, Actual_CFP_cost
                FROM Dataset_table
                WHERE Cust_id = ? AND is_being_used_for_retraining = 'Yes'
                """
    logging.info(f"Fetching data for customer {cust_id}...")
    print(f"Fetching data for customer {cust_id}...")

    # Execute the query and load data into a pandas DataFrame
    data = pd.read_sql_query(query, conn, params=[cust_id])

    logging.info(f"Loaded data for customer {cust_id}...")
    print(f"Loaded data for customer {cust_id}...")

    logging.info(f"Data shape: {data.head()}")
    print(f"Data shape: {data.head()}")

    logging.info(f"Data shape: {data.shape}")
    print(f"Data shape: {data.shape}")

    # Debugging step: print the columns to verify if Row_ID is included
    logging.info("Data columns retrieved: {}".format(data.columns))
    print("Data columns retrieved: {}".format(data.columns))

    # Create interaction features
    data_cleaned = preprocess_data(data)
    logging.info('Preprocessing complete...')
    print('Preprocessing complete...')

    # Splitting the data into features and target based on the model to train
    if target == 'CFP_time':
        X = data_cleaned.drop(columns=['Actual_CFP_time'])
        y = data_cleaned['Actual_CFP_time']
    elif target == 'Total_time':
        X = data_cleaned.drop(columns=['Actual_production_labour_time_mins'])
        y = data_cleaned['Actual_production_labour_time_mins']
    elif target == 'CFP_cost':
        X = data_cleaned.drop(columns=['Actual_CFP_cost'])
        y = data_cleaned['Actual_CFP_cost']

    # Remove outliers from the target variable
    X, y = remove_outliers(X, y)
    logging.info('Removed outliers from the target variable...')
    print('Removed outliers from the target variable...')

    # Splitting into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info('Split data into training and testing sets...')
        print('Split data into training and testing sets...')
    except Exception as e:
        logging.exception("Error splitting data into train and test sets")
        print("Error splitting data into train and test sets")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        

    # Define categorical and numerical features
    categorical_features = [
        'Prod_type', 'Cust_id', 'Design_ref', 'Design_ref_category', 'Assembly_type',
        'RH_type', 'Enamel_coat', 'Ceramic_coat', 'Alloy-1', 'Metal-1', 'Alloy-2',
        'Metal-2', 'PL_material_type'
    ]

    numerical_features = [
        'Prod_volume_mm3', 'Prod_surface_area_mm2', 'Prod_weight_gms',
        'Volume_Area_Ratio', 'Weight_Volume_Ratio', 'Weight_Area_Ratio'
    ]

    # Additional numerical features based on the target model
    if target == 'Total_time':
        numerical_features.append('Actual_CFP_time')
    elif target == 'CFP_cost':
        numerical_features.extend(['Actual_CFP_time', 'Actual_production_labour_time_mins'])

    try:
        # Preprocessing pipeline for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),   # Impute missing values with median
            ('scaler', MinMaxScaler()),                      # Scale values to [0, 1]
            ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Add polynomial features
        ])
    except Exception as e:
        logging.exception("Error preprocessing pipeline numerical data: ")
        print("Error preprocessing pipeline numerical data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        # Preprocessing pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except Exception as e:
        logging.exception("Error preprocessing pipeline categorical data: ")
        print("Error preprocessing pipeline categorical data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        # Combine numerical and categorical transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
    except Exception as e:
        logging.exception("Error combining preprocessing pipelines: ")
        print("Error combining preprocessing pipelines: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Preprocess the data
    try:
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
    except Exception as e:
        logging.exception("Error occured during preprocessing pipelines (fit_transform): ")
        print("Error occured during preprocessing pipelines (fit_transform): ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        best_params = fetch_best_params_from_db(model_type=model_type, model_for=target, cust_id=cust_id)
        logging.info(f"Previously used best parameters from DB: {best_params}")
        print(f"Previously used best parameters from DB: {best_params}")

    except Exception as e:
        logging.error(f"Error fetching previously used best parameters: {e}")
        print(f"Error fetching previously used best parameters: {e}")
        
        # Assign default parameters if fetching from the database fails
        best_params = {
            'depth': 6,             # Example default depth
            'iterations': 100,      # Example default iterations
            'learning_rate': 0.1,   # Example default learning rate
            'l2_leaf_reg': 3.0      # Example default L2 regularization
        }
        logging.info(f"Using default parameters: {best_params}")
        print(f"Using default parameters: {best_params}")

    # Building param_grid based on dataset size
    try:
        if dataset_size_changed_scale == 'minimal':
            # Define a small parameter grid around the best parameters
            param_grid = {
                'depth': [int(best_params['depth'])],  # Fixing depth to current best for minimal change
                'iterations': [int(best_params['iterations']) - 10, int(best_params['iterations']), int(best_params['iterations']) + 10],
                'learning_rate': [best_params['learning_rate'] - 0.005, best_params['learning_rate'], best_params['learning_rate'] + 0.005],
                'l2_leaf_reg': [best_params['l2_leaf_reg'] - 0.05, best_params['l2_leaf_reg'], best_params['l2_leaf_reg'] + 0.05],
                # 'thread_count': -1
            }
            logging.info(f"Param grid for minimal dataset size change: {param_grid}")
            print(f"Param grid for minimal dataset size change: {param_grid}")

            # Perform Randomized Search based on the model selected
            if model_type == 'CatBoost':
                logging.info('Performing Randomized Search for CatBoost...')
                print('\nPerforming Randomized Search for CatBoost...')
                
                model = CatBoostRegressor(random_seed=42, silent=True)
                
                # Randomized search with 10 iterations (you can adjust this)
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=10,  # Number of parameter settings sampled (can be adjusted)
                    cv=5,
                    n_jobs=-1,
                    scoring='neg_mean_absolute_error',
                    random_state=42
                )
                
                random_search.fit(X_train_preprocessed, y_train)
                best_params = random_search.best_params_
            
            logging.info(f"Best parameters from Randomized Search: {best_params}")
            print(f"Best parameters from Randomized Search: {best_params}")

        elif dataset_size_changed_scale == 'marginal':
            # For small changes in dataset size, build a grid with the last best parameters +/- 1 step.
            param_grid = {
                'depth': [int(best_params['depth']) - 1, int(best_params['depth']), int(best_params['depth']) + 1],
                'iterations': [int(best_params['iterations']) - 50, int(best_params['iterations']), int(best_params['iterations']) + 50],
                'learning_rate': [best_params['learning_rate'] - 0.01, best_params['learning_rate'], best_params['learning_rate'] + 0.01],
                'l2_leaf_reg': [best_params['l2_leaf_reg'] - 0.1, best_params['l2_leaf_reg'], best_params['l2_leaf_reg'] + 0.1],
                #'thread_count': -1
            }
            logging.info(f"Param grid for marginal dataset size change: {param_grid}")
            print(f"Param grid for marginal dataset size change: {param_grid}")

            # Perform Grid Search based on the model selected
            if model_type == 'CatBoost':
                logging.info('Performing Grid Search for CatBoost...')
                print('\nPerforming Grid Search for CatBoost...')
                model = CatBoostRegressor(random_seed=42, silent=True)
                best_params = perform_grid_search(model, param_grid, X_train_preprocessed, y_train)
            # Additional models can be added here if needed

            logging.info(f"Best parameters from Grid Search: {best_params}")
            print(f"Best parameters from Grid Search: {best_params}")

        elif dataset_size_changed_scale == 'significant':

            # For significant changes in dataset size, implement Bayesian optimization
            param_grid = {
                'depth': Integer(int(best_params['depth']) - 2, int(best_params['depth']) + 2),
                'iterations': Integer(max(1, int(best_params['iterations']) - 100), int(best_params['iterations']) + 100),
                'learning_rate': Real(best_params['learning_rate'] - 0.02, best_params['learning_rate'] + 0.02),
                'l2_leaf_reg': Real(best_params['l2_leaf_reg'] - 0.2, best_params['l2_leaf_reg'] + 0.2)
            }
            logging.info(f"Param grid for Bayesian optimization: {param_grid}")
            print(f"Param grid for Bayesian optimization: {param_grid}")

            try:
                bayes_search = BayesSearchCV(
                    estimator=CatBoostRegressor(random_seed=42, silent=True),
                    search_spaces=param_grid,
                    n_iter=30,
                    cv=5,
                    n_jobs=-1,
                    scoring='neg_mean_absolute_error'
                )
                logging.info('Performing Bayesian optimization...')
                print('Performing Bayesian optimization...')
                bayes_search.fit(X_train_preprocessed, y_train)
                best_params = bayes_search.best_params_
            except Exception as e:
                logging.error(f"Error during Bayesian optimization: {e}")
                print(f"Error during Bayesian optimization: {e}")
                raise

            logging.info(f"Best parameters from Bayesian optimization: {best_params}")
            print(f"Best parameters from Bayesian optimization: {best_params}")
    
    except Exception as e:
        logging.exception("Error occured during preprocessing pipelines (fit_transform): ")
        print("Error occured during preprocessing pipelines (fit_transform): ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Saving the best parameters
    try:
        with open(f'{save_to_location}\{cust_id}_{target}_model_best_parameters.txt', 'w') as f:
            json.dump(best_params, f, indent=4)
    except Exception as e:
        logging.exception("Error saving the best parameters as a text file: ")
        print("Error saving the best parameters as a text file: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return
    
    logging.info('Saved the best parameters from hyperparameter tuning...')
    print('Saved the best parameters from hyperparameter tuning...')

    # Initialize the model with best parameters from hyperparameter tuning
    try:
        catboost_model = CatBoostRegressor(**best_params, random_seed=42, silent=True, thread_count=-1)
    except Exception as e:
        logging.exception("Error occured during model initialization: ")
        print("Error occured during model initialization: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Create a pipeline with preprocessing and the model
    try:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', catboost_model)])
    except Exception as e:
        logging.exception("Error occured during model pipeline creation: ")
        print("Error occured during model pipeline creation: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Fit the pipeline on the training data
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logging.exception("Error fitting pipeline on data: ")
        print("Error fitting pipeline on data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return
    
    logging.info('Fitted the pipeline with best parameters')
    print('Fitted the pipeline with best parameters...')

    # Use the pipeline to predict on the test data
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        logging.exception("Error predicting on test set: ")
        print("Error predicting on test set: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Evaluate the model
    print('Evaluating the model...')
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f'MAE: {mae}')
    print(f'\nMAE: {mae}')
    logging.info(f'MSE: {mse}')
    print(f'MSE: {mse}')
    logging.info(f'RMSE: {rmse}')
    print(f'RMSE: {rmse}')
    logging.info(f'R2 Score: {r2*100:.2f} %')
    print(f'R2 Score: {r2*100:.2f} %\n')

    # Save the pipeline
    model_filename = f'{cust_id}_{target}_{model_type}_pipeline.joblib'
    model_filepath = os.path.join(folder_path, model_filename)
    joblib.dump(pipeline, model_filepath)
    logging.info('Saved the pipeline')
    print('Saved the pipeline...')

    # Save model details in the Model_Registry table
    model_log_file = log_filename
    model_accuracy = r2 * 100

    # Make the previous default model as non-default for the customer
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Make the previous default model as non-default for the customer
            update_query = """
            UPDATE Model_Registry
            SET IsDefault = 0
            WHERE IsDefault = 1 AND Model_For = ? AND Model_Type = ? AND Cust_id = ?
            """
            cursor.execute(update_query, (target, model_type, cust_id))
            conn.commit()
            
            logging.info(f"Updated the previous default model as non-default for target: {target} and model type: {model_type}")
            print(f"Updated the previous default model as non-default for target: {target} and model type: {model_type}")
            
            break  # If successful, exit the loop
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1}: Update failed due to a database connection issue. Retrying...")
            
            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to update the database after multiple attempts.")
                print("Failed to update the database after multiple attempts.")
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                raise e
            
            # Wait before retrying
            time.sleep(5)

    for attempt in range(max_attempts):
        try:
            # Insert into Model_Registry table and retrieve the inserted Model_ID
            insert_query = """
                INSERT INTO Model_Registry (Model_For, Model_Name, Model_Type, Model_Location, Model_Accuracy, Model_Log_File_Location, Created_Date, IsDefault, Cust_id)
                OUTPUT INSERTED.Model_ID
                VALUES (?, ?, ?, ?, ?, ?, GETDATE(), 1, ?)
            """
            cursor.execute(insert_query, (target, model_filename, model_type, model_filepath, model_accuracy, model_log_file, cust_id))
            model_id = cursor.fetchone()[0]
            conn.commit()
            
            logging.info(f"New model registered with Model_ID: {model_id}")
            print(f"New model registered with Model_ID: {model_id}")
            
            break  # Exit the loop if insertion is successful
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} to insert model failed: {e}")
            print(f"Attempt {attempt + 1}: Insert failed due to a database connection issue. Retrying...")
            
            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to insert into the database after multiple attempts.")
                print("Failed to insert into the database after multiple attempts.")
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                raise e
            
            # Wait before retrying
            time.sleep(5)

    # Save best parameters into Model_Parameters table with retry logic
    logging.info('Saving best parameters into Model_Parameters table...')
    print('Saving best parameters into Model_Parameters table...')

    for param_key, param_value in best_params.items():
        for attempt in range(max_attempts):
            try:
                # Prepare the query and attempt to insert the parameter
                insert_param_query = """
                    INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type, Cust_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_param_query, (model_id, param_key, str(param_value), target, model_type, cust_id))
                conn.commit()
                
                logging.info(f"Parameter {param_key} with value {param_value} saved successfully.")
                print(f"Parameter {param_key} with value {param_value} saved successfully.")
                
                break  # Exit the retry loop if the insertion is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} to save parameter '{param_key}' failed: {e}")
                print(f"Attempt {attempt + 1}: Failed to save parameter '{param_key}'. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error(f"Failed to save parameter '{param_key}' after multiple attempts.")
                    print(f"Failed to save parameter '{param_key}' after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

    logging.info('All best parameters saved in Model_Parameters table.')
    print('All best parameters saved in Model_Parameters table.')

    # After retraining, update only the rows used in training
    if target == 'CFP_cost':
        # Retry logic for updating 'Used_in_Retraining' column
        for attempt in range(max_attempts):
            try:
                # Update query to set 'Used_in_Retraining' to 1 for specified rows
                update_query = """
                UPDATE Dataset_table
                SET Used_in_Retraining = 1
                WHERE Cust_id = ? AND is_being_used_for_retraining = 'Yes'
                """
                cursor.execute(update_query, (cust_id,))
                conn.commit()
                
                logging.info(f"Updated 'Used_in_Retraining' column to 1 for used rows in customer {cust_id}'s retraining.")
                print(f"Updated 'Used_in_Retraining' column to 1 for used rows in customer {cust_id}'s retraining.")
                
                break  # Exit the loop if the update is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                print(f"Attempt {attempt + 1}: Update failed due to a database connection issue. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error("Failed to update 'Used_in_Retraining' column after multiple attempts.")
                    print("Failed to update 'Used_in_Retraining' column after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

        # Retry logic for resetting 'is_being_used_for_retraining' flag
        for attempt in range(max_attempts):
            try:
                # Reset the flag to 'No' for the specified customer
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                
                logging.info(f"Resetting 'is_being_used_for_retraining' to 'No' for customer {cust_id} as final retraining is complete.")
                print(f"Resetting 'is_being_used_for_retraining' to 'No' for customer {cust_id} as final retraining is complete.")
                
                break  # Exit the loop if the update is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                print(f"Attempt {attempt + 1}: Resetting flag failed due to a database connection issue. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error("Failed to reset 'is_being_used_for_retraining' flag after multiple attempts.")
                    print("Failed to reset 'is_being_used_for_retraining' flag after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

    end_time = time.time()
    formatted_end_time = datetime.fromtimestamp(end_time).strftime('%I:%M:%S %p')
    print(f"Retraining completed for {target} for customer {cust_id} at {formatted_end_time}...")

    if target == "CFP_cost":
        logging.info("Retraining for all 3 models completed successfully.")
        print("Retraining for all 3 models completed successfully.")

    # Save training details summary
    save_training_details_summary(user, r2, mae, mse, rmse, folder_name, save_to_location, cust_id, target, start_time, end_time)

    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")

# --------------------------------------------------------------------------------------------------------------------
# Section-IV: Main function to start training the ML model for "Generic" customers
# --------------------------------------------------------------------------------------------------------------------

def start_generic_model_training(cust_id: str, target, model_type, user, save_to_location):

    # Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # Set `is_being_used_for_retraining` to 'Yes' for the loaded data when training begins with the first model (i.e., the CFP_time)
    if target == 'CFP_time':
        set_is_being_used_for_retraining_flag(toggle="Yes", cust_id=cust_id)

    # Determine dataset size change
    logging.info(f"Determining dataset size change for customer {cust_id} since last training...")
    print(f"Determining dataset size change for customer {cust_id} since last training...")
    dataset_size_changed_scale = determine_dataset_size_change(Cust_id=cust_id)

    if dataset_size_changed_scale == 'minimal':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Random Search will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Random Search will be used for hyperparameter tuning...")

    elif dataset_size_changed_scale == 'marginal':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Grid Search will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Grid Search will be used for hyperparameter tuning...")

    elif dataset_size_changed_scale == 'significant':
        logging.info(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        logging.info("Bayesian Optimization will be used for hyperparameter tuning...")
        print(f"Dataset size has changed {dataset_size_changed_scale}ly since last training...")
        print("Bayesian Optimization will be used for hyperparameter tuning...")

    # Note the training start time
    start_time = time.time()

    # Create a new ML model folder
    folder_name = f"{cust_id}_Trained_Model_Version-{datetime.now().strftime('%b-%d-%Y_%HH.%MM_%p')}"
    base_path = save_to_location
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Set up logging
    formatted_time = datetime.fromtimestamp(time.time()).strftime("%b %d %Y (%I:%M:%S %p)")
    log_filename = os.path.join(folder_path, f'{cust_id}_ML_models_retraining_{formatted_time}_log.txt')

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    logging.info("Started retraining for {} model for customer {} at {}...".format(target, cust_id, formatted_time))
    print("Started retraining for {} model for customer {} at {}...".format(target, cust_id, formatted_time))

    # Load the dataset based on the model to train
    if target == 'CFP_time':
        query = """
                SELECT  Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time
                FROM Dataset_table
                WHERE is_being_used_for_retraining = 'Yes'
                """
    elif target == 'Total_time':
        query = """
                SELECT Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time, Actual_production_labour_time_mins
                FROM Dataset_table 
                WHERE is_being_used_for_retraining = 'Yes'
                """
    elif target == 'CFP_cost':
        query = """
                SELECT  Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time, Actual_production_labour_time_mins, Actual_CFP_cost
                FROM Dataset_table
                WHERE is_being_used_for_retraining = 'Yes'
                """
    logging.info(f"Fetching data for customer {cust_id}...")
    print(f"Fetching data for customer {cust_id}...")

    # Execute the query and load data into a pandas DataFrame
    try:
        data = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error: {e}")

    logging.info(f"Loaded data for customer {cust_id}...")
    print(f"Loaded data for customer {cust_id}...")

    logging.info(f"Data shape: {data.head()}")
    print(f"Data shape: {data.head()}")

    logging.info(f"Data shape: {data.shape}")
    print(f"Data shape: {data.shape}")

    # Debugging step: print the columns to verify if Row_ID is included
    logging.info("Data columns retrieved: {}".format(data.columns))
    print("Data columns retrieved: {}".format(data.columns))

    # Create interaction features
    data_cleaned = preprocess_data(data)
    logging.info('Preprocessing complete...')
    print('Preprocessing complete...')

    # Splitting the data into features and target based on the model to train
    if target == 'CFP_time':
        X = data_cleaned.drop(columns=['Actual_CFP_time'])
        y = data_cleaned['Actual_CFP_time']
    elif target == 'Total_time':
        X = data_cleaned.drop(columns=['Actual_production_labour_time_mins'])
        y = data_cleaned['Actual_production_labour_time_mins']
    elif target == 'CFP_cost':
        X = data_cleaned.drop(columns=['Actual_CFP_cost'])
        y = data_cleaned['Actual_CFP_cost']

    # Remove outliers from the target variable
    X, y = remove_outliers(X, y)
    logging.info('Removed outliers from the target variable...')
    print('Removed outliers from the target variable...')

    # Splitting into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info('Split data into training and testing sets...')
        print('Split data into training and testing sets...')
    except Exception as e:
        logging.exception("Error splitting data into train and test sets")
        print("Error splitting data into train and test sets")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        

    # Define categorical and numerical features
    categorical_features = [
        'Prod_type', 'Cust_id', 'Design_ref', 'Design_ref_category', 'Assembly_type',
        'RH_type', 'Enamel_coat', 'Ceramic_coat', 'Alloy-1', 'Metal-1', 'Alloy-2',
        'Metal-2', 'PL_material_type'
    ]

    numerical_features = [
        'Prod_volume_mm3', 'Prod_surface_area_mm2', 'Prod_weight_gms',
        'Volume_Area_Ratio', 'Weight_Volume_Ratio', 'Weight_Area_Ratio'
    ]

    # Additional numerical features based on the target model
    if target == 'Total_time':
        numerical_features.append('Actual_CFP_time')
    elif target == 'CFP_cost':
        numerical_features.extend(['Actual_CFP_time', 'Actual_production_labour_time_mins'])

    try:
        # Preprocessing pipeline for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),   # Impute missing values with median
            ('scaler', MinMaxScaler()),                      # Scale values to [0, 1]
            ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Add polynomial features
        ])
    except Exception as e:
        logging.exception("Error preprocessing pipeline numerical data: ")
        print("Error preprocessing pipeline numerical data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        # Preprocessing pipeline for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except Exception as e:
        logging.exception("Error preprocessing pipeline categorical data: ")
        print("Error preprocessing pipeline categorical data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        # Combine numerical and categorical transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
    except Exception as e:
        logging.exception("Error combining preprocessing pipelines: ")
        print("Error combining preprocessing pipelines: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Preprocess the data
    try:
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
    except Exception as e:
        logging.exception("Error occured during preprocessing pipelines (fit_transform): ")
        print("Error occured during preprocessing pipelines (fit_transform): ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    try:
        best_params = fetch_best_params_from_db(model_type=model_type, model_for=target, cust_id=cust_id)
        logging.info(f"Previously used best parameters from DB: {best_params}")
        print(f"Previously used best parameters from DB: {best_params}")

    except Exception as e:
        logging.error(f"Error fetching previously used best parameters: {e}")
        print(f"Error fetching previously used best parameters: {e}")
        
        # Assign default parameters if fetching from the database fails
        best_params = {
            'depth': 6,             # Example default depth
            'iterations': 100,      # Example default iterations
            'learning_rate': 0.1,   # Example default learning rate
            'l2_leaf_reg': 3.0      # Example default L2 regularization
        }
        logging.info(f"Using default parameters: {best_params}")
        print(f"Using default parameters: {best_params}")

    # Building param_grid based on dataset size
    try:
        dataset_size_changed_scale = 'significant'                       # -------------------- TEMPPPPP ------------
        if dataset_size_changed_scale == 'minimal':
            # Define a small parameter grid around the best parameters
            param_grid = {
                'depth': [int(best_params['depth'])],  # Fixing depth to current best for minimal change
                'iterations': [int(best_params['iterations']) - 10, int(best_params['iterations']), int(best_params['iterations']) + 10],
                'learning_rate': [best_params['learning_rate'] - 0.005, best_params['learning_rate'], best_params['learning_rate'] + 0.005],
                'l2_leaf_reg': [best_params['l2_leaf_reg'] - 0.05, best_params['l2_leaf_reg'], best_params['l2_leaf_reg'] + 0.05],
                # 'thread_count': -1
            }
            logging.info(f"Param grid for minimal dataset size change: {param_grid}")
            print(f"Param grid for minimal dataset size change: {param_grid}")

            # Perform Randomized Search based on the model selected
            if model_type == 'CatBoost':
                logging.info('Performing Randomized Search for CatBoost...')
                print('\nPerforming Randomized Search for CatBoost...')
                
                model = CatBoostRegressor(random_seed=42, silent=True)
                
                # Randomized search with 10 iterations (you can adjust this)
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=10,  # Number of parameter settings sampled (can be adjusted)
                    cv=5,
                    n_jobs=-1,
                    scoring='neg_mean_absolute_error',
                    random_state=42
                )
                
                random_search.fit(X_train_preprocessed, y_train)
                best_params = random_search.best_params_
            
            logging.info(f"Best parameters from Randomized Search: {best_params}")
            print(f"Best parameters from Randomized Search: {best_params}")

        elif dataset_size_changed_scale == 'marginal':
            # For small changes in dataset size, build a grid with the last best parameters +/- 1 step.
            param_grid = {
                'depth': [int(best_params['depth']) - 1, int(best_params['depth']), int(best_params['depth']) + 1],
                'iterations': [int(best_params['iterations']) - 50, int(best_params['iterations']), int(best_params['iterations']) + 50],
                'learning_rate': [best_params['learning_rate'] - 0.01, best_params['learning_rate'], best_params['learning_rate'] + 0.01],
                'l2_leaf_reg': [best_params['l2_leaf_reg'] - 0.1, best_params['l2_leaf_reg'], best_params['l2_leaf_reg'] + 0.1],
                #'thread_count': -1
            }
            logging.info(f"Param grid for marginal dataset size change: {param_grid}")
            print(f"Param grid for marginal dataset size change: {param_grid}")

            # Perform Grid Search based on the model selected
            if model_type == 'CatBoost':
                logging.info('Performing Grid Search for CatBoost...')
                print('\nPerforming Grid Search for CatBoost...')
                model = CatBoostRegressor(random_seed=42, silent=True)
                best_params = perform_grid_search(model, param_grid, X_train_preprocessed, y_train)
            # Additional models can be added here if needed

            logging.info(f"Best parameters from Grid Search: {best_params}")
            print(f"Best parameters from Grid Search: {best_params}")

        elif dataset_size_changed_scale == 'significant':

            # For significant changes in dataset size, implement Bayesian optimization
            param_grid = {
                'depth': Integer(int(best_params['depth']) - 2, int(best_params['depth']) + 2),
                'iterations': Integer(max(1, int(best_params['iterations']) - 100), int(best_params['iterations']) + 100),
                'learning_rate': Real(best_params['learning_rate'] - 0.02, best_params['learning_rate'] + 0.02),
                'l2_leaf_reg': Real(best_params['l2_leaf_reg'] - 0.2, best_params['l2_leaf_reg'] + 0.2)
            }
            logging.info(f"Param grid for Bayesian optimization: {param_grid}")
            print(f"Param grid for Bayesian optimization: {param_grid}")

            try:
                bayes_search = BayesSearchCV(
                    estimator=CatBoostRegressor(random_seed=42, silent=True),
                    search_spaces=param_grid,
                    n_iter=30,
                    cv=5,
                    n_jobs=-1,
                    scoring='neg_mean_absolute_error'
                )
                logging.info('Performing Bayesian optimization...')
                print('Performing Bayesian optimization...')
                bayes_search.fit(X_train_preprocessed, y_train)
                best_params = bayes_search.best_params_
            except Exception as e:
                logging.error(f"Error during Bayesian optimization: {e}")
                print(f"Error during Bayesian optimization: {e}")
                raise

            logging.info(f"Best parameters from Bayesian optimization: {best_params}")
            print(f"Best parameters from Bayesian optimization: {best_params}")
    
    except Exception as e:
        logging.exception("Error occured during preprocessing pipelines (fit_transform): ")
        print("Error occured during preprocessing pipelines (fit_transform): ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Saving the best parameters
    try:
        with open(f'{save_to_location}\{cust_id}_{target}_model_best_parameters.txt', 'w') as f:
            json.dump(best_params, f, indent=4)
    except Exception as e:
        logging.exception("Error saving the best parameters as a text file: ")
        print("Error saving the best parameters as a text file: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return
    
    logging.info('Saved the best parameters from hyperparameter tuning...')
    print('Saved the best parameters from hyperparameter tuning...')

    # Initialize the model with best parameters from hyperparameter tuning
    try:
        catboost_model = CatBoostRegressor(**best_params, random_seed=42, silent=True, thread_count=-1)
    except Exception as e:
        logging.exception("Error occured during model initialization: ")
        print("Error occured during model initialization: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Create a pipeline with preprocessing and the model
    try:
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', catboost_model)])
    except Exception as e:
        logging.exception("Error occured during model pipeline creation: ")
        print("Error occured during model pipeline creation: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Fit the pipeline on the training data
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        logging.exception("Error fitting pipeline on data: ")
        print("Error fitting pipeline on data: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return
    
    logging.info('Fitted the pipeline with best parameters')
    print('Fitted the pipeline with best parameters...')

    # Use the pipeline to predict on the test data
    try:
        y_pred = pipeline.predict(X_test)
    except Exception as e:
        logging.exception("Error predicting on test set: ")
        print("Error predicting on test set: ")
        set_is_being_used_for_retraining_flag(toggle="No", cust_id=cust_id)
        return

    # Evaluate the model
    print('Evaluating the model...')
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f'MAE: {mae}')
    print(f'\nMAE: {mae}')
    logging.info(f'MSE: {mse}')
    print(f'MSE: {mse}')
    logging.info(f'RMSE: {rmse}')
    print(f'RMSE: {rmse}')
    logging.info(f'R2 Score: {r2*100:.2f} %')
    print(f'R2 Score: {r2*100:.2f} %\n')

    # Save the pipeline
    model_filename = f'{cust_id}_{target}_{model_type}_pipeline.joblib'
    model_filepath = os.path.join(folder_path, model_filename)
    joblib.dump(pipeline, model_filepath)
    logging.info('Saved the pipeline')
    print('Saved the pipeline...')

    # Save model details in the Model_Registry table
    model_log_file = log_filename
    model_accuracy = r2 * 100

    # Make the previous default model as non-default for the customer
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Make the previous default model as non-default for the customer
            update_query = """
            UPDATE Model_Registry
            SET IsDefault = 0
            WHERE IsDefault = 1 AND Model_For = ? AND Model_Type = ? AND Cust_id = ?
            """
            cursor.execute(update_query, (target, model_type, cust_id))
            conn.commit()
            
            logging.info(f"Updated the previous default model as non-default for target: {target} and model type: {model_type}")
            print(f"Updated the previous default model as non-default for target: {target} and model type: {model_type}")
            
            break  # If successful, exit the loop
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")
            print(f"Attempt {attempt + 1}: Update failed due to a database connection issue. Retrying...")
            
            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to update the database after multiple attempts.")
                print("Failed to update the database after multiple attempts.")
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                raise e
            
            # Wait before retrying
            time.sleep(5)

    for attempt in range(max_attempts):
        try:
            # Insert into Model_Registry table and retrieve the inserted Model_ID
            insert_query = """
                INSERT INTO Model_Registry (Model_For, Model_Name, Model_Type, Model_Location, Model_Accuracy, Model_Log_File_Location, Created_Date, IsDefault, Cust_id)
                OUTPUT INSERTED.Model_ID
                VALUES (?, ?, ?, ?, ?, ?, GETDATE(), 1, ?)
            """
            cursor.execute(insert_query, (target, model_filename, model_type, model_filepath, model_accuracy, model_log_file, cust_id))
            model_id = cursor.fetchone()[0]
            conn.commit()
            
            logging.info(f"New model registered with Model_ID: {model_id}")
            print(f"New model registered with Model_ID: {model_id}")
            
            break  # Exit the loop if insertion is successful
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} to insert model failed: {e}")
            print(f"Attempt {attempt + 1}: Insert failed due to a database connection issue. Retrying...")
            
            # If max attempts reached, raise the error to stop execution
            if attempt == max_attempts - 1:
                logging.error("Failed to insert into the database after multiple attempts.")
                print("Failed to insert into the database after multiple attempts.")
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                raise e
            
            # Wait before retrying
            time.sleep(5)

    # Save best parameters into Model_Parameters table with retry logic
    logging.info('Saving best parameters into Model_Parameters table...')
    print('Saving best parameters into Model_Parameters table...')

    for param_key, param_value in best_params.items():
        for attempt in range(max_attempts):
            try:
                # Prepare the query and attempt to insert the parameter
                insert_param_query = """
                    INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type, Cust_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_param_query, (model_id, param_key, str(param_value), target, model_type, cust_id))
                conn.commit()
                
                logging.info(f"Parameter {param_key} with value {param_value} saved successfully.")
                print(f"Parameter {param_key} with value {param_value} saved successfully.")
                
                break  # Exit the retry loop if the insertion is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} to save parameter '{param_key}' failed: {e}")
                print(f"Attempt {attempt + 1}: Failed to save parameter '{param_key}'. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error(f"Failed to save parameter '{param_key}' after multiple attempts.")
                    print(f"Failed to save parameter '{param_key}' after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

    logging.info('All best parameters saved in Model_Parameters table.')
    print('All best parameters saved in Model_Parameters table.')

    # After retraining, update only the rows used in training
    if target == 'CFP_cost':
        # Retry logic for updating 'Used_in_Retraining' column
        for attempt in range(max_attempts):
            try:
                # Update query to set 'Used_in_Retraining' to 1 for specified rows
                update_query = """
                UPDATE Dataset_table
                SET Used_in_Retraining = 1
                WHERE is_being_used_for_retraining = 'Yes'
                """
                cursor.execute(update_query)
                conn.commit()
                
                logging.info(f"Updated 'Used_in_Retraining' column to 1 for used rows in customer {cust_id}'s retraining.")
                print(f"Updated 'Used_in_Retraining' column to 1 for used rows in customer {cust_id}'s retraining.")
                
                break  # Exit the loop if the update is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                print(f"Attempt {attempt + 1}: Update failed due to a database connection issue. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error("Failed to update 'Used_in_Retraining' column after multiple attempts.")
                    print("Failed to update 'Used_in_Retraining' column after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

        # Retry logic for resetting 'is_being_used_for_retraining' flag
        for attempt in range(max_attempts):
            try:
                # Reset the flag to 'No' for the specified customer
                set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                
                logging.info(f"Resetting 'is_being_used_for_retraining' to 'No' for customer {cust_id} as final retraining is complete.")
                print(f"Resetting 'is_being_used_for_retraining' to 'No' for customer {cust_id} as final retraining is complete.")
                
                break  # Exit the loop if the update is successful
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                print(f"Attempt {attempt + 1}: Resetting flag failed due to a database connection issue. Retrying...")

                # If max attempts reached, raise the error to stop execution
                if attempt == max_attempts - 1:
                    logging.error("Failed to reset 'is_being_used_for_retraining' flag after multiple attempts.")
                    print("Failed to reset 'is_being_used_for_retraining' flag after multiple attempts.")
                    set_is_being_used_for_retraining_flag(toggle='No', cust_id=cust_id)
                    raise e
                
                # Wait before retrying
                time.sleep(5)

    end_time = time.time()
    formatted_end_time = datetime.fromtimestamp(end_time).strftime('%I:%M:%S %p')
    print(f"Retraining completed for {target} for customer {cust_id} at {formatted_end_time}...")

    if target == "CFP_cost":
        logging.info("Retraining for all 3 models completed successfully.")
        print("Retraining for all 3 models completed successfully.")

    # Save training details summary
    save_training_details_summary(user, r2, mae, mse, rmse, folder_name, save_to_location, cust_id, target, start_time, end_time)

    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")