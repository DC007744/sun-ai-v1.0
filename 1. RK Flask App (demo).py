from datetime import datetime, timedelta
from time import time
from dateutil.relativedelta import relativedelta
from werkzeug.utils import secure_filename
import os
import logging
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify
import atexit
from pyngrok import ngrok, conf
import joblib
import pandas as pd
import io, json
import subprocess
import pyodbc
from werkzeug.security import generate_password_hash, check_password_hash
from db import get_db_connection
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import tempfile
from Model_retraining_logic import start_model_training,start_generic_model_training

# Set up logging
logging.basicConfig(filename='sunai_flask_app_logs.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s')

app = Flask(__name__)
app.secret_key = 'Rsec123key456K'
ADMIN_SECRET_KEY = "Sunaisecretkey*1234"

# Define global variables
global_user_name = ''
global_customer_id = ''
global_is_admin = 0

# Define the UPLOAD_FOLDER configuration and create it if it does not exist
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_metal_density(fg_metal_kt, alloy):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
    SELECT Gravity 
    FROM MetalAlloys 
    WHERE FG_Metal_Kt = ? AND Alloy = ?
    """
    cursor.execute(query, (fg_metal_kt, alloy))
    result = cursor.fetchone()
    conn.close()
    if result:
        return float(result[0])
    else:
        raise ValueError("Metal density not found for the given FG_Metal_Kt and Alloy")

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global uploaded_file
    # Clear the old session data before handling the new file
    session.pop('temp_file_path', None)

    error_messages = []  # List to store all error messages

    if 'file' not in request.files:
        error_messages.append("\n1. No file part in the request.")
    else:
        file = request.files['file']

        if file.filename == '':
            error_messages.append("\n2. No selected file.")
        elif not file.filename.endswith(('.xlsx', '.csv')):
            error_messages.append("\n3. Invalid file type. Only Excel (.xlsx) and CSV (.csv) files are allowed.")
        else:
            try:
                # Read the file
                if file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)

                # Check if the DataFrame has any rows (i.e., it isn't just headers or empty)
                if df.empty:
                    error_messages.append("\n4. The uploaded file is empty or contains only headers. Please provide a valid dataset.")

                # Check if any columns are unnamed (i.e., have no header)
                unnamed_columns = [col for col in df.columns if col.startswith('Unnamed')]
                unnamed_column_numbers = [str(int(item.split(':')[1]) + 1) for item in unnamed_columns]

                if unnamed_columns:
                    error_messages.append(f"\n5. The columns {', '.join(unnamed_column_numbers)} have missing headers. Please provide proper headers for all columns.")

                # Define required columns
                required_columns = ['J.No.', 'Prod_type', 'Cust_id', 'Prod_volume_mm3', 'Prod_surface_area_mm2',
                                    'Prod_weight_gms', 'Assembly_type', 'RH_type', 'Enamel_coat',
                                    'Ceramic_coat', 'Alloy-1', 'Metal-1', 'Alloy-2', 'Metal-2', 'PL_material_type',
                                    'CFP_time_minutes', 'Total_labour_time_minutes', 'CFP_labour_cost_USD']

                # Convert column names to lower case to match
                df.columns = df.columns.str.strip().str.lower()
                required_columns = [col.lower() for col in required_columns]

                # Check for missing columns
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    error_messages.append(f"\n6. Missing or incorrectly named columns: {', '.join(missing_columns)}")

                # Condition: Iterate through each required column and find columns with blank cells
                columns_with_blank_cells = [col for col in required_columns if df[col].isnull().any() or df[col].eq('').any()]

                if columns_with_blank_cells:
                    error_messages.append(f"\n7. The columns {', '.join(columns_with_blank_cells)} have one or more blank cells. Please upload a complete dataset.")

                # Check 'RH_type' values
                if not df['rh_type'].isin(['Pen', 'Dip', 'NORH']).all():
                    error_messages.append("\n8. Column 'RH_type' must have only 'Pen', 'Dip', or 'NORH' values.")

                # Check 'Enamel_coat' values
                if not df['enamel_coat'].isin(['NOEN', 'YESEN']).all():
                    error_messages.append("\n9. Column 'Enamel_coat' must have only 'NOEN' or 'YESEN' values.")

                # Check 'Ceramic_coat' values
                if not df['ceramic_coat'].isin(['NOCER', 'YESCER']).all():
                    error_messages.append("\n10. Column 'Ceramic_coat' must have only 'NOCER' or 'YESCER' values.")

                # Check 'PL_material_type' values
                if not df['pl_material_type'].isin(['NOPL', 'Gold', 'Silver']).all():
                    error_messages.append("\n11. Column 'PL_material_type' must have only 'NOPL', 'Gold', or 'Silver' values.")

                # Condition for 'Alloy-2': Ensure that if 'noal2' is present, it is in uppercase 'NOAL2'
                if not df['alloy-2'].apply(lambda x: True if x not in ['noal2', 'noal', 'none', 'no', 'None', 'No'] or x == 'NOAL2' else False).all():
                    error_messages.append("\n12. Column 'Alloy-2' contains 'noal2'. Please use uppercase 'NOAL2'.")

                # Condition for 'Metal-2': Ensure that if 'nom2' is present, it is in uppercase 'NOM2'
                if not df['metal-2'].apply(lambda x: True if x not in ['nom2', 'nom', 'none', 'no', 'None', 'No'] or x == 'NOM2' else False).all():
                    error_messages.append("\n13. The column 'Metal-2' must have the value 'NOM2' (in uppercase) if there is no second metal in the product.")

                # Condition for 'Assembly_type': Ensure that if 'noas' is present, it is in uppercase 'NOAS'
                if not df['assembly_type'].apply(lambda x: x not in ['noas', 'none', 'no', 'None', 'No'] or x == 'NOAS').all():
                    error_messages.append("\n14. The column 'Assembly_type' must have the value 'NOAS' (in uppercase) if there is no assembly.")

                # New Condition: Check if numeric columns contain only numeric values
                numeric_columns = ['cust_id', 'prod_volume_mm3', 'prod_surface_area_mm2', 'prod_weight_gms', 
                                   'cfp_time_minutes', 'total_labour_time_minutes', 'cfp_labour_cost_usd']
                non_numeric_columns = [col for col in numeric_columns if not pd.to_numeric(df[col], errors='coerce').notnull().all()]

                if non_numeric_columns:
                    error_messages.append(f"\n15. The columns {', '.join(non_numeric_columns)} must contain only numeric values.")

                # New Condition: Check if categorical columns contain only text values
                categorical_columns = ['prod_type', 'assembly_type', 'alloy-1', 'metal-1', 'alloy-2', 'metal-2']
                non_text_columns = [col for col in categorical_columns if df[col].apply(lambda x: str(x).isdigit()).any()]

                if non_text_columns:
                    error_messages.append(f"\n16. The columns {', '.join(non_text_columns)} must contain only text values.")

                # New Condition: Check if Cust_id column contains only one unique value
                unique_cust_ids = df['cust_id'].nunique()
                if unique_cust_ids > 1:
                    error_messages.append("\n17. The 'Cust_id' column must contain only one unique customer ID. Multiple customer IDs found.")

                # If there are any error messages, flash them all together
                if error_messages:
                    flash("Please address the following issues:\n" + "\n".join(error_messages), 'error')
                    return render_template('edit_dataset.html', message="File validation failed", file_uploaded=False)

                # Save the dataframe in a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', dir=app.config['UPLOAD_FOLDER'])
                df.to_csv(temp_file.name, index=False)

                # Store the new file path in the session
                session['temp_file_path'] = temp_file.name

                # Flash success message
                flash('File uploaded and validated successfully!', 'success')
                logging.info(f"User {session['username']} uploaded a valid file.")
                print(f"User {session['username']} uploaded a valid file.")
                return render_template('edit_dataset.html', message="File uploaded successfully!", file_uploaded=True)

            except Exception as e:
                error_messages.append(f"\n18. Error processing the file: {str(e)}")
                flash("\n".join(error_messages), 'error')
                return render_template('edit_dataset.html', message=f"Error processing the file: {str(e)}", file_uploaded=False)

    # If no file was uploaded or any error occurred
    if error_messages:
        flash("\n".join(error_messages), 'error')
        return render_template('edit_dataset.html', message="File validation failed", file_uploaded=False)

@app.route('/preview_dataset')
def preview_dataset():
    # Ensure we have a file path in the session
    if 'temp_file_path' not in session:
        flash('No dataset found to preview. Please upload a file first.', 'error')
        return redirect(url_for('edit_dataset'))
    
    logging.info(f"User {session['username']} opened the Preview Data page.")
    print(f"User {session['username']} opened the Preview Data page.")

    temp_file_path = session['temp_file_path']

    # Ensure the file exists at the stored path
    if not os.path.exists(temp_file_path):
        flash('No data found to preview. Please upload a valid file first.', 'error')
        return redirect(url_for('edit_dataset'))

    try:
        # Load the latest file stored in the session for preview
        df = pd.read_csv(temp_file_path)
        logging.info(f"User {session['username']} is previewing the uploaded file.")
        print(f"User {session['username']} is previewing the uploaded file.")
        return render_template('preview_dataset.html', data=df.to_html(classes='table table-striped'))
    except Exception as e:
        flash(f'Error displaying the data: {str(e)}', 'error')
        logging.info(f'Error displaying the data: {str(e)}')
        print(f'Error displaying the data: {str(e)}')
        return redirect(url_for('edit_dataset'))

@app.route('/confirm_upload', methods=['POST'])
def confirm_upload():

    logging.info(f"User {session['username']} is attempting to confirm data upload.")
    print(f"User {session['username']} is attempting to confirm data upload.")

    if 'temp_file_path' not in session or not os.path.exists(session['temp_file_path']):
        flash('No dataset is ready for upload.', 'error')
        return redirect(url_for('preview_dataset'))
    
    try:

        # Read the validated data from a temporary file
        df = pd.read_csv(session['temp_file_path'])
        
        # Convert column names to lowercase to ensure consistency
        df.columns = df.columns.str.strip().str.lower()

        # Rename the 'J.No.' column to 'j_num' to match the database column name
        if 'j.no.' in df.columns:
            df.rename(columns={'j.no.': 'j_num'}, inplace=True)
            df.rename(columns={'ref_type': 'design_ref'}, inplace=True)

        # Connect to your database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Prepare the insert statement
        insert_query = """
            INSERT INTO Dataset_table (
                Date_time_added, added_by_user, Prod_type, Cust_id, Prod_volume_mm3,
                Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type,
                Predicted_production_labour_time_mins, Actual_production_labour_time_mins,
                Predicted_CFP_time, Actual_CFP_time, Predicted_CFP_cost, Actual_CFP_cost,
                Accuracy_labour_time, Accuracy_labour_CFP_time, Accuracy_labour_CFP_cost, is_a_pred, feedback_provided,
                edited_on, J_Num, Design_ref_cat, Used_in_Retraining
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Create a list of data tuples from the dataframe
        data_tuples = [
            (
                datetime.now().strftime("%b %d %Y (%I:%M:%S %p)"),  # Current timestamp
                session['username'],
                row.get('prod_type', None),
                row.get('cust_id', None),
                row.get('prod_volume_mm3', None),
                row.get('prod_surface_area_mm2', None),
                row.get('prod_weight_gms', None),
                row.get('design_ref', None),
                row.get('assembly_type', None),
                row.get('rh_type', None),
                row.get('enamel_coat', None),
                row.get('ceramic_coat', None),
                row.get('alloy-1', None),
                row.get('metal-1', None),
                row.get('alloy-2', None),
                row.get('metal-2', None),
                row.get('pl_material_type', None),
                row.get('predicted_production_labour_time_mins', None),
                row.get('total_labour_time_minutes', None),
                row.get('predicted_cfp_time', None),
                row.get('cfp_time_minutes', None),
                row.get('predicted_cfp_cost', None),
                row.get('cfp_labour_cost_usd', None),
                row.get('accuracy_labour_time', None),
                # row.get('accuracy_labour_cost', None),
                row.get('accuracy_labour_cfp_time', None),
                row.get('accuracy_labour_cfp_cost', None),
                'No', # is_a_pred
                'NA', # feedback_provided
                None,  # edited_on, assuming None for now or can be set as current timestamp
                row.get('j_num', None),  # Mapped from 'J.No.',
                row.get('design_ref', None)[:-3],  # Slicing to remove the last 3 characters
                0 # Used_in_Retraining is set to 0 initially as the data is not used in retraining yet
            ) for index, row in df.iterrows()
        ]

        # Execute the query and commit changes
        logging.info(f"Uploading data of size {len(data_tuples)} to the database")
        print(f"Uploading data of size {len(data_tuples)} to the database")
        
        cursor.executemany(insert_query, data_tuples)
        conn.commit()
        cursor.close()
        conn.close()

        logging.info('Dataset successfully uploaded to the database.')
        print('Dataset successfully uploaded to the database.')

        flash('Dataset successfully uploaded to the database.', 'success')
    except Exception as e:
        logging.info(f"Dataset upload initiated by {session['username']} at {datetime.now().strftime('%b %d %Y (%I:%M:%S %p)')} failed due to the following error: {str(e)}")
        print(f"Dataset upload initiated by {session['username']} at {datetime.now().strftime('%b %d %Y (%I:%M:%S %p)')} failed due to the following error: {str(e)}")
        flash(f'Failed to upload dataset: {str(e)}', 'error')

    return redirect(url_for('preview_dataset'))

@app.route('/download_format')
def download_format():
    # Path to the file
    file_path = "C:\\Users\\800649\\Desktop\\ML Project\\My RK Flask App\\Media\\Dataset Upload Format.xlsx"
    
    try:
        return send_file(file_path, as_attachment=True, download_name='Dataset Upload Format.xlsx')
    except Exception as e:
        return str(e)
    
@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.get_json()

        # Extract the actual values and metadata from the request
        actual_labour_time = float(data.get('actual_labour_time') or 0.0)
        actual_cfp_time = float(data.get('actual_cfp_time') or 0.0)
        actual_cfp_cost = float(data.get('actual_cfp_cost') or 0.0)
        date_time = data.get('date_time')
        user_name = data.get('user_name') 

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch the predicted values from the database
        fetch_query = """
            SELECT Predicted_production_labour_time_mins, Predicted_CFP_time, Predicted_CFP_cost 
            FROM Dataset_table 
            WHERE Date_time_added = ? AND added_by_user = ?
        """
        cursor.execute(fetch_query, (date_time, user_name))
        result = cursor.fetchone()

        if result:
            predicted_labour_time, predicted_cfp_time, predicted_cfp_cost = result

            # Calculate accuracy values
            accuracy_labour_time = 1 - (abs(actual_labour_time - predicted_labour_time) / actual_labour_time) if actual_labour_time != 0 else 0
            accuracy_cfp_time = 1 - (abs(actual_cfp_time - predicted_cfp_time) / actual_cfp_time) if actual_cfp_time != 0 else 0
            accuracy_cfp_cost = 1 - (abs(actual_cfp_cost - predicted_cfp_cost) / actual_cfp_cost) if actual_cfp_cost != 0 else 0

            # Log accuracy for debugging
            print("Calculated Accuracies:", accuracy_labour_time, accuracy_cfp_time, accuracy_cfp_cost)

            # Update the database with actual values and calculated accuracy values
            update_query = """
                UPDATE Dataset_table
                SET Actual_production_labour_time_mins = ?, Actual_CFP_time = ?, Actual_CFP_cost = ?, 
                    feedback_provided = 'Yes', edited_on = ?, 
                    Accuracy_labour_time = ?, Accuracy_labour_CFP_time = ?, Accuracy_labour_CFP_cost = ?
                WHERE Date_time_added = ? AND added_by_user = ?
            """
            current_timestamp = datetime.now().strftime("%b %d %Y (%I:%M:%S %p)")
            cursor.execute(update_query, (
                actual_labour_time, actual_cfp_time, actual_cfp_cost,
                current_timestamp, accuracy_labour_time, accuracy_cfp_time, accuracy_cfp_cost,
                date_time, user_name
            ))
            conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()

        # Return success
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error while saving feedback: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/get_alloy", methods=["POST"])
def get_alloy():
    fg_metal_kt = request.form.get('fg_metal_kt')  # Fetching the FG_Metal_Kt value from the POST data

    conn = get_db_connection()
    cursor = conn.cursor()

    # Query to get Alloy for the selected FG_Metal_Kt
    query = "SELECT Alloy FROM MetalAlloys WHERE FG_Metal_Kt = ?"
    cursor.execute(query, (fg_metal_kt,))
    result = cursor.fetchall()

    cursor.close()
    conn.close()

    # Extract alloys and return them as a JSON response
    if result:
        alloys = [row[0] for row in result]
        return jsonify({'alloys': alloys})
    else:
        return jsonify({'alloys': []})  # Return an empty list if no alloys are found

def retrain_model_function(customer_id):
    
    global global_user_name
    # save_location = r'/Users/dipeshchandiramani/Desktop/Projects/RK_SunAI_repository-main/trained ML models'
    save_location = r'C:\Users\800649\Desktop\SunAI-main\trained ML models'

    if customer_id == "Generic":
        # Start the model retraining process for the Generic model
        start_generic_model_training(cust_id="Generic", target="CFP_time", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for CFP_time for Customer: {customer_id}")
        print(f"Retraining completed for CFP_time for Customer: {customer_id}")

        start_generic_model_training(cust_id="Generic", target="Total_time", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for Total_time for Customer: {customer_id}")
        print(f"Retraining completed for Total_time for Customer: {customer_id}")

        start_generic_model_training(cust_id="Generic", target="CFP_cost", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for CFP_cost for Customer: {customer_id}")
        print(f"Retraining completed for CFP_cost for Customer: {customer_id}")
    else:
        # Start the model retraining process with the chosen customer ID
        start_model_training(cust_id=customer_id, target="CFP_time", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for CFP_time for Customer: {customer_id}")
        print(f"Retraining completed for CFP_time for Customer: {customer_id}")

        start_model_training(cust_id=customer_id, target="Total_time", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for Total_time for Customer: {customer_id}")
        print(f"Retraining completed for Total_time for Customer: {customer_id}")

        start_model_training(cust_id=customer_id, target="CFP_cost", model_type='CatBoost', user=session['username'], save_to_location=save_location)
        logging.info(f"Retraining completed for CFP_cost for Customer: {customer_id}")
        print(f"Retraining completed for CFP_cost for Customer: {customer_id}")

@app.route('/get_customer_ids', methods=['GET'])
def get_customer_ids():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query to get unique Cust_id values
        cursor.execute("SELECT DISTINCT Cust_id FROM Dataset_table")
        rows = cursor.fetchall()
        
        # Extract Cust_id values from the query results
        customer_ids = [row[0] for row in rows]

        customer_ids.append("Generic")
        
        # Close the connection
        cursor.close()
        conn.close()

        # Return customer IDs as JSON
        return jsonify(customer_ids=customer_ids)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/")
def hello():
    logging.info("Accessed login page")
    return render_template('login.html')

@app.route("/login", methods=["POST"])
def login():
    global global_user_name
    global global_customer_id
    global global_is_admin

    username = request.form['username']
    global_user_name = username
    session['username'] = username
    
    logging.info(f"Login attempt for user: {username}")
    print(f"Login attempt for user: {username}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    cursor.execute("SELECT Cust_id_managed_by_user, is_admin FROM users WHERE username = ?", (username,))
    customer_id_row = cursor.fetchone()

    # Ensure that you have a result before trying to access the value
    if customer_id_row:
        global_customer_id = customer_id_row[0]     # Extract the first column (Cust_id_managed_by_user)
        global_is_admin = customer_id_row[1]        # Extract the second column (is_admin flag)
        session['customer_id'] = global_customer_id
        session['is_admin'] = global_is_admin
    else:
        # Handle the case where no result was found (user not found)
        session['customer_id'] = None
        session['is_admin'] = 0

    cursor.close()
    conn.close()
    
    if user and check_password_hash(user[3], request.form['password']):
        session['logged_in'] = True
        session['username'] = username  # Store the username in the session
        logging.info(f"User {session['username']} {'(ADMIN)' if session['is_admin'] else ''} managing customer {session['customer_id']} logged in successfully.")
        print(f"User {session['username']} {'(ADMIN)' if session['is_admin'] else ''} managing customer {session['customer_id']} logged in successfully.")
        return redirect(url_for('home'))
    else:
        logging.warning(f"Failed login attempt for user: {session['username']} managing customer_id {session['customer_id']}.")
        print(f"Failed login attempt for user: {session['username']} managing customer_id {session['customer_id']}.")
        flash("Login Failed. Please check your username and password.")
        return redirect(url_for('hello'))

@app.route("/home")
def home():
    if not session.get('logged_in'):
        logging.warning("Unauthorized access attempt to home page")
        return redirect(url_for('hello'))
    logging.info("Accessed home page")
    return render_template('home.html')

@app.route("/about")
def about():
    logging.info("Accessed about page")
    return render_template('about.html')

@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch unique Cust_id values from Dataset_table
    cursor.execute("SELECT DISTINCT Cust_id FROM Dataset_table")
    cust_ids = [row.Cust_id for row in cursor.fetchall()]

    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        selected_cust_id = request.form.get('customerSelect')  # Use .get() to avoid KeyError
        is_admin = request.form.get('admin')  # 'on' if checkbox is checked

        logging.info(f"Sign-up attempt for user: {username}")
        print(f"Sign-up attempt for user: {username}")

        if is_admin:  # Admin checkbox is checked
            admin_secret = request.form.get('admin_secret')
            if not admin_secret:
                flash("Admin secret key is required.")
                return redirect(url_for('sign_up'))

            if admin_secret != ADMIN_SECRET_KEY:
                flash("Invalid admin secret key.")
                return redirect(url_for('sign_up'))
            else:
                is_admin = 1  # Mark the user as admin
        else:
            is_admin = 0  # Mark the user as a regular user

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        try:
            cursor.execute(
                "INSERT INTO users (username, email, pass, is_admin, Cust_id_managed_by_user) VALUES (?, ?, ?, ?, ?)",
                (username, email, hashed_password, is_admin, selected_cust_id)
            )
            conn.commit()
            logging.info(f"User {username} signed up successfully")
            print(f"User {username} signed up successfully")
        except pyodbc.Error as e:
            logging.error(f"Error in database connection during sign-up: {e}")
            flash("An error occurred while signing up. Please try again.")
            return redirect(url_for('sign_up'))
        finally:
            cursor.close()
            conn.close()

        flash("Sign up successful! Please log in.")
        return redirect(url_for('hello'))
    
    # Pass cust_ids to the template for the dropdown
    cursor.close()
    conn.close()
    return render_template('sign-up.html', cust_ids=cust_ids)

@app.route('/get_dropdown_data', methods=['POST'])
def get_dropdown_data():
    cust_id = request.json.get('customer_id', None)  # Fetch customer_id from the request payload
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Base query to get distinct values
        query = """
            SELECT DISTINCT Prod_type, Cust_id, Design_ref, Assembly_type, RH_type, PL_material_type, [Metal-1], [Metal-2]
            FROM Dataset_table
            WHERE is_a_pred = 'No'
        """

        # If a specific customer ID is provided and is not "Generic," filter by Cust_id
        params = ()
        if cust_id and cust_id != "Generic":
            query += " AND Cust_id = ?"
            params = (cust_id,)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Prepare unique values for each field
        product_types = set()
        customer_ids = set()
        references = set()
        assembly_types = set()
        rhodium_plating_types = set()  # For RH_type
        metal_plating_types = set()    # For PL_material_type
        metal_1_values = set()  # For Metal-1
        metal_2_values = set()  # For Metal-2

        for row in rows:
            product_types.add(row[0])  # Prod_type
            customer_ids.add(row[1])   # Cust_id
            if not cust_id or cust_id == "Generic" or row[1] == cust_id:
                references.add(row[2])  # Design_ref
                assembly_types.add(row[3])  # Assembly_type
                rhodium_plating_types.add(row[4])  # RH_type
                metal_plating_types.add(row[5])  # PL_material_type
                metal_1_values.add(row[6])  # Metal-1
                metal_2_values.add(row[7])  # Metal-2

        # Append "Generic" to the customer ID options
        customer_ids.add("Generic")

        dropdown_data = {
            'product_types': sorted(product_types),
            'customer_ids': sorted(customer_ids),
            'references': sorted(references) if cust_id else [],
            'assembly_types': sorted(assembly_types) if cust_id else [],
            'rhodium_plating_types': sorted(rhodium_plating_types) if cust_id else [],
            'metal_plating_types': sorted(metal_plating_types) if cust_id else [],
            'metal_1_values': sorted(metal_1_values),
            'metal_2_values': sorted(metal_2_values)
        }

        logging.info("Fetched product types, customer IDs, references, metals, and other data successfully")
        print("Fetched product types, customer IDs, references, metals, and other data successfully")
        return jsonify(dropdown_data)
    except pyodbc.Error as e:
        logging.error(f"Error fetching data: {e}")
        print(f"Error fetching data: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route("/retrain_model")
def retrain_model():
    if not session.get('logged_in'):
        logging.warning("Unauthorized access attempt to retrain model page")
        return redirect(url_for('hello'))
    
    global global_user_name
    global global_customer_id
    global global_is_admin

    logging.info(f"{session['username']}{' (ADMIN)' if session['is_admin'] else ''} opened Dashboard page.")
    print(f"{session['username']}{' (ADMIN)' if session['is_admin'] else ''} opened Dashboard page.")

    # Default filter limit, can be updated based on request argument
    limit = request.args.get('limit', 25, type=int)  # Default to 25 if no limit provided

    conn = get_db_connection()
    cursor = conn.cursor()

    # Count the number of values
    cursor.execute("""
        SELECT COUNT(Accuracy_labour_CFP_cost)
        FROM Dataset_table
        WHERE Cust_id = ? AND is_a_pred = 'Yes' AND feedback_provided = 'Yes'
    """, (session['customer_id'],))

    pred_count = cursor.fetchone()
    pred_count = pred_count[0]
    
    # Calculate the model average accuracy
    cursor.execute("""
        SELECT Accuracy_labour_CFP_cost
        FROM Dataset_table
        WHERE Cust_id = ? AND is_a_pred = 'Yes' AND feedback_provided = 'Yes'
    """, (session['customer_id'],))

    # Get all rows from the query result
    cfp_accuracy_values = [row[0] for row in cursor.fetchall()]

    # Defining the initial avg_pred value
    avg_pred_calculated = 'N/A'
    
    # Calculate the median in Python
    if cfp_accuracy_values:
        if pred_count > 2:
            cfp_accuracy_values.sort()
            n = len(cfp_accuracy_values)
            avg_pred_calculated = round((cfp_accuracy_values[n // 2] if n % 2 != 0 else (cfp_accuracy_values[n // 2 - 1] + cfp_accuracy_values[n // 2]) / 2), 2)
            avg_pred_calculated = int(avg_pred_calculated * 100)
    else:
        avg_pred_calculated = 'N/A'  # Handle cases where there are no records

    # Fetch the last retrained date from Model_Registry for the specified global_customer_id
    cursor.execute("""
        SELECT TOP 1 Created_Date 
        FROM Model_Registry 
        WHERE Cust_id = ? AND Model_For = 'CFP_cost'
        ORDER BY Model_ID DESC
    """, (session['customer_id'],))
    last_retrained_date = cursor.fetchone()

    # Fetch the last dataset update date
    cursor.execute("""
        SELECT TOP 1 Date_time_added 
        FROM Dataset_table 
        WHERE Cust_id = ? AND feedback_provided = 'NA' AND is_a_pred = 'No'
        ORDER BY Row_id DESC
    """, (session['customer_id'],))
    last_dataset_updated_date = cursor.fetchone()
    print(last_dataset_updated_date)
    
    # Format the date to "DD MMM YYYY" if it exists, else set to "N/A"
    if last_retrained_date:
        
        # Calculate the next retraining date by adding 60 days
        next_retraining_date = last_retrained_date[0] + timedelta(days=60)
        
        # Format both dates as strings
        last_retrained_date = last_retrained_date[0].strftime("%b %d %Y")
        next_retraining_date = next_retraining_date.strftime("%b %d %Y")
    else:
        last_retrained_date = "N/A"
        next_retraining_date = "N/A"

    if last_dataset_updated_date:
        last_dataset_updated_date = last_dataset_updated_date[0][:-14]
    else:
        last_dataset_updated_date = "N/A"


    # Query for recent predictions based on admin or non-admin status
    if global_is_admin:
        query = f"""
        SELECT TOP ({limit})
            Date_time_added,
            added_by_user,
            Cust_id,
            Design_ref,
            Prod_type,
            Predicted_production_labour_time_mins,
            Predicted_CFP_time,
            Predicted_CFP_cost
        FROM dataset_table
        WHERE is_a_pred = 'Yes'
        ORDER BY Row_id DESC
        """
        cursor.execute(query)
        recent_predictions = cursor.fetchall()

        # Query for feedback data
        query_2 = """
        SELECT 
            Date_time_added,
            added_by_user,
            Cust_id,
            Design_ref,
            Prod_type,
            Predicted_production_labour_time_mins,
            Actual_production_labour_time_mins,
            Predicted_CFP_time,
            Actual_CFP_time,
            Predicted_CFP_cost,
            Actual_CFP_cost
        FROM dataset_table
        WHERE feedback_provided = 'No'
        """
        cursor.execute(query_2)
        feedback_data = cursor.fetchall()

    else:
        # Non-admin users, filter by specific Cust_id
        query = f"""
        SELECT TOP ({limit})
            Date_time_added,
            added_by_user,
            Cust_id,
            Design_ref,
            Prod_type,
            Predicted_production_labour_time_mins,
            Predicted_CFP_time,
            Predicted_CFP_cost
        FROM dataset_table
        WHERE is_a_pred = 'Yes' AND Cust_id = ?
        ORDER BY Row_id DESC
        """
        cursor.execute(query, (session['customer_id'],))
        recent_predictions = cursor.fetchall()

        # Query for feedback data with Cust_id filter
        query_2 = """
        SELECT 
            Date_time_added,
            added_by_user,
            Cust_id,
            Design_ref,
            Prod_type,
            Predicted_production_labour_time_mins,
            Actual_production_labour_time_mins,
            Predicted_CFP_time,
            Actual_CFP_time,
            Predicted_CFP_cost,
            Actual_CFP_cost
        FROM dataset_table
        WHERE feedback_provided = 'No' AND Cust_id = ?
        """
        cursor.execute(query_2, (session['customer_id']))
        feedback_data = cursor.fetchall()

    # Close the database connection
    cursor.close()
    conn.close()

    # Check if there are predictions to display
    if not recent_predictions:
        logging.info("No data found matching the criteria.")
        flash("No recent predictions found.", "info")


    #avg_pred_calculated = '19.88 %'

    # Pass recent predictions, feedback data, and last retrained date to the template
    return render_template('retrain_model.html', 
                           recent_predictions=recent_predictions, 
                           feedback_data=feedback_data,
                           last_retrained_date=last_retrained_date,
                           next_retraining_date=next_retraining_date,
                           last_dataset_updated_date=last_dataset_updated_date,
                           avg_pred_calculated=avg_pred_calculated)

@app.route('/edit_dataset')
def edit_dataset():
    try:
        logging.info(f"{session['username']} {'(Admin)' if session['is_admin'] else ''} opened the Edit Dataset page.")
        print(f"{session['username']} {'(Admin)' if session['is_admin'] else ''} opened the Edit Dataset page.")

        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to fetch unique Cust_id from Dataset_table
        cursor.execute("SELECT DISTINCT Cust_id FROM Dataset_table")
        customer_ids = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

    except Exception as e:
        customer_ids = []
        print(f"Error fetching customer ids: {e}")
    
    # Pass customer_ids to the template
    return render_template('edit_dataset.html', customer_ids=customer_ids)

@app.route('/download_dataset/<cust_id>')
def download_dataset_by_cust_id(cust_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to fetch the dataset uploaded till date (this does not include predictions as that is a separate feature)
        query = """
        SELECT J_Num, Prod_type, Cust_id, Prod_volume_mm3, 
                Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                Actual_production_labour_time_mins, Actual_CFP_time, Actual_CFP_cost
        FROM Dataset_table 
        WHERE Cust_id = ? AND is_a_pred = 'No' AND feedback_provided = 'NA'
        """

        cursor.execute(query, (cust_id,))
        data = cursor.fetchall()

        # Check if data was found
        if not data:
            flash(f"No data found for customer {cust_id}.", 'error')
            return redirect(url_for('edit_dataset'))

        # Log the shape of the data returned
        print(f"Fetched data shape: {len(data)} rows and {len(data[0]) if data else 0} columns")

        # Define the columns for the DataFrame in the correct order
        columns = ['J_Num', 'Prod_type', 'Cust_id', 'Prod_volume_mm3', 'Prod_surface_area_mm2',
                   'Prod_weight_gms', 'Design_ref', 'Assembly_type', 'RH_type', 'Enamel_coat',
                   'Ceramic_coat', 'Alloy-1', 'Metal-1', 'Alloy-2', 'Metal-2', 'PL_material_type',
                   'Actual_production_labour_time_mins', 'Actual_CFP_time', 'Actual_CFP_cost']

        # Convert the fetched tuples to a list of lists
        data = [list(row) for row in data]

        # Convert the data to a DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Save the DataFrame to an Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)

        output.seek(0)

        # Send the Excel file as a response
        return send_file(output, download_name=f'dataset_{cust_id}.xlsx', as_attachment=True,
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        flash(f"Error downloading dataset for customer {cust_id}: {e}", 'error')
        return redirect(url_for('edit_dataset'))
    
def load_model(model_for, cust_id):
    """
    Load the model for a specific model type (e.g., 'CFP_time', 'Total_time', or 'CFP_cost') 
    where IsDefault is set to 1 in the database.

    Parameters:
        model_for (str): The model type to load (e.g., 'CFP_time', 'Total_time', 'CFP_cost')

    Returns:
        Loaded model pipeline if found; None otherwise.
    """
    try:
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query to get the latest model path for the specified Model_For with IsDefault = 1
        query = """
        SELECT Model_Location 
        FROM Model_Registry
        WHERE Model_For = ? AND Cust_id = ? AND IsDefault = 1
        ORDER BY Model_ID DESC
        """
        
        cursor.execute(query, (model_for, cust_id))
        result = cursor.fetchone()

        # Check if a result was found
        if result:
            model_location = result[0]
            model = joblib.load(model_location)
            logging.info(f"Loaded model path for {model_for} from {model_location}")
            print(f"Loaded model path for {model_for} from {model_location}")
        else:
            logging.error(f"No path found for default model {model_for}")
            print(f"No path found for default model {model_for}")
            model = None

        # Close the cursor and connection
        cursor.close()
        conn.close()

        return model

    except FileNotFoundError:
        logging.error(f"Model file not found for {model_for}")
        print("Model file not found. Please ensure the model is trained and the file exists.")
        return None
    except Exception as e:
        logging.error(f"An error occurred while loading model for {model_for}: {e}")
        print(f"An error occurred: {e}")
        return None

@app.route("/ai_production_cost_estimate", methods=["GET", "POST"])
def ai_production_cost_estimate():
    global global_is_admin
    global global_user_name
    
    logging.info(f"{session['username']}{' (ADMIN)' if session['is_admin'] else ''} opened AI prediction page.")
    print(f"{session['username']}{' (ADMIN)' if session['is_admin'] else ''} opened AI prediction page.")
    if not session.get('logged_in'):
        logging.warning("Unauthorized access attempt to AI production cost estimate page")
        return redirect(url_for('hello'))

    if request.method == "POST" and request.form['customer_id'] != "Generic":
        logging.info("Processing AI production cost estimate")

        # Get the reference category from the form data
        ref_type = request.form['reference']
        ref_cat = ref_type[:-3]  # Slicing to remove the last 3 characters
        
        # Update the form data with new fields
        form_data = {
            'Prod_type': str(request.form['product_type']),
            'Cust_id': request.form['customer_id'],
            'Prod_volume_mm3': float(request.form['product_volume']),#total_volume,
            'Prod_surface_area_mm2': float(request.form['product_area']),#total_surface_area,
            'Prod_weight_gms': float(request.form['product_weight']),#prod_weight if not request.form['customer_id'] else request.form['customer_id'],
            'Design_ref': request.form['reference'],
            'Design_ref_category': ref_cat,
            'Assembly_type': request.form['assembly_type'],
            'RH_type': request.form['rhodium_plating_type'],
            'PL_material_type': request.form['metal_plating_type'],
            'Enamel_coat': request.form['hidden_enamel_coat'] if not request.form['enamel_coat'] else request.form['enamel_coat'],
            'Ceramic_coat': request.form['hidden_ceramic_coat'] if not request.form['ceramic_coat'] else request.form['ceramic_coat'],
            'Metal-1': request.form['metal-1'],
            'Alloy-1': request.form['alloy-1'],
            'Metal-2': request.form['metal-2'],
            'Alloy-2': request.form['alloy-2'],
        }
        logging.info(f"Form data: {form_data}")

        # Update form data with calculated ratios
        if form_data['Prod_surface_area_mm2'] != 0:
            form_data['Volume_Area_Ratio'] = form_data['Prod_volume_mm3'] / form_data['Prod_surface_area_mm2']
            form_data['Weight_Area_Ratio'] = form_data['Prod_weight_gms'] / form_data['Prod_surface_area_mm2']
        else:
            form_data['Volume_Area_Ratio'] = 0
            form_data['Weight_Area_Ratio'] = 0

        if form_data['Prod_volume_mm3'] != 0:
            form_data['Weight_Volume_Ratio'] = form_data['Prod_weight_gms'] / form_data['Prod_volume_mm3']
        else:
            form_data['Weight_Volume_Ratio'] = 0

        logging.info(f"Calculated ratios: Volume_Area_Ratio={form_data['Volume_Area_Ratio']}, Weight_Area_Ratio={form_data['Weight_Area_Ratio']}, Weight_Volume_Ratio={form_data['Weight_Volume_Ratio']}")

        try:
            # Load the time and cost prediction pipelines
            pipeline_CFP_time = load_model('CFP_time', form_data['Cust_id'])
            if pipeline_CFP_time is None:
                logging.error("Failed to load model for CFP_time")
                flash("Error: CFP_time model could not be loaded.")
                return redirect(request.url)

            pipeline_tot_time = load_model('Total_time', form_data['Cust_id'])
            if pipeline_tot_time is None:
                logging.error("Failed to load model for Total_time")
                flash("Error: Total_time model could not be loaded.")
                return redirect(request.url)

            pipeline_CFP_cost = load_model('CFP_cost', form_data['Cust_id'])
            if pipeline_CFP_cost is None:
                logging.error("Failed to load model for CFP_cost")
                flash("Error: CFP_cost model could not be loaded.")
                return redirect(request.url)

            logging.info("ML models loaded successfully")
        except FileNotFoundError:
            logging.error("Model file not found")
            flash("Model file not found. Please ensure the model is trained and the file exists.")
            return redirect(request.url)

        input_df = pd.DataFrame([form_data])
        logging.info(f"Input DataFrame:\n{input_df}")
        print(f"Input DataFrame:\n{input_df}")

        # Log each row separately
        for index, row in input_df.iterrows():
            logging.info(f"Row {index + 1}:")
            for column, value in row.items():
                logging.info(f"{column}: {value}")
            logging.info("\n")

        try:
            print("Starting with the First prediction")
            # Predict cfp labour time and add it to the input data
            predicted_CFP_time = pipeline_CFP_time.predict(input_df)[0]
            input_df['Actual_CFP_time'] = predicted_CFP_time
            print(f"Predicted{predicted_CFP_time}")
            
            print("Starting with the Second prediction")
            # Predict total labour time and add it to the input data
            predicted_tot_time = pipeline_tot_time.predict(input_df)[0]
            input_df['Actual_production_labour_time_mins'] = predicted_tot_time

            print("Starting with the Last prediction")
            # Finally predict cfp labour cost
            predicted_CFP_cost = pipeline_CFP_cost.predict(input_df)[0]

            # Add predictions to the logs
            logging.info(f"Predicted CFP labour time: {predicted_CFP_time}, Predicted total labour time: {predicted_tot_time}, Predicted CFP labour cost: {predicted_CFP_cost}")
            print(f"Predicted CFP labour time: {predicted_CFP_time}, Predicted total labour time: {predicted_tot_time}, Predicted CFP labour cost: {predicted_CFP_cost}")
        
        except ValueError as e:
            logging.error(f"Error in prediction: {e}")
            flash(f"Error in prediction: {e}")
            return redirect(request.url)

        output_df = pd.DataFrame([form_data])
        output_df['Predicted_CFP_time_minutes'] = predicted_CFP_time
        output_df['Predicted_Total_labour_time_minutes'] = predicted_tot_time
        output_df['CFP_labour_cost_USD'] = predicted_CFP_cost

        # Create an Excel file
        wb = Workbook()
        ws = wb.active
        ws.title = "Prediction"

        # Write the DataFrame to the Excel sheet
        for r in dataframe_to_rows(output_df, index=False, header=True):
            ws.append(r)

        # Save the DataFrame to an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Prediction')

        output.seek(0)

        # Verify content of the output before sending (optional)
        logging.debug("Excel file created successfully")
        print("Excel file created successfully")

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO Dataset_table (Date_time_added, edited_on, added_by_user, Prod_type, Cust_id, Prod_volume_mm3, Prod_surface_area_mm2, Prod_weight_gms, 
                           Design_ref, Design_ref_cat, Assembly_type, RH_type, Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type,
                           Predicted_production_labour_time_mins, Actual_production_labour_time_mins, Predicted_CFP_time, Actual_CFP_time, Predicted_CFP_cost, Actual_CFP_cost,
                           Accuracy_labour_time, Accuracy_labour_CFP_time, Accuracy_labour_CFP_cost, is_a_pred, feedback_provided, Used_in_Retraining)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%b %d %Y (%I:%M:%S %p)"), # timestamp
                None, # edited_on is set to NULL initially
                session['username'],
                form_data['Prod_type'],
                form_data['Cust_id'],
                form_data['Prod_volume_mm3'],
                form_data['Prod_surface_area_mm2'],
                form_data['Prod_weight_gms'],
                form_data['Design_ref'],
                form_data['Design_ref_category'],
                form_data['Assembly_type'],
                form_data['RH_type'],
                form_data['Enamel_coat'],
                form_data['Ceramic_coat'],
                form_data['Alloy-1'],
                form_data['Metal-1'],
                form_data['Alloy-2'],
                form_data['Metal-2'],
                form_data['PL_material_type'],
                predicted_tot_time,
                None,   # Initial value for Actual_production_labour_time_mins is set to NULL
                predicted_CFP_time,    # Test dummy value for Predicted_CFP_time (yet to create & integrate model)
                None,   # Initial value for Actual_CFP_time is set to NULL
                predicted_CFP_cost,    # Test dummy value for Predicted_CFP_cost (yet to create & integrate model)
                None,   # Initial value for Actual CFP Cost is set to NULL
                None,   # Accuracy_labour_time
                None,   # Accuracy_labour_CFP_time
                None,   # Accuracy_labour_CFP_cost
                'Yes',  # is_a_pred is set to 'Yes' for predictions
                'No',   # feedback_provided is set to 'No' initially
                0       # Used_in_Retraining is set to 0 initially
            ))
            conn.commit()
            logging.info("Data inserted into database successfully")
        except pyodbc.Error as e:
            logging.error(f"Error in database connection: {e}. \nData not inserted.")
            flash("An error occurred while saving the data to the database. Please try again.")
            return redirect(request.url)
        finally:
            cursor.close()
            conn.close()

        # Send the Excel file
        try:
            return send_file(
                output,
                download_name="prediction.xlsx",
                as_attachment=True,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            logging.error(f"Error sending file: {e}")
            flash(f"Error sending file: {e}")
            return redirect(request.url)

    return render_template('ai_production_cost_estimate.html')

@app.route("/process_data")
def process_data():
    if not session.get('logged_in'):
        logging.warning("Unauthorized access attempt to process data")
        return redirect(url_for('hello'))
    form_data = session.get('form_data')
    if not form_data:
        logging.warning("No form data found in session")
        return redirect(url_for('ai_production_cost_estimate'))

    with open('form_data.json', 'w') as f:
        json.dump(form_data, f)
    logging.info(f"Processed data: {form_data}")

    return f"Processing data: {form_data}"

@app.route("/get_serial_numbers", methods=["POST"])
def get_serial_numbers():
    customer_id = request.form['customer_id']
    print(customer_id)

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT Serial_nums FROM C_{customer_id}")
        serial_nums = [row[0] for row in cursor.fetchall()]
        logging.info(f"Fetched serial numbers for customer {customer_id}: {serial_nums}")
    except pyodbc.Error as e:
        # logging.error(f"Error in database connection: {e}")
        print(f"Error in database connection: {e}")
        return {"error": str(e)}
    finally:
        cursor.close()
        conn.close()

    return {"serial_nums": serial_nums}

@app.route('/retrain_model', methods=['POST'])
def retrain_model_endpoint():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        # Call the retrain function with the selected customer ID
        retrain_model_function(customer_id)  
        return jsonify({'message': 'Model retrained successfully!'}), 200
    except Exception as e:
        # Log the error
        logging.error(f"Error retraining model: {e}")
        print(f"Error retraining model: {e}")
        return jsonify({'error': 'Error retraining model.'}), 500
    
@app.route("/logout")
def logout():
    logging.info("User logged out")
    session['logged_in'] = False
    session.pop('form_data', None)
    return redirect(url_for('hello'))

if __name__ == "__main__":
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        conf.get_default().auth_token = "2gqyoMoMy0b61CBZ7IEym7upSDy_7GUceQDiVzJwXkLLyjWAq"
        try:
            ngrok_tunnel = ngrok.connect(5000)
            logging.info(f'ngrok URL: {ngrok_tunnel.public_url}')
        except Exception as e:
            logging.error(f"Failed to start ngrok tunnel: {e}")
            exit(1)

        def shutdown_ngrok():
            logging.info('Shutting down ngrok...')
            ngrok.disconnect(ngrok_tunnel.public_url)
            ngrok.kill()

        atexit.register(shutdown_ngrok)

    app.run(port=5000, debug=True)