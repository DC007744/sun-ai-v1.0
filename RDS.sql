-- Point to SUN database
USE [SUN];
GO

-- Show all tables
SELECT table_name, table_type FROM INFORMATION_SCHEMA.TABLES;
SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Dataset_table' AND TABLE_SCHEMA = 'SUN';

select * FROM MetalAlloys;

INSERT INTO MetalAlloys (FG_Metal_Kt, Alloy, Gravity)
VALUES ('Titanium', 'Ti', 4.5);

-- Rename table
-- EXEC sp_rename 'C_4121', 'Styles_4121';

-- Drop a table
-- DROP TABLE assembly_types;

-- UPDATE Dataset_table
-- SET is_a_pred = 'Yes'
-- WHERE row_id = 2;

-- Count columns
-- SELECT COUNT(*) AS Total_Columns
-- FROM INFORMATION_SCHEMA.COLUMNS
-- WHERE TABLE_NAME = 'Dataset_table';

-- View all columns in a table
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Dataset_table';

select * from users;
-- Add a column in a table
ALTER TABLE users
ADD Cust_id_managed_by_user nvarchar(255);

UPDATE users
SET Cust_id_managed_by_user = '4121'
WHERE id = 1;

-- Insert values into a column
-- UPDATE Dataset_table
-- SET is_a_pred = 'Yes';

-- Drop column(s)
-- ALTER TABLE users
-- DROP COLUMN Cust_ids_managed;

UPDATE Dataset_table
SET Actual_production_labour_time_mins = 192,
    Actual_CFP_time = 76,
    Actual_CFP_Cost = 33

UPDATE Dataset_table
SET is_a_pred = 'Yes'
WHERE Row_id > 1 AND Row_id < 240;

SELECT * FROM Dataset_table
WHERE Used_in_Retraining = 1;--Date_time_added = 'Oct 18 2024 (10:12 AM)' AND added_by_user = 'admin';

SELECT COUNT(*) AS total_count
FROM Dataset_table
WHERE is_a_pred = 'Yes';

-- Show a table's rows
SELECT * FROM Dataset_table
where Accuracy_labour_CFP_cost > 0;



DELETE FROM Dataset_table
WHERE Row_id < 2132;

-- Date_time_added column value of the table "Dataset_table" where the "is_a_pred" column value is "No"

select * from users;
                                                                         -- RK Dashboard --
SELECT * FROM Model_Registry
WHERE Cust_id = 4121;

CREATE TABLE Dashboard_Stats (
    Stat_ID INT IDENTITY(1,1) PRIMARY KEY,
    Row_value_updated_on DATETIME DEFAULT GETDATE(),   -- When the row was last updated
    Last_Retrained_Date DATE,                          -- Last retraining date
    Next_Retraining_Date DATE,                         -- Next scheduled retraining date
    Dataset_Last_Updated DATE,                         -- When the dataset was last updated
    Average_Prediction_Accuracy FLOAT                  -- Average accuracy of predictions
);


CREATE TABLE Model_Registry (
    Model_ID INT IDENTITY(1,1) PRIMARY KEY,          -- Unique row id
    Model_For VARCHAR(255),                          -- Model can be for Total_time, CFP_cost, or CFP_time
    Model_Name VARCHAR(255),                         -- Actual name of the model
    Model_Location VARCHAR(255),                     -- Model path in Azure (local for now during development)
    Created_Date DATETIME DEFAULT GETDATE(),         -- Date for model creation
    IsDefault BIT DEFAULT 0,                         -- Whether the model is currently being used for prediction
    Model_Accuracy FLOAT,                            -- R2 score of the model
    Model_Log_File_Location VARCHAR(255)             -- Model log file path in Azure (local for now during development)
);

ALTER TABLE Model_Registry
ADD Cust_id VARCHAR(255);

-- UPDATE Model_Registry
-- SET IsDefault = 1
-- WHERE Model_ID > 63;-- AND Model_ID < 33;

SELECT * FROM Model_Registry;
SELECT * FROM Model_Parameters;
SELECT * FROM Dataset_table;

UPDATE Model_Parameters
SET Model_For = 'Total_time'
WHERE Parameter_ID = 12;

UPDATE Model_Parameters
SET Model_Type = 'CatBoost'
WHERE Parameter_ID = 7;

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type)
VALUES (1, 'depth', 7, 'CFP_cost', 'CatBoost');

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type)
VALUES (1, 'iterations', 377, 'CFP_cost', 'CatBoost');

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type)
VALUES (1, 'learning_rate', 0.27, 'CFP_cost', 'CatBoost');

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type)
VALUES (1, 'l2_leaf_reg', 1.37, 'CFP_cost', 'CatBoost');

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value, Model_For, Model_Type)
VALUES (1, 'subsample', 0.7, 'CFP_cost', 'CatBoost');

--ADD Model_Type VARCHAR(255);-- forign_key;


CREATE TABLE Model_Parameters (
    Parameter_ID INT IDENTITY(1,1) PRIMARY KEY,						-- Unique row ID for each parameter entry
    Model_ID INT FOREIGN KEY REFERENCES Model_Registry(Model_ID),	-- Foreign key to link with Model_Registry
    Param_Key VARCHAR(255),											-- The name of the parameter (e.g., 'depth', 'iterations')
    Param_Value VARCHAR(255),										-- The value of the parameter (stored as a string for flexibility)
    CONSTRAINT FK_Model_Parameters_ModelID FOREIGN KEY (Model_ID) REFERENCES Model_Registry(Model_ID)
);


-- TEST 3 --

-- Insert into Model_Registry
INSERT INTO Model_Registry (Model_For, Model_Name, Model_Type, Model_Location, Model_Accuracy, Model_Log_File_Location, Created_Date, IsDefault)
VALUES ('CFP_cost', 'Generic_CFP_cost_CatBoost_pipeline.joblib', 'Catboost', 'C:\Users\800649\Desktop\SunAI-main\trained ML models\Generic_Trained_Model_Version-Nov-12-2024_20H.35M_PM\Generic_CFP_cost_CatBoost_pipeline.joblib', 93.89, 'C:/path/to/logfile', GETDATE(), 1);

-- Retrieve the most recent Model_ID using IDENT_CURRENT
SELECT IDENT_CURRENT('Model_Registry') AS Model_ID;



UPDATE Model_Registry
SET Model_Location = 'C:\Users\800649\Desktop\SunAI-main\trained ML models\4273_Trained_Model_Version-Nov-07-2024_08H.03M_AM\4273_CFP_cost_CatBoost_pipeline.joblib'
WHERE Model_ID = 9;

select * from Model_Registry






ALTER TABLE Model_Parameters
ADD Model_For VARCHAR(255);

-- Insert example rows into the Model_Registry table
INSERT INTO Model_Registry (Model_For, Model_Name, Model_Location, Created_Date, IsDefault, Model_Accuracy, Model_Log_File_Location)
VALUES
('Total_time', 'Total_time_model_v1', 'local_path_to_model_v1', GETDATE(), 1, 0.89, 'local_path_to_log_v1'),   -- Model 1
('CFP_cost', 'CFP_cost_model_v2', 'local_path_to_model_v2', GETDATE(), 0, 0.92, 'local_path_to_log_v2'),       -- Model 2
('CFP_time', 'CFP_time_model_v3', 'local_path_to_model_v3', GETDATE(), 0, 0.88, 'local_path_to_log_v3');       -- Model 3


SELECT * FROM Model_Registry;

SELECT * FROM Dataset_table;

UPDATE Dataset_table
SET is_being_used_for_retraining = 'No';

INSERT INTO Model_Parameters (Model_ID, Param_Key, Param_Value)
VALUES
    (1, 'depth', '6'),
    (1, 'iterations', '378'),
    (1, 'learning_rate', '0.20'),
    (1, 'l2_leaf_reg', '1.30'),
    (1, 'subsample', '0.8');

SELECT * FROM Model_Parameters;
                                                                         -- RK Dashboard --

select * from Dashboard_Stats;

UPDATE Dashboard_Stats
SET Last_Retrained_Date
WHERE Row_id > 9;

alter table Dashboard_Stats
drop column Average_Prediction_Accuracy;

alter table Dashboard_Stats
add Retraining_in_progress varchar(3);

ALTER TABLE Dataset_table
ADD is_being_used_for_retraining varchar(3) DEFAULT 'No';

--delete from Dataset_table
--where Row_id > 18;

SELECT * FROM Dataset_table
WHERE is_being_used_for_retraining = 'Yes'

SELECT  Prod_type, Cust_id, Prod_volume_mm3, 
                        Prod_surface_area_mm2, Prod_weight_gms, Design_ref, Assembly_type, RH_type,
                        Enamel_coat, Ceramic_coat, [Alloy-1], [Metal-1], [Alloy-2], [Metal-2], PL_material_type, 
                        Actual_CFP_time
                FROM Dataset_table
                WHERE is_being_used_for_retraining = 'Yes'



UPDATE Dataset_table
SET Cust_id = 4121
WHERE Row_id > 2 AND Row_id < 558; -- AND Row_id < 1193;

-- Creating the table
CREATE TABLE MetalAlloys (
    FG_Metal_Kt VARCHAR(50),
    Alloy VARCHAR(50),
    Gravity DECIMAL(5, 2)
);

INSERT INTO MetalAlloys (FG_Metal_Kt, Alloy, Gravity)
VALUES ('NOM2', 'NOAL2', 0);

-- Inserting the data
INSERT INTO MetalAlloys (FG_Metal_Kt, Alloy, Gravity) VALUES
('10kt Pink', 'UPMR 540', 11.55),
('10kt Pink', 'UPMR 548', 11.52),
('10kt White', 'UPMR 860', 11.23),
('10kt White', 'UPMR 915', 11.15),
('10kt White', 'UPMR 943', 11.5),
('10kt White', 'UPMR 980', 12.37),
('10kt White', 'UPMR 982', 11.55),
('10kt White', 'UPMR F1', 12.74),
('10kt White', 'UPMR M-930', 11.08),
('10kt White', 'UPMR NAUA1', 11.62),
('10kt White', 'LegOr WB143C', 11.3),
('10kt Yellow', 'UPMR 200', 11.35),
('10kt Yellow', 'UPMR 278', 11.39),
('10kt Yellow', 'UPMR 2SA', 11.6),
('10kt Yellow', 'LegOr SCA1V', 11.3),
('14kt Pink', 'UPMR 540', 13.05),
('14kt Pink', 'UPMR 548', 13.02),
('14kt White', 'UPMR 860', 12.7),
('14kt White', 'UPMR 915', 12.68),
('14kt White', 'UPMR 943', 13),
('14kt White', 'UPMR 982', 13.05),
('14kt White', 'UPMR F1', 14.14),
('14kt White', 'UPMR M-930', 12.62),
('14kt White', 'UPMR NAUA1', 13.11),
('14kt White', 'LegOr WB486CW', 14.4),
('14kt White', 'LegOr WA12B1 PD', 14.4),
('14kt Yellow', 'UPMR 200', 12.87),
('14kt Yellow', 'UPMR 278', 12.9),
('14kt Yellow', 'UPMR 2SA', 13.09),
('14kt Yellow', 'LegOr SCA1V', 12.9),
('18kt Pink', 'UPMR 534', 14.94),
('18kt Pink', 'UPMR 540', 15),
('18kt Pink', 'UPMR 548', 14.97),
('18kt Pink', 'PANDORA Extra 5N45', 15.7),
('18kt Pink', 'LegOr OR 133', 11.3),
('18kt Pink', 'LegOr OR 134', 14.7),
('18kt White', 'UPMR 860', 14.96),
('18kt White', 'UPMR 915', 14.7),
('18kt White', 'UPMR 982', 15),
('18kt White', 'UPMR M-930', 14.65),
('18kt White', 'UPMR NAUA1', 15.04),
('18kt White', 'LegOr NF509', 15.7),
('18kt White', 'LegOr NF510', 15.7),
('18kt White', 'LegOr NF511', 15.7),
('18kt White', 'LegOr NF512', 15.8),
('18kt White', 'LegOr NI1811-03', 14.6),
('18kt White', 'UPMR PD-11', 15.81),
('18kt White', 'LegOr WF1487C', 14.8),
('18kt Yellow', 'UPMR 713', 15.56),
('18kt Yellow', 'UPMR 716', 15.41),
('18kt Yellow', 'UPMR 720', 15.4),
('18kt Yellow', 'UPMR 790', 15.35),
('18kt Yellow', 'LegOr A18VN', 15.5),
('18kt Yellow', 'LegOr C183N', 15),
('18kt Yellow', 'LegOr C183N1', 15),
('18kt Yellow', 'LegOr C183NL', 15),
('18kt Yellow', 'LegOr C18VN', 15.3),
('Silver', 'LegOr AG108M', 10.3),
('Silver', 'LegOr AG109M', 10.3),
('Silver', 'LegOr AG115M', 10.26),
('Silver', 'LegOr ALPCAST', 8.4),
('Silver', 'UPMR S-86', 10.25),
('Silver', 'LegOr S925NP', 10.3),
('Silver', 'UPMR S-95', 10.34),
('Silver', 'UPMR S-97NA', 10.28),
('Brass', 'LegOr OTT30SC', 8.4),
('Platinum', 'Legor PT950COS', 21.55),
('Bronze', 'Belmont Everdur Silicon Bronze 4951', 8.36);

-- Show a table's rows in a particular order
SELECT 
    timestamp,
    Year_added_to_dataset,
	by_user,
    Prod_type,
    Cust_id,
    Prod_volume_mm3,
    Prod_surface_area_mm2,
    Prod_weight_gms,
    ref_type,
    Assembly_type,
    Base_metal,
    RH_type,
    Predicted_total_production_time_mins,
    Actual_total_production_time_mins,
    Predicted_total_production_cost_usd,
    Actual_total_production_cost_usd,
    Predicted_CFP_cost,
    Actual_CFP_cost,
    metal_plating_type,
    enamel_coat,
    ceramic_coat,
    metal_1,
    alloy_1,
    metal_2,
    alloy_2,
    C_time,
    F_time,
    A_time,
    S_time,
    P_time,
    Other_time
FROM 
    ai_production_cost_estimate;


-- ---------------------------------------Reseed DB Row_id Counter-------------------------------------
-- DBCC CHECKIDENT ('Dataset_table', RESEED, 0);
select * from Dataset_table

SELECT *
INTO Model_Parameters_copy
FROM Model_Parameters;

--UPDATE Dataset_table
--SET is_being_used_for_retraining = 'No'

-- ********************* Resetting Row_id 

select * from Dataset_table ORDER BY Row_id ASC;
select * from Model_Registry;
select * from Model_Parameters;

DBCC CHECKIDENT ('Model_Registry', RESEED, 0);

-- Create a new table with the same structure as your original table but without the Row_id as an identity column
CREATE TABLE Temp_Dataset_table AS
SELECT 
    ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS Row_id, 
    * 
FROM Dataset_table
ORDER BY Row_id;

select * from Dataset_table;
DBCC CHECKIDENT ('Dataset_table', RESEED, 0);

CREATE TABLE Dataset_table (
	Row_id INT IDENTITY(1,1) PRIMARY KEY,
	Date_time_added NVARCHAR(255),
	added_by_user VARCHAR(255),
    Prod_type VARCHAR(255),
    Cust_id NVARCHAR(255),
    Prod_volume_mm3 FLOAT,
    Prod_surface_area_mm2 FLOAT,
    Prod_weight_gms FLOAT,
    Design_ref NVARCHAR(255),
    Assembly_type VARCHAR(255),
    RH_type VARCHAR(255),
    Enamel_coat VARCHAR(255),
    Ceramic_coat VARCHAR(255),
    Alloy_1 NVARCHAR(255),
    Metal_1 NVARCHAR(255),
    Alloy_2 NVARCHAR(255),
    Metal_2 NVARCHAR(255),
    PL_material_type VARCHAR(255),
	Predicted_production_labour_time_mins FLOAT,
    Actual_production_labour_time_mins FLOAT,
	Predicted_production_labour_cost_usd FLOAT,
    Actual_production_labour_cost_usd FLOAT,
	Predicted_CFP_time FLOAT,
    Actual_CFP_time FLOAT,
	Predicted_CFP_cost FLOAT,
    Actual_CFP_cost FLOAT,
    C_time FLOAT,
    F_time FLOAT,
    A_time FLOAT,
    S_time FLOAT,
    P_time FLOAT,
    Other_time FLOAT,
	Accuracy_labour_time FLOAT,
	Accuracy_labour_cost FLOAT,
	Accuracy_labour_CFP_time FLOAT,
	Accuracy_labour_CFP_cost FLOAT
);


select * from Dataset_table;

-- Rename column
EXEC sp_rename 'Dashboard_stats.Row_value_updated_on', 'timestamp', 'COLUMN';


-- Update column datatype
-- ALTER TABLE ai_production_cost_estimate
-- ALTER COLUMN timestamp NVARCHAR(255);

select * from Dataset_table;

UPDATE Dataset_table
SET Cust_id = '4121';
--WHERE Prod_type = 'RING'

-- Create Users table
CREATE TABLE users (
    id INT PRIMARY KEY,
    username NVARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    pass VARCHAR(255) NOT NULL,
    is_admin INT CHECK (is_admin IN (0, 1)),
    created_at DATETIME DEFAULT GETDATE()
);

-- Create AI production cost estimate table
CREATE TABLE Predictions_table (
    row_id VARCHAR(255) PRIMARY KEY,
    Prod_type VARCHAR(50),
    Cust_id VARCHAR(50),
    Prod_volume_mm3 FLOAT,
    Prod_surface_area_mm2 FLOAT,
    Prod_weight_gms FLOAT,
    Design_ref VARCHAR(50),
    Assembly_type VARCHAR(50),
    RH_type VARCHAR(50),
    Predicted_Assembly_Labor_Time_Mins FLOAT,
    Predicted_Assembly_Labor_Cost_USD FLOAT,
    Actual_cost FLOAT,
    metal_plating_type VARCHAR(255),
    enamel_coat VARCHAR(255),
    ceramic_coat VARCHAR(255),
    metal_1 VARCHAR(255),
    alloy_1 VARCHAR(255),
    metal_2 VARCHAR(255),
    alloy_2 VARCHAR(255),
    C_time FLOAT,
    F_time FLOAT,
    A_time FLOAT,
    S_time FLOAT,
    P_time FLOAT,
    Other_time FLOAT,
    C_cost FLOAT,
    F_cost FLOAT,
    A_cost FLOAT,
    S_cost FLOAT,
    P_cost FLOAT,
    Other_cost FLOAT
);


-- Create Assembly types table
CREATE TABLE assembly_types (
    id INT PRIMARY KEY NOT NULL,
    Assembly_type VARCHAR(100)
);

INSERT INTO assembly_types (id, Assembly_type) VALUES (1, 'CHAIN');
INSERT INTO assembly_types (id, Assembly_type) VALUES (2, 'OMEGA');
INSERT INTO assembly_types (id, Assembly_type) VALUES (3, 'POST');
INSERT INTO assembly_types (id, Assembly_type) VALUES (4, 'POST & BUTTERFLY');
INSERT INTO assembly_types (id, Assembly_type) VALUES (5, 'POST & CHAIN');
INSERT INTO assembly_types (id, Assembly_type) VALUES (6, 'STEEL BEAD STRING WIRE');
INSERT INTO assembly_types (id, Assembly_type) VALUES (7, 'No Assembly');

-- Create Customer Sr Numbers table
CREATE TABLE C_4121 (
    Serial_nums VARCHAR(50)
);


-- Create Metal Density table
CREATE TABLE metal_density_table (
    id VARCHAR(3) PRIMARY KEY NOT NULL,
    metal_name VARCHAR(100) NOT NULL,
    purity_percentage FLOAT NOT NULL,
    code VARCHAR(3) NOT NULL,
    density FLOAT NOT NULL,
    base_ppg FLOAT NOT NULL
);

INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D1', 'Palladium 950', 100, 'PD', 12.1, 34.27);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D10', '18kt White Gold', 75, '18W', 18.1, 48.05);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D11', '9kt White Gold', 37.5, '9W', 13.75, 24.02);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D12', '10kt Yellow Gold', 41.67, '10Y', 14.85, 26.69);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D13', '14kt Yellow Gold', 58.5, '14Y', 15.95, 37.48);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D14', '18kt Yellow Gold', 75, '18Y', 18.1, 48.05);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D15', '9kt Yellow Gold', 37.5, '9Y', 13.75, 24.02);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D2', 'Platinum 950', 100, 'PT', 21.55, 29.93);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D3', '10kt Rose Gold', 41.67, '10R', 14.85, 26.69);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D4', '14kt Rose Gold', 58.5, '14R', 15.95, 37.48);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D5', '18kt Rose Gold', 75, '18R', 18.1, 48.05);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D6', '9kt Rose Gold', 37.5, '9R', 13.75, 24.02);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D7', '14kt White Gold', 58.5, '14W', 15.95, 37.48);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D8', '10kt White Gold', 41.67, '10W', 14.85, 26.69);
INSERT INTO metal_density_table (id, metal_name, purity_percentage, code, density, base_ppg) VALUES ('D9', '14kt White Gold', 58.5, '14W', 15.95, 37.48);


-- Create Product types table
CREATE TABLE prod_types (
    id INT PRIMARY KEY NOT NULL,
    Prod_type VARCHAR(50)
);

select * from Dataset_table;

delete from Dataset_table
where Row_id >1642;

INSERT INTO prod_types (id, Prod_type) VALUES (1, 'Ring');
INSERT INTO prod_types (id, Prod_type) VALUES (2, 'Pendant');
INSERT INTO prod_types (id, Prod_type) VALUES (3, 'Earring');
INSERT INTO prod_types (id, Prod_type) VALUES (4, 'Necklace');
INSERT INTO prod_types (id, Prod_type) VALUES (5, 'Bracelet');
INSERT INTO prod_types (id, Prod_type) VALUES (6, 'Bangle');
INSERT INTO prod_types (id, Prod_type) VALUES (7, 'Charm');