# config.py
# import os

# class Config:
#     SECRET_KEY = os.environ.get('SECRET_KEY') or 'Rsec123key456K'
#     DATABASE_CONFIG = {
#         'driver': '{ODBC Driver 17 for SQL Server}',
#         'server': 'sunai.database.windows.net',
#         'database': 'SUN AI',
#         'username': 'sunai-admin',
#         'password': 'ai_app_pass*1',
#     }

import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'Rsec123key456K'
    DATABASE_CONFIG = {
        'driver': '{ODBC Driver 17 for SQL Server}',
        'server': 'sun-ai.chiy40yoaad1.us-east-2.rds.amazonaws.com',
        'database': 'SUN',
        'username': 'sunai_admin',
        'password': 'Sun75483pass',
        'port': '1433',
        'timeout': 30,
        'encrypt': 'yes',
        'TrustServerCertificate': 'yes'
    }

    DATABASE_CONFIG_SQLAlchemy = {
        'driver': 'ODBC+Driver+17+for+SQL+Server',
        'server': 'sun-ai.chiy40yoaad1.us-east-2.rds.amazonaws.com',
        'database': 'SUN',
        'username': 'sunai_admin',
        'password': 'Sun75483pass',
        'port': '1433',
        'timeout': 30,
        'encrypt': 'yes',
        'TrustServerCertificate': 'yes'
    }