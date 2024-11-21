import pyodbc
from config import Config

def get_db_connection():
    config = Config.DATABASE_CONFIG
    connection_string = (
        f"DRIVER={config['driver']};"
        f"SERVER={config['server']};"
        f"DATABASE={config['database']};"
        f"UID={config['username']};"
        f"PWD={config['password']}"
    )
    conn = pyodbc.connect(connection_string)
    return conn