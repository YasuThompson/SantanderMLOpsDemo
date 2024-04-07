import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

# Azure SQL Database connection details
server =  os.getenv('SERVER')
database = os.getenv('DATABASE')
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')
driver= '{ODBC Driver 17 for SQL Server}'


def query_from_db(query_text):

    # Establishing the connection
    conn = pyodbc.connect(f'SERVER={server};DATABASE={database};UID={username};PWD={password};DRIVER={driver}')

    # Creating a cursor
    cursor = conn.cursor()

    # Executing the query
    cursor.execute(query_text)

    rows = cursor.fetchall()
    columns = [column[0] for column in cursor.description]
    # data = pd.DataFrame.from_records(rows, columns=columns)

    # Closing cursor and connection
    cursor.close()
    conn.close()

    return columns, rows


if __name__=="__main__":
    table_name = 'santander_cleaned_small'
    query = f"SELECT TOP(100) * FROM {table_name}"
    columns, rows = query_from_db(query)


