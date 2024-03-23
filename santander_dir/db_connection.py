from sqlalchemy import event, create_engine, text
import logging
import struct
from azure.identity import DefaultAzureCredential
import os
import pandas as pd

def query_azure_db(query_text):
    db_name = "db_benchmark"
    db_server = "benchmark-demo"
    table_name = 'santander_raw'

    connection_string = f"mssql+pyodbc://{db_server}.database.windows.net/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
    azure_engine = create_engine(connection_string)

    @event.listens_for(azure_engine, "do_connect")
    def provide_token(dialect, conn_rec, cargs, cparams):
        SQL_COPT_SS_ACCESS_TOKEN = (
            1256  # Connection option for access tokens, as defined in msodbcsql.h
        )
        TOKEN_URL = (
            "https://database.windows.net/.default"  # The token URL for any Azure SQL database
        )

        azure_credentials = DefaultAzureCredential()
        raw_token = azure_credentials.get_token(TOKEN_URL).token.encode("utf-16-le")

        # remove the "Trusted_Connection" parameter that SQLAlchemy adds
        cargs[0] = cargs[0].replace(";Trusted_Connection=Yes", "")

        # create token credential
        azure_credentials = DefaultAzureCredential()
        raw_token = azure_credentials.get_token(TOKEN_URL).token.encode(
            "utf-16-le")
        token_struct = struct.pack(
            f"<I{len(raw_token)}s",
            len(raw_token),
            raw_token)

        # apply it to keyword arguments
        cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: token_struct}

    az_conn = azure_engine.connect()

    df = pd.read_sql_query(query_text, az_conn)
    # result = az_conn.execute(text(query_text))
    # rows = [row for row in result.fetchall()]

    # Fetch the column names
    # columns = result.keys()


    az_conn.close()

    return df








