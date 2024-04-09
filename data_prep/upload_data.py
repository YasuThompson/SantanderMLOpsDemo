from sqlalchemy import event, create_engine, text
import logging
import struct
from azure.identity import DefaultAzureCredential
import os
import pandas as pd

def query_azure_db():
    db_name = "database"
    db_server = "benchmark-demo"
    table_name = 'santander_raw_original'

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

    train_date_list = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28', '2015-06-28', '2015-07-28',
                       '2015-08-28', '2015-09-28', '2015-10-28', '2015-11-28', '2015-12-28', '2016-01-28']

    # train_date_list = ['2015-02-28', ]

    df = pd.read_csv('../data/train_ver2.csv')

    df = df[df['fecha_dato'].isin(train_date_list)]

    df['sequential_index'] = df.reset_index().index


    interval = 10000
    # for i in range(0, len(df), interval):
    for i in range(3640000, 4000000, interval):
        print('Loading row {}:{}'.format(i, i + interval))
        df_extracted = df.iloc[i:i + interval]

        # df_extracted.to_sql(name=table_name, con=az_conn, if_exists='append', index=False)
        try:
            df_extracted.to_sql(name=table_name, con=az_conn, if_exists='append', index=False)
            print('Successfully loaded')
        except:
            print('Failed in loading')

    az_conn.close()











if __name__ == '__main__':
    # TODO: to implement a script which pushes a monthly record to a database to trigger the ML pipeline
    # TODO: Maybe with SQLAlchemy
    query_azure_db()




    # df = pd.read_csv('../data/train_ver2.csv')
    # df = pd.read_csv('../data/sample.csv')
    # df['unique_id'] = df.reset_index().index

    pass