import yaml
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

data_config_path = '../feature_engineering/feature_engineering.yaml'
with open(data_config_path, 'r') as file:
     data_config_dict = yaml.safe_load(file)

data = pd.read_csv('sample_val_result.csv')


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Product purchase probability"),

    # Dropdown for Filter 1
    html.Label("Select a customer code:"),
    dcc.Dropdown(
        id='filter1-dropdown',
        options=[
            {'label': f, 'value': f} for f in data['customer_code'].unique()
        ],
        value=data['customer_code'].unique()[0]
    ),

    # Multiselect dropdown for Filter 2
    html.Label("Select products:"),
    dcc.Dropdown(
        id='filter2-dropdown',
        options=[
            {'label': f, 'value': f} for f in data_config_dict['product_columns']
        ],
        value=data_config_dict['product_columns'],
        multi=True
    ),

    # Graph
    dcc.Graph(id='bar-plot')
])


# Define callback to update the bar plot
@app.callback(
    Output('bar-plot', 'figure'),
    [Input('filter1-dropdown', 'value'),
     Input('filter2-dropdown', 'value')]
)
def update_bar_plot(filter1_value, filter2_value):
    if not isinstance(filter2_value, list):
        filter2_value = [filter2_value]

    filtered_data = data[data['customer_code'] == filter1_value]
    filtered_data = filtered_data[filter2_value].iloc[0].tolist()

    fig = px.bar(filtered_data, x=filter2_value, y=filtered_data)
    fig.update_layout(yaxis=dict(title='Probability of purchase'))

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

