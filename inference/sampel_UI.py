import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value': [10, 15, 8, 12, 9, 6],
    'Filter1': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Filter2': ['M', 'M', 'N', 'N', 'M', 'N']
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Bar Plot with Filters"),

    # Dropdown for Filter 1
    html.Label("Select Filter 1:"),
    dcc.Dropdown(
        id='filter1-dropdown',
        options=[
            {'label': f, 'value': f} for f in data['Filter1'].unique()
        ],
        value=data['Filter1'].unique()[0]
    ),

    # Multiselect dropdown for Filter 2
    html.Label("Select Filter 2:"),
    dcc.Dropdown(
        id='filter2-dropdown',
        options=[
            {'label': f, 'value': f} for f in data['Filter2'].unique()
        ],
        value=[data['Filter2'].unique()[0]],
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

    filtered_data = data[(data['Filter1'] == filter1_value) & (data['Filter2'].isin(filter2_value))]
    fig = px.bar(filtered_data, x='Category', y='Value', title='Filtered Bar Plot')
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

