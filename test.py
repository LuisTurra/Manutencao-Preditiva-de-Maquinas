# test.py
from dash import Dash, html, dcc,Input, Output
import plotly.graph_objects as go

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("TESTE CALLBACK"),
    html.Button("CLIQUE AQUI", id="btn", n_clicks=0),
    html.Div(id="output"),
    dcc.Graph(id="gauge")
])

@app.callback(
    Output("output", "children"),
    Output("gauge", "figure"),
    Input("btn", "n_clicks")
)
def update(n):
    return f"Você clicou {n} vezes!", go.Figure(go.Indicator(
        mode="gauge+number",
        value=n * 10,
        title={'text': "CLICKS x10"},
        gauge={'axis': {'range': [0, 100]}}
    ))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)