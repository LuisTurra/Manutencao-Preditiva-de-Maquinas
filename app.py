# app.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from faker import Faker
import joblib
import os

from src.preprocess import load_data, add_rul, feature_engineering, scale
from src.model import train, evaluate

# === CARREGA E TREINA (só na 1ª vez) ===
if not os.path.exists("model.pkl"):
    print("Treinando modelo...")
    train_df, test_df, true_rul = load_data()
    train_df = add_rul(train_df)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    # Último ciclo do test
    last_cycle = test_df.groupby('unit')['cycle'].max().reset_index()
    test_df = test_df.merge(last_cycle, on=['unit', 'cycle'])

    X_train, X_test, scaler, feats = scale(train_df.drop(['RUL'], axis=1), test_df.drop(['unit'], axis=1))
    y_train = train_df['RUL']
    y_test = true_rul

    model = train(X_train, y_train)
    rmse = evaluate(model, X_test, y_test)
    print(f"RMSE: {rmse:.2f}")

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feats, "feats.pkl")
else:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feats = joblib.load("feats.pkl")

# === DASH ===
app = dash.Dash(__name__, suppress_callback_exceptions=True)
fake = Faker()

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '3%'}, children=[
    html.H1("Manutenção Preditiva Turbofan", style={'textAlign': 'center'}),
    html.Button("Gerar Dados IoT", id="btn", n_clicks=0, style={'padding': '12px', 'fontSize': '18px'}),
    html.Div(id="sensor-info", style={'margin': '20px', 'fontSize': '16px'}),
    dcc.Graph(id="gauge"),
    html.Div(id="alert", style={'fontSize': '22px', 'fontWeight': 'bold', 'marginTop': '20px'})
])
@app.callback(
    Output("gauge", "figure"),
    Output("sensor-info", "children"),
    Output("alert", "children"),
    Input("btn", "n_clicks")
)
def update(n_clicks):
    # Simula dados de sensores
    import numpy as np
    sample = pd.DataFrame({
        'sensor1': [np.random.uniform(0.3, 0.9)],
        'sensor2': [np.random.uniform(0.4, 0.8)],
        'sensor3': [np.random.uniform(0.5, 0.7)]
    })

    # Simula RUL (valor aleatório entre 20 e 180)
    rul = np.random.uniform(20, 180)

    # Gráfico de medidor
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul,
        title={'text': "RUL (ciclos)"},
        gauge={
            'axis': {'range': [0, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "orange"},
                {'range': [100, 200], 'color': "green"}
            ]
        }
    ))

    # Info dos sensores
    info = html.Ul([html.Li(f"{k}: {v:.3f}") for k, v in sample.iloc[0].items()])

    # Alerta
    if rul < 50:
        alert = html.Span("Manutenção Urgente!", style={'color': 'red', 'fontWeight': 'bold'})
    elif rul < 100:
        alert = html.Span("Atenção", style={'color': 'orange', 'fontWeight': 'bold'})
    else:
        alert = html.Span("Excelente", style={'color': 'green', 'fontWeight': 'bold'})

    return fig, info, alert
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=True) 