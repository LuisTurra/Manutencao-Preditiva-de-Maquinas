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
app = dash.Dash(__name__)
fake = Faker()

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '3%'}, children=[
    html.H1("Manutenção Preditiva Turbofan", style={'textAlign': 'center'}),
    html.Button("Gerar Dados IoT", id="btn", n_clicks=0, style={'padding': '12px', 'fontSize': '18px'}),
    html.Div(id="sensor-info", style={'margin': '20px', 'fontSize': '16px'}),
    dcc.Graph(id="gauge"),
    html.Div(id="alert", style={'fontSize': '22px', 'fontWeight': 'bold', 'marginTop': '20px'})
])

@app.callback(
    [Output("sensor-info", "children"), Output("gauge", "figure"), Output("alert", "children")],
    Input("btn", "n_clicks")
)
def update(n):
    # Simula sensores
    sample = pd.DataFrame({col: [np.random.uniform(0.3, 0.9)] for col in feats[:5]})
    scaled = scaler.transform(sample)
    rul = model.predict(scaled)[0]

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul,
        title={'text': "RUL (ciclos)"},
        gauge={'axis': {'range': [0, 200]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "red"},
                   {'range': [50, 100], 'color': "orange"},
                   {'range': [100, 200], 'color': "green"}
               ]}
    ))

    alert = "Manutenção Urgente!" if rul < 50 else "OK" if rul < 100 else "Excelente"
    color = "red" if rul < 50 else "orange" if rul < 100 else "green"

    info = html.Ul([html.Li(f"{k}: {v:.3f}") for k, v in sample.iloc[0].items()])

    return info, fig, html.Span(alert, style={'color': color})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))   
    app.run_server(host='0.0.0.0', port=port, debug=False)