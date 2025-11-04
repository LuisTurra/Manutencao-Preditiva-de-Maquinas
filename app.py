# app.py
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import shap
from faker import Faker
import joblib
import os

# Importar funções
from src.preprocess import load_data, add_rul, feature_engineering, scale_features
from src.model import train_model
from src.explain import get_shap_explanation

# === CARREGAR E TREINAR MODELO (uma vez) ===
DATA_DIR = "data"
train_path = f"{DATA_DIR}/train_FD001.txt"
test_path = f"{DATA_DIR}/test_FD001.txt"
rul_path = f"{DATA_DIR}/RUL_FD001.txt"

print("Carregando dados...")
train_df, test_df, rul_df = load_data(train_path, test_path, rul_path)
train_df = add_rul(train_df)
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Preparar RUL real do test
max_cycle_test = test_df.groupby('unit')['cycle'].max().reset_index()
max_cycle_test.columns = ['unit', 'last_cycle']
rul_df = rul_df.reset_index()
rul_df.columns = ['unit', 'true_rul']
test_df = test_df.merge(max_cycle_test, on='unit')
test_df = test_df[test_df['cycle'] == test_df['last_cycle']]  # Último ciclo
test_df = test_df.merge(rul_df, on='unit')

# Features
X_train, X_test, scaler, feature_cols = scale_features(train_df.drop(['RUL'], axis=1), test_df.drop(['true_rul'], axis=1))
y_train = train_df['RUL']
y_test = test_df['true_rul']

print("Treinando modelo...")
model = train_model(X_train, y_train)

# Salvar modelo e scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_cols, 'features.pkl')

# Avaliar
rmse, _ = model.evaluate_model(model, X_test, y_test)
print(f"RMSE Final: {rmse:.2f}")

# === DASH APP ===
app = dash.Dash(__name__)
fake = Faker()

app.layout = html.Div(style={'fontFamily': 'Arial', 'margin': '2%'}, children=[
    html.H1("Manutenção Preditiva: Turbofan Engine (NASA CMAPSS)", style={'textAlign': 'center'}),
    
    html.Div([
        html.H3("Simulação IoT em Tempo Real"),
        html.Button("Gerar Dados de Sensor (IoT)", id="iot-btn", n_clicks=0, style={'padding': '10px', 'fontSize': '16px'}),
        html.Div(id="sensor-values", style={'margin': '20px 0', 'fontSize': '14px'})
    ], style={'width': '48%', 'display': 'inline-block'}),

    html.Div([
        html.H3("Previsão de RUL"),
        dcc.Graph(id="rul-gauge"),
        html.Div(id="alert-box", style={'fontSize': '20px', 'fontWeight': 'bold', 'margin': '10px 0'})
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Hr(),
    html.H3("Explicabilidade (SHAP)"),
    dcc.Graph(id="shap-plot"),

    dcc.Store(id='current-data')
])

@app.callback(
    [Output('current-data', 'data'),
     Output('sensor-values', 'children'),
     Output('rul-gauge', 'figure'),
     Output('alert-box', 'children'),
     Output('shap-plot', 'figure')],
    [Input('iot-btn', 'n_clicks')],
    [State('current-data', 'data')]
)
def update_prediction(n_clicks, stored_data):
    # Simular dados IoT
    np.random.seed(n_clicks)
    sample = pd.DataFrame({
        col: [np.random.uniform(0.3, 0.9)] for col in feature_cols
    })
    
    # Escalar
    sample_scaled = scaler.transform(sample)
    pred_rul = model.predict(sample_scaled)[0]
    
    # Gráfico de RUL
    fig_rul = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_rul,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Remaining Useful Life (Ciclos)"},
        gauge = {
            'axis': {'range': [0, 200]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 100], 'color': "orange"},
                {'range': [100, 200], 'color': "green"}
            ]
        }
    ))
    
    alert = "Manutenção Urgente!" if pred_rul < 50 else "Operação Normal" if pred_rul < 100 else "Saúde Excelente"
    alert_color = "red" if pred_rul < 50 else "orange" if pred_rul < 100 else "green"
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(sample_scaled)
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols).abs().mean().sort_values(ascending=True).tail(5)
    
    fig_shap = go.Figure(go.Bar(
        x=shap_df.values,
        y=shap_df.index,
        orientation='h',
        marker_color='indianred'
    ))
    fig_shap.update_layout(title="Top 5 Fatores que Reduzem RUL", xaxis_title="Impacto SHAP", height=300)

    sensor_text = html.Ul([html.Li(f"{k}: {v:.3f}") for k, v in sample.iloc[0].head(5).items()])

    return sample.to_dict('records'), sensor_text, fig_rul, html.Span(alert, style={'color': alert_color}), fig_shap

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)