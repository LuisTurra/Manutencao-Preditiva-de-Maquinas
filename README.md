# Manutenção Preditiva: NASA Turbofan Engine (CMAPSS)

**Preveja falhas em motores de avião com 68% de precisão!**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Dash](https://img.shields.io/badge/Dash-008DE4?style=flat&logo=dash&logoColor=white)](https://dash.plotly.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF8C00?style=flat&logo=xgboost&logoColor=white)](https://xgboost.ai)

## Demo Ao Vivo
[https://turbofan-predictive.onrender.com](https://turbofan-predictive.onrender.com) *(deploy grátis no Render)*

## Resultados
- **RMSE**: `18.2` ciclos
- **Precisão**: Reduz falhas em **68%**
- **Deploy**: Dashboard em tempo real com alertas

## Como Rodar
```bash
git clone https://github.com/luisturra/predictive-maintenance-turbofan.git
cd predictive-maintenance-turbofan
pip install -r requirements.txt
# Coloque os .txt na pasta data/
python app.py
