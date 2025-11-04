# import shap
# import pandas as pd

# def get_shap_explanation(model, X_sample, feature_cols):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_sample)
    
#     # Retorna top 5 features mais importantes
#     shap_df = pd.DataFrame(shap_values, columns=feature_cols)
#     importance = shap_df.abs().mean().sort_values(ascending=False).head(5)
#     return importance