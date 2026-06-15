# ==========================================
# IMPORTA BIBLIOTECAS
# ==========================================

from fastapi import FastAPI

import pandas as pd

import joblib

# ==========================================
# INICIA API
# ==========================================

app = FastAPI()

# ==========================================
# CARREGA MODELO
# ==========================================

model = joblib.load(
    "data/raw/binaryC_churn_model.pkl"
)

# ==========================================
# MAPA DE CLASSES
# ==========================================

status_map = {
    0: "ativo",
    1: "churn"
}

# ==========================================
# ROTA PRINCIPAL
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Steam Churn API Online"
    }

# ==========================================
# ROTA DE PREVISÃO
# ==========================================

@app.post("/predict")
def predict(data: dict):

    # transforma entrada em DataFrame
    df = pd.DataFrame([data])

    # faz previsão
    prediction = model.predict(df)

    # converte classe
    result = status_map[prediction[0]]

    return {
        "prediction": result
    }