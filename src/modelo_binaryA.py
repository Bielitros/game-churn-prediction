# ==========================================
# IMPORTA BIBLIOTECAS
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# CONFIGURAÇÃO MLFLOW
# ==========================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Steam Churn Project")

# ==========================================
# CARREGA O DATASET
# ==========================================

df = pd.read_csv("data/raw/steam_games_dataset.csv")

# ==========================================
# REMOVE POSSÍVEIS VALORES NULOS
# ==========================================

df = df.dropna()

# ==========================================
# FEATURES UTILIZADAS
# ==========================================

"""
SEM DATA LEAKAGE
"""

X = df[
    [
        "playtime_forever",
        "game_count",
        "recent_active_games",
        "avg_playtime",
        "top_game_ratio"
    ]
]

# ==========================================
# TARGET
# ==========================================

"""
PROBLEMA BINÁRIO

0 = ativo
1 = churn

A classe "possível churn" será
agrupada com a classe ativo.
"""

y = df["status"].replace({

    1: 0,
    2: 1
})

# ==========================================
# DIVISÃO TREINO / TESTE
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# PARÂMETROS DO MODELO
# ==========================================

n_estimators = 100
random_state = 42
class_weight = "balanced"

# ==========================================
# EXECUÇÃO MLFLOW
# ==========================================

with mlflow.start_run(
    run_name="binario_ativo"
):

    # ==========================================
    # LOG DOS PARÂMETROS
    # ==========================================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "class_weight",
        class_weight
    )

    mlflow.log_param(
        "data_leakage",
        False
    )

    mlflow.log_param(
        "target_strategy",
        "possible_churn_as_active"
    )

    # ==========================================
    # CRIA MODELO
    # ==========================================

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight
    )

    # ==========================================
    # TREINAMENTO
    # ==========================================

    print("\nTreinando modelo...\n")

    model.fit(X_train, y_train)

    print("Treinamento concluído!")

    # ==========================================
    # PREVISÕES
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # ACURÁCIA
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"\nAcurácia: {accuracy:.4f}")

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    # ==========================================
    # MATRIZ DE CONFUSÃO
    # ==========================================

    print("\n===== MATRIZ DE CONFUSÃO =====\n")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # ==========================================
    # RELATÓRIO
    # ==========================================

    print("\n===== RELATÓRIO =====\n")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "ativo",
            "churn"
        ],
        output_dict=True
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "ativo",
                "churn"
            ]
        )
    )

    # ==========================================
    # LOG DAS MÉTRICAS
    # ==========================================

    for classe in [
        "ativo",
        "churn"
    ]:

        mlflow.log_metric(
            f"precision_{classe}",
            report[classe]["precision"]
        )

        mlflow.log_metric(
            f"recall_{classe}",
            report[classe]["recall"]
        )

        mlflow.log_metric(
            f"f1_{classe}",
            report[classe]["f1-score"]
        )

    # ==========================================
    # IMPORTÂNCIA DAS FEATURES
    # ==========================================

    print("\n===== IMPORTÂNCIA DAS FEATURES =====\n")

    importance_df = pd.DataFrame({

        "feature": X.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    print(importance_df)

    # ==========================================
    # SALVA IMPORTÂNCIAS
    # ==========================================

    importance_path = (
        "data/raw/feature_importance_binaryA.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    mlflow.log_artifact(
        importance_path
    )

    # ==========================================
    # TESTE MANUAL
    # ==========================================

    example_player = pd.DataFrame([{

        "playtime_forever": 250000,
        "game_count": 20,
        "recent_active_games": 1,
        "avg_playtime": 12000,
        "top_game_ratio": 0.80
    }])

    prediction = model.predict(
        example_player
    )

    status_map = {
        0: "ativo",
        1: "churn"
    }

    print("\n===== TESTE MANUAL =====\n")

    print(
        "Status previsto:",
        status_map[prediction[0]]
    )

    # ==========================================
    # SALVA MODELO
    # ==========================================

    model_path = (
        "data/raw/binaryA_churn_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModelo salvo como binaryA_churn_model.pkl"
    )

    # ==========================================
    # REGISTRA MODELO NO MLFLOW
    # ==========================================

    mlflow.sklearn.log_model(
        model,
        artifact_path="binary_active_random_forest_model"
    )

    mlflow.log_artifact(
        model_path
    )

print("\nExecução registrada no MLflow!")