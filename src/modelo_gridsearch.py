# ==========================================
# IMPORTA BIBLIOTECAS
# ==========================================

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

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
status:
0 = ativo
1 = possível churn
2 = churn
"""

y = df["status"]

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
# EXECUÇÃO MLFLOW
# ==========================================

with mlflow.start_run(
    run_name="grid_search_random_forest"
):

    # ==========================================
    # LOG DOS PARÂMETROS
    # ==========================================

    mlflow.log_param(
        "model_type",
        "RandomForest_GridSearch"
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "data_leakage",
        False
    )

    # ==========================================
    # CRIA MODELO
    # ==========================================

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {

        "n_estimators": [
            100,
            300
        ],

        "max_depth": [
            10,
            None
        ],

        "min_samples_split": [
            2,
            5
        ]
    }

    grid_search = GridSearchCV(

        estimator=rf,

        param_grid=param_grid,

        cv=5,

        scoring="f1_weighted",

        n_jobs=-1,

        verbose=2
    )

    # ==========================================
    # TREINAMENTO
    # ==========================================

    print("\nTreinando modelo...\n")

    grid_search.fit(
        X_train,
        y_train
    )

    model = grid_search.best_estimator_

    print("\n===== MELHORES PARÂMETROS =====\n")

    print(
        grid_search.best_params_
    )

    mlflow.log_param(
        "best_n_estimators",
        grid_search.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "best_max_depth",
        grid_search.best_params_["max_depth"]
    )

    mlflow.log_param(
        "best_min_samples_split",
        grid_search.best_params_["min_samples_split"]
    )

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
            "possivel_churn",
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
                "possivel_churn",
                "churn"
            ]
        )
    )

    # ==========================================
    # LOG DAS MÉTRICAS
    # ==========================================

    for classe in [
        "ativo",
        "possivel_churn",
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
        "data/raw/feature_importance.csv"
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
        1: "possivel_churn",
        2: "churn"
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
        "data/raw/grid_search.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "\nModelo salvo como grid_search.pkl"
    )

    # ==========================================
    # REGISTRA MODELO NO MLFLOW
    # ==========================================

    mlflow.sklearn.log_model(
        model,
        artifact_path="grid_search_random_forest_model"
    )

    mlflow.log_artifact(
        model_path
    )

print("\nExecução registrada no MLflow!")