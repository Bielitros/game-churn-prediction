import requests
import pandas as pd
import time

# ==========================================
# CONFIGURAÇÕES
# ==========================================

API_KEY = "" # Sua chave da API Steam deve ser inserida aqui

steam_ids = [] # Os ids das contas STEAM que deseja analisar entram aqui

# ==========================================
# JOGOS ACEITOS COMO MAIN GAME
# ==========================================

relevant_games = [
    730,      # CS2
    570,      # Dota 2
    578080,   # PUBG
    2767030,   # Marvel Rivals
    381210,  # Dead by Daylight
    444090,   # Overwatch 2
    359550,   # Rainbow Six Siege
    440,      # Team Fortress 2
    2915500,  # Brawlhalla
    252950,   # Rocket League
    1808500 # Arc Raiders
]

# ==========================================
# LISTA FINAL DO DATASET
# ==========================================

dataset = []

# ==========================================
# COLETA DOS DADOS
# ==========================================

for steam_id in steam_ids:

    print(f"\nColetando perfil: {steam_id}")

    url = (
        f"https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
        f"?key={API_KEY}"
        f"&steamid={steam_id}"
        f"&include_appinfo=1"
        f"&include_played_free_games=1"
    )

    try:

        response = requests.get(url)

        # ==========================================
        # VERIFICAÇÕES DE SEGURANÇA
        # ==========================================

        if response.status_code != 200:

            print(f"Erro HTTP: {response.status_code}")
            continue

        if not response.text.strip():

            print("Resposta vazia.")
            continue

        try:

            data = response.json()

        except Exception:

            print("Erro ao converter JSON.")
            continue

        # ==========================================
        # VERIFICA SE O PERFIL POSSUI JOGOS
        # ==========================================

        if "games" not in data["response"]:

            print("Perfil sem jogos públicos.")
            continue

        games = data["response"]["games"]

        # ==========================================
        # REMOVE JOGOS COM 0 MINUTOS
        # ==========================================

        valid_games = [
            game for game in games
            if game.get("playtime_forever", 0) > 0
        ]

        if len(valid_games) == 0:

            print("Perfil sem jogos válidos.")
            continue

        # ==========================================
        # PEGA O JOGO MAIS JOGADO
        # ==========================================

        top_game = max(
            valid_games,
            key=lambda g: g.get("playtime_forever", 0)
        )

        appid = top_game.get("appid")

        # ==========================================
        # IGNORA PERFIS CUJO MAIN GAME
        # NÃO ESTÁ NA LISTA
        # ==========================================

        if appid not in relevant_games:

            print("Main game fora da lista.")
            continue

        # ==========================================
        # FEATURES GERAIS DO JOGADOR
        # ==========================================

        game_count = len(games)

        total_playtime_all_games = sum(
            game.get("playtime_forever", 0)
            for game in games
        )

        recent_active_games = sum(
            1
            for game in games
            if game.get("playtime_2weeks", 0) > 0
        )

        avg_playtime = (
            total_playtime_all_games / game_count
            if game_count > 0 else 0
        )

        max_playtime = top_game.get("playtime_forever", 0)

        top_game_ratio = (
            max_playtime / total_playtime_all_games
            if total_playtime_all_games > 0 else 0
        )

        # ==========================================
        # FEATURES DO MAIN GAME
        # ==========================================

        playtime_forever = top_game.get("playtime_forever", 0)

        # Remove jogos com menos de 10h
        if playtime_forever < 600:

            print("Main game com menos de 10h.")
            continue

        playtime_2weeks = top_game.get("playtime_2weeks", 0)

        engagement_ratio = (
            playtime_2weeks / playtime_forever
            if playtime_forever > 0 else 0
        )

        # ==========================================
        # CLASSIFICAÇÃO DE STATUS
        # ==========================================

        """
        status:
        0 = ativo
        1 = possível churn
        2 = churn
        """

        if playtime_2weeks == 0:

            status = 2
            status_label = "churn"

        elif (
            engagement_ratio < 0.01
            and playtime_2weeks < 300
        ):

            status = 1
            status_label = "possivel_churn"

        else:

            status = 0
            status_label = "ativo"

        # ==========================================
        # MONTA LINHA DO DATASET
        # ==========================================

        row = {

            # Identificação
            "steam_id": steam_id,

            # Main game
            "appid": appid,
            "game_name": top_game.get("name"),

            # Features do jogo
            "playtime_forever": playtime_forever,
            "playtime_2weeks": playtime_2weeks,
            "engagement_ratio": engagement_ratio,

            # Features do jogador
            "game_count": game_count,
            "recent_active_games": recent_active_games,
            "avg_playtime": avg_playtime,
            "top_game_ratio": top_game_ratio,

            # Target
            "status": status,
            "status_label": status_label
        }

        dataset.append(row)

        print(f"Main game identificado: {top_game.get('name')}")

        # ==========================================
        # DELAY PARA EVITAR RATE LIMIT
        # ==========================================

        time.sleep(1)

    except Exception as e:

        print(f"Erro no perfil {steam_id}: {e}")

# ==========================================
# TRANSFORMA EM DATAFRAME
# ==========================================

df = pd.DataFrame(dataset)

# ==========================================
# EVITA ERRO SE DATASET ESTIVER VAZIO
# ==========================================

if len(df) == 0:

    print("\nNenhum dado coletado.")

else:

    # ==========================================
    # FORMATAÇÃO DAS FEATURES
    # ==========================================

    df["engagement_ratio"] = df["engagement_ratio"].round(4)

    df["avg_playtime"] = df["avg_playtime"].round(2)

    df["top_game_ratio"] = df["top_game_ratio"].round(4)

    # ==========================================
    # ESTATÍSTICAS
    # ==========================================

    print("\n===== PRIMEIRAS LINHAS =====")
    print(df.head())

    print("\n===== TAMANHO DO DATASET =====")
    print(df.shape)

    print("\n===== DISTRIBUIÇÃO DOS STATUS =====")
    print(df["status_label"].value_counts())

    print("\n===== ESTATÍSTICAS GERAIS =====")
    print(df.describe())

    # ==========================================
    # SALVA DATASET
    # ==========================================

    output_path = "data/raw/steam_games_dataset.csv"

    df.to_csv(output_path, index=False)

    print(f"\nDataset salvo em: {output_path}")