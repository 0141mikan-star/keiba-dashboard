import streamlit as st
import pandas as pd
import numpy as np
import re

def get_default_win5_races(track_options: list = None) -> list[dict]:
    """WIN5用のリアルなサンプル5レースデータ"""
    t_opt = track_options or [
        "東京 (左回り / 直線525m・日本屈指の末脚勝負)",
        "中山 (右回り / 直線310m・急坂小回り前残り)",
        "中京 (左回り / 直線412m・急坂タフ)"
    ]
    nakayama_opt = next((t for t in t_opt if "中山" in t), t_opt[0])
    chukyo_opt = next((t for t in t_opt if "中京" in t), t_opt[0])
    tokyo_opt = next((t for t in t_opt if "東京" in t), t_opt[0])

    return [
        {
            "id": 1,
            "name": "WIN1: 中山10R 外房S (ダ1200m)",
            "track": nakayama_opt,
            "distance": 1200,
            "course_type": "ダート",
            "pace": "ハイペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "url": "",
            "past_records": {},
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "エスカル", "単勝人気": 1, "想定単勝オッズ": 2.1, "騎手名": "戸崎圭太", "脚質": "逃げ", "斤量": 58.0},
                {"馬番": 2, "馬名": "コパノパサディナ", "単勝人気": 2, "想定単勝オッズ": 4.5, "騎手名": "横山武史", "脚質": "先行", "斤量": 57.0},
                {"馬番": 3, "馬名": "スマートフォルス", "単勝人気": 3, "想定単勝オッズ": 6.8, "騎手名": "ルメール", "脚質": "先行", "斤量": 58.0},
                {"馬番": 4, "馬名": "サザンエルフ", "単勝人気": 4, "想定単勝オッズ": 9.2, "騎手名": "津村明秀", "脚質": "差し", "斤量": 55.0},
                {"馬番": 5, "馬名": "アイアムユウシュン", "単勝人気": 5, "想定単勝オッズ": 14.5, "騎手名": "佐々木大輔", "脚質": "差し", "斤量": 56.0},
                {"馬番": 6, "馬名": "ショウナンカブト", "単勝人気": 6, "想定単勝オッズ": 22.0, "騎手名": "石川裕紀人", "脚質": "追込", "斤量": 58.0},
            ])
        },
        {
            "id": 2,
            "name": "WIN2: 中京10R 甲南S (ダ1800m)",
            "track": chukyo_opt,
            "distance": 1800,
            "course_type": "ダート",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "url": "",
            "past_records": {},
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "バハルダール", "単勝人気": 1, "想定単勝オッズ": 2.8, "騎手名": "川田将雅", "脚質": "先行", "斤量": 58.0},
                {"馬番": 2, "馬名": "クールミラボー", "単勝人気": 2, "想定単勝オッズ": 3.6, "騎手名": "松山弘平", "脚質": "差し", "斤量": 58.0},
                {"馬番": 3, "馬名": "サンライズアリオン", "単勝人気": 3, "想定単勝オッズ": 5.4, "騎手名": "坂井瑠星", "脚質": "先行", "斤量": 57.0},
                {"馬番": 4, "馬名": "カズペトシーン", "単勝人気": 4, "想定単勝オッズ": 8.9, "騎手名": "西村淳也", "脚質": "差し", "斤量": 58.0},
                {"馬番": 5, "馬名": "オメガタキシード", "単勝人気": 5, "想定単勝オッズ": 12.3, "騎手名": "岩田望来", "脚質": "追込", "斤量": 58.0},
            ])
        },
        {
            "id": 3,
            "name": "WIN3: 中山11R オールカマー (芝2200m)",
            "track": nakayama_opt,
            "distance": 2200,
            "course_type": "芝",
            "pace": "スローペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "url": "",
            "past_records": {},
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "レーベンスティール", "単勝人気": 1, "想定単勝オッズ": 1.9, "騎手名": "ルメール", "脚質": "先行", "斤量": 57.0},
                {"馬番": 2, "馬名": "サヴォーナ", "単勝人気": 2, "想定単勝オッズ": 4.8, "騎手名": "池添謙一", "脚質": "先行", "斤量": 57.0},
                {"馬番": 3, "馬名": "ステラヴェローチェ", "単勝人気": 3, "想定単勝オッズ": 6.2, "騎手名": "横山典弘", "脚質": "差し", "斤量": 58.0},
                {"馬番": 4, "馬名": "リカンカブール", "単勝人気": 4, "想定単勝オッズ": 11.5, "騎手名": "津村明秀", "脚質": "先行", "斤量": 57.0},
                {"馬番": 5, "馬名": "キラーアビリティ", "単勝人気": 5, "想定単勝オッズ": 15.0, "騎手名": "北村友一", "脚質": "差し", "斤量": 58.0},
            ])
        },
        {
            "id": 4,
            "name": "WIN4: 中京11R 神戸新聞杯 (芝2200m)",
            "track": chukyo_opt,
            "distance": 2200,
            "course_type": "芝",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "url": "",
            "past_records": {},
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "メイショウタバル", "単勝人気": 2, "想定単勝オッズ": 4.1, "騎手名": "浜中俊", "脚質": "逃げ", "斤量": 56.0},
                {"馬番": 2, "馬名": "ジューンテイク", "単勝人気": 3, "想定単勝オッズ": 5.5, "騎手名": "藤岡佑介", "脚質": "先行", "斤量": 56.0},
                {"馬番": 3, "馬名": "メリオーレム", "単勝人気": 1, "想定単勝オッズ": 2.9, "騎手名": "川田将雅", "脚質": "差し", "斤量": 56.0},
                {"馬番": 4, "馬名": "ショウナンラプンタ", "単勝人気": 4, "想定単勝オッズ": 7.8, "騎手名": "鮫島克駿", "脚質": "差し", "斤量": 56.0},
                {"馬番": 5, "馬名": "オールセインツ", "単勝人気": 5, "想定単勝オッズ": 12.0, "騎手名": "坂井瑠星", "脚質": "追込", "斤量": 56.0},
            ])
        },
        {
            "id": 5,
            "name": "WIN5: 東京11R 天皇賞(秋) (芝2000m)",
            "track": tokyo_opt,
            "distance": 2000,
            "course_type": "芝",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "url": "",
            "past_records": {},
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "ドウデュース", "単勝人気": 2, "想定単勝オッズ": 3.8, "騎手名": "武豊", "脚質": "差し", "斤量": 58.0},
                {"馬番": 2, "馬名": "リバティアイランド", "単勝人気": 1, "想定単勝オッズ": 2.3, "騎手名": "川田将雅", "脚質": "先行", "斤量": 56.0},
                {"馬番": 3, "馬名": "ベラジオオペラ", "単勝人気": 3, "想定単勝オッズ": 6.5, "騎手名": "横山和生", "脚質": "先行", "斤量": 58.0},
                {"馬番": 4, "馬名": "ジャスティンパレス", "単勝人気": 4, "想定単勝オッズ": 9.4, "騎手名": "坂井瑠星", "脚質": "追込", "斤量": 58.0},
                {"馬番": 5, "馬名": "レーベンスティール", "単勝人気": 5, "想定単勝オッズ": 11.2, "騎手名": "ルメール", "脚質": "先行", "斤量": 58.0},
                {"馬番": 6, "馬名": "ソールオリエンス", "単勝人気": 6, "想定単勝オッズ": 18.0, "騎手名": "横山武史", "脚質": "追込", "斤量": 58.0},
            ])
        }
    ]

POP_ESTIMATED_ODDS = {
    1: 2.4, 2: 4.2, 3: 6.5, 4: 9.2, 5: 13.5,
    6: 18.0, 7: 25.0, 8: 38.0, 9: 55.0, 10: 75.0,
    11: 100.0, 12: 130.0, 13: 160.0, 14: 200.0,
    15: 250.0, 16: 300.0, 17: 350.0, 18: 400.0
}

def get_horse_effective_odds(horse_row) -> float:
    """出走馬データから実効単勝オッズを取得（欠損時は単勝人気から推計）"""
    try:
        odds = horse_row.get("想定単勝オッズ")
        if odds is None or pd.isna(odds) or str(odds).strip() in ["", "-", "0", "0.0"]:
            odds = horse_row.get("単勝オッズ")
        if odds is not None and not pd.isna(odds):
            val = float(odds)
            if val > 1.0:
                return val
    except Exception:
        pass
    try:
        pop = int(horse_row.get("単勝人気", 5)) if not pd.isna(horse_row.get("単勝人気")) else 5
    except Exception:
        pop = 5
    return POP_ESTIMATED_ODDS.get(pop, 25.0)

def select_win5_horses_for_quota(p_df: pd.DataFrame, quota: int) -> pd.DataFrame:
    """
    WIN5専用の勝ち切り推奨馬選定アルゴリズム
    複勝圏狙いの馬を排除し、単勝勝率・逃げ先行力・勝負強さ・オッズ妙味から
    『真に1着を獲る馬』をquota頭厳選する
    """
    if p_df.empty or quota <= 0:
        return p_df.head(0)

    df_copy = p_df.copy()

    # WIN5勝利スコアの算出
    scores = []
    for _, row in df_copy.iterrows():
        base_win = float(row.get("勝率(%)", 5.0))
        style = str(row.get("脚質", "先行"))
        if style == "逃げ":
            style_mult = 1.25
        elif style == "先行":
            style_mult = 1.15
        elif style == "差し":
            style_mult = 1.00
        else:
            style_mult = 0.85

        j_rate = float(row.get("騎手勝率(%)", 9.0))
        j_mult = 1.0 + max(-0.1, min(0.3, (j_rate - 9.0) * 0.015))

        odds = float(row.get("想定単勝オッズ", 10.0))
        odds_bonus = 4.0 if odds <= 2.5 else (2.0 if odds <= 4.5 else 0.0)

        is_upset = bool(row.get("特注穴馬", False))
        ev_val = float(row.get("期待値(EV)", 1.0))
        upset_bonus = 3.5 if (is_upset or ev_val >= 1.2) else 0.0

        v_score = (base_win * style_mult * j_mult) + odds_bonus + upset_bonus
        scores.append(v_score)

    df_copy["_win5_v_score"] = scores
    df_sorted = df_copy.sort_values(by="_win5_v_score", ascending=False).reset_index(drop=True)

    if quota == 1:
        # 1頭枠: 勝ち切り信頼度No.1の絶対本命
        return df_sorted.head(1).drop(columns=["_win5_v_score"], errors="ignore")

    elif quota == 2:
        # 2頭枠: 勝ち切り1位 + 単勝勝率上位または穴馬
        picked = [df_sorted.iloc[0], df_sorted.iloc[1]]
        return pd.DataFrame(picked).drop(columns=["_win5_v_score"], errors="ignore")

    else:
        # 3頭以上枠: 本命・対抗に加え、特注穴馬☆や先行逃げ馬を必ずブレンド
        picked = [df_sorted.iloc[0], df_sorted.iloc[1]]
        picked_nums = {df_sorted.iloc[0]["馬番"], df_sorted.iloc[1]["馬番"]}

        # 3頭目以降の候補: 特注穴馬や逃げ馬、高EV馬を優先
        remaining = df_sorted.iloc[2:]
        upset_candidates = remaining[remaining.apply(
            lambda r: bool(r.get("特注穴馬", False)) or str(r.get("脚質")) == "逃げ" or float(r.get("期待値(EV)", 0)) >= 1.2,
            axis=1
        )]

        for _, u_row in upset_candidates.iterrows():
            if len(picked) >= quota:
                break
            if u_row["馬番"] not in picked_nums:
                picked.append(u_row)
                picked_nums.add(u_row["馬番"])

        # 足りない分はスコア順で補充
        for _, rem_row in remaining.iterrows():
            if len(picked) >= quota:
                break
            if rem_row["馬番"] not in picked_nums:
                picked.append(rem_row)
                picked_nums.add(rem_row["馬番"])

        return pd.DataFrame(picked).drop(columns=["_win5_v_score"], errors="ignore")

def render_win5_simulator(
    calculate_predictions_func,
    learned_weights: dict,
    fetch_netkeiba_func=None,
    select_recommended_func=None,
    generate_comment_func=None,
    diagnose_volatility_func=None,
    track_options: list = None,
    save_ai_model_func=None,
    generate_race_sig_func=None,
    default_weights: dict = None,
    **kwargs
):
    """
    WIN5攻略シミュレーターのメインUI
    通常レースと同等の高精度AI解析・netkeiba URL自動取得・詳細診断カード・リアルタイム生還追跡を完全搭載
    """
    st.title("🏆 WIN5 攻略シミュレーター")
    st.markdown(
        """
        **低予算（2,400円〜4,800円前後）でWIN5の高額配当を射止めるAIメリハリ戦略システム**です。  
        通常レースと同じく**「netkeiba URLを貼るだけで5レース一括自動入力」**に対応！  
        **「鉄板レースを1頭で突破し、浮いた予算を混戦レースに回す」**ことで、わずか**24点〜48点**で最大のカバー率と高回収率を実現します！
        """
    )

    # トラック選択肢の初期化
    if not track_options:
        track_options = [
            "東京 (左回り / 直線525m・日本屈指の末脚勝負)",
            "中山 (右回り / 直線310m・急坂小回り前残り)",
            "阪神 (右回り / 直線474m・急坂パワー勝負)",
            "京都 (右回り / 直線404m・平坦高速持続)",
            "中京 (左回り / 直線412m・急坂タフ)",
            "新潟 (左回り / 直線659m・日本最長直線)",
            "小倉・福島・函館・札幌 (小回り平坦・先行有利)",
            "盛岡 (左回り / 起伏4.4mタフ・地方唯一芝併設)",
            "大井 (右回り / 直線386m・南関王道・外差し届く)",
            "金沢 (右回り / 直線236m・超小回り前残り鉄則)",
            "笠松 (右回り / 直線201m・超小回り前残り・内枠先行鉄則)",
            "園田 (右回り / 直線213m・小回りスパイラル・先行有利＆捲り警戒)",
            "帯広（ば） (直線200m / 世界唯一ばんえい・障害登坂)",
            "標準・地方他 (平均的・先行やや有利)"
        ]

    condition_options = [
        "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
        "稍重 (標準〜やや湿り / 総合力)",
        "重・不良 (雨・泥濘 / パワー＆スタミナ前残り波乱・ばんえいはソリ滑走軽馬場)"
    ]

    # セッション状態にWIN5レースデータを保持
    if "win5_races_data" not in st.session_state:
        st.session_state["win5_races_data"] = get_default_win5_races(track_options)

    # 予算プランの選択
    st.markdown("### 💰 予算プランの選択")
    plan_col1, plan_col2 = st.columns([2, 1])
    with plan_col1:
        plan_choice = st.radio(
            "目的に合った買い目プランを選択してください:",
            [
                "🛡️ 【全レース2頭以上・事故防止】3,200円プラン (32点) : 2頭 × 2頭 × 2頭 × 2頭 × 2頭 (1頭固定の突然死を完全回避・超安定！)",
                "🎯 【本命・メリハリ】3,600円プラン (36点) : 1頭 × 2頭 × 2頭 × 3頭 × 3頭 (王道バランス)",
                "🚀 【手広く安心】4,800円プラン (48点) : 1頭 × 2頭 × 2頭 × 3頭 × 4頭",
                "🏆 【高確率・本気的中】7,200円プラン (72点) : 2頭 × 2頭 × 2頭 × 3頭 × 3頭 (全レース2頭以上＋混戦3頭カバー)",
                "⚡ 【超エコノミー】2,400円プラン (24点) : 1頭 × 2頭 × 2頭 × 2頭 × 3頭",
                "🛠️ 【カスタム配分】各レースの頭数を自由指定"
            ],
            index=0
        )
    with plan_col2:
        if "3,200円" in plan_choice:
            quota_pattern = [2, 2, 2, 2, 2]
            base_cost = 3200
            base_points = 32
            st.info("💡 **おすすめNo.1（最も安定！）**\n全5レースをすべて【2頭以上】でカバー！ 1頭固定による『ハナ差2着での即死』を完全回避する最強の安全設計！")
        elif "3,600円" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 3]
            base_cost = 3600
            base_points = 36
            st.info("💡 **王道メリハリ**\n最も鉄板の1レースを本命◎1点で突破し、浮いた予算を残りのレースに集中投下する高回収率設計！")
        elif "4,800円" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 4]
            base_cost = 4800
            base_points = 48
            st.info("💡 **カバー重視**\n大荒れが予想されるメイン重賞や難関ハンデ戦を4頭まで手広く抑える安心設計！")
        elif "7,200円" in plan_choice:
            quota_pattern = [2, 2, 2, 3, 3]
            base_cost = 7200
            base_points = 72
            st.info("💡 **高的中・本気プラン**\n全レース2頭以上で不慮の取りこぼしを防ぎつつ、混戦レースを3頭で厚めに包囲する本格派！")
        elif "2,400円" in plan_choice:
            quota_pattern = [1, 2, 2, 2, 3]
            base_cost = 2400
            base_points = 24
            st.info("💡 **超節約エコノミー**\nわずか2,400円（24点）で毎週気軽にWIN5に挑戦できる高回収率特化プラン！")
        else:
            quota_pattern = [2, 2, 2, 2, 2]
            base_cost = 3200
            base_points = 32
            st.info("💡 **カスタム配分**\nAI診断後、各レースの選択頭数や対象馬を自由に微調整できます。")

    # データ更新バージョンカウンタ（外部取得・サンプルロード時にキーを一新して確実にUI反映させる）
    if "win5_data_version" not in st.session_state:
        st.session_state["win5_data_version"] = 0

    # ==========================================
    # 🌐 出馬表＆レースデータ入力エリア
    # ==========================================
    st.markdown("---")
    with st.expander("🌐 **【netkeiba URLから全自動取得】WIN1〜WIN5の出走馬・オッズ・過去走を一括入力**", expanded=True):
        st.markdown(
            """
            ##### 🚀 netkeibaの出走表URLを貼るだけで、5レースすべての出走馬・オッズ・本物過去走を全自動取得！
            中央競馬（`race.netkeiba.com`）の出走表URL（`shutuba.html?race_id=...`）を入力すると、枠番・馬番・馬名・人気・オッズ・騎手フルネーム・脚質・斤量・過去走カルテが1秒で一括反映されます。
            """
        )
        url_cols = st.columns(5)
        batch_urls = []
        for i in range(5):
            curr_r = st.session_state["win5_races_data"][i]
            with url_cols[i]:
                # セッションステートにある入力値を優先保持
                default_u = st.session_state.get(f"win5_batch_url_{i}", curr_r.get("url", ""))
                u_val = st.text_input(
                    f"WIN{i+1} URL",
                    value=default_u,
                    placeholder="...race_id=...",
                    key=f"win5_batch_url_{i}"
                )
                batch_urls.append(u_val)

        btn_c1, btn_c2 = st.columns([1.5, 1.0])
        with btn_c1:
            btn_fetch_all = st.button("📥 5レース一括取得実行 (URLから自動解析)", type="primary", use_container_width=True)
        with btn_c2:
            btn_sample_all = st.button("🎲 サンプル5レースを一括ロード (即座にお試し)", use_container_width=True)

        if btn_sample_all:
            st.session_state["win5_races_data"] = get_default_win5_races(track_options)
            st.session_state["win5_has_run"] = False
            st.session_state["win5_custom_selections"] = {}
            st.session_state["win5_data_version"] += 1
            st.success("🎉 サンプル5レース（中山/中京/東京）を正常にセットしました！")
            st.rerun()

        if btn_fetch_all:
            if not fetch_netkeiba_func:
                st.error("⚠️ URL解析機能が接続されていません。")
            else:
                # 入力されたURLを取得
                urls_to_fetch = []
                for i in range(5):
                    raw_u = st.session_state.get(f"win5_batch_url_{i}", "") or batch_urls[i]
                    urls_to_fetch.append(raw_u.strip())

                valid_indices = [(i, u) for i, u in enumerate(urls_to_fetch) if u]
                if not valid_indices:
                    st.warning("⚠️ 少なくとも1つ以上のnetkeibaレースURLを入力してください。")
                else:
                    success_count = 0
                    fetched_details = []
                    error_details = []

                    with st.spinner("🌐 netkeibaから5レースの出走馬・オッズ・本物過去走データを取得中..."):
                        for idx, u in valid_indices:
                            res_data, err = fetch_netkeiba_func(u)
                            if err:
                                error_details.append(f"WIN{idx+1}: {err}")
                            else:
                                success_count += 1
                                detected_t = res_data["track_name"]
                                matched_track = next((t for t in track_options if detected_t in t), track_options[0])

                                horses_df = pd.DataFrame(res_data["horses"])
                                if "想定単勝オッズ" not in horses_df.columns or horses_df["想定単勝オッズ"].isna().all():
                                    if "単勝オッズ" in horses_df.columns:
                                        horses_df["想定単勝オッズ"] = horses_df["単勝オッズ"]
                                    else:
                                        horses_df["想定単勝オッズ"] = 10.0
                                elif "単勝オッズ" in horses_df.columns:
                                    horses_df["想定単勝オッズ"] = horses_df["想定単勝オッズ"].fillna(horses_df["単勝オッズ"])
                                if "斤量" not in horses_df.columns:
                                    horses_df["斤量"] = 56.0

                                r_name_formatted = f"WIN{idx+1}: {res_data['race_name']} ({res_data['course_type']}{res_data['distance']}m)"
                                act_ranks = res_data.get("actual_ranks")
                                act_payouts = res_data.get("official_payouts")
                                win_no = 0
                                if act_ranks and isinstance(act_ranks, dict):
                                    for _, h_row in horses_df.iterrows():
                                        if act_ranks.get(str(h_row["馬名"]).strip()) == 1:
                                            win_no = int(h_row["馬番"])
                                            st.session_state[f"actual_win_{idx}"] = win_no
                                            break

                                st.session_state["win5_races_data"][idx] = {
                                    "id": idx + 1,
                                    "name": r_name_formatted,
                                    "track": matched_track,
                                    "distance": res_data["distance"],
                                    "course_type": res_data["course_type"],
                                    "pace": st.session_state["win5_races_data"][idx].get("pace", "ミドルペース"),
                                    "condition": st.session_state["win5_races_data"][idx].get("condition", condition_options[0]),
                                    "url": u,
                                    "past_records": res_data.get("past_records", {}),
                                    "horses": horses_df,
                                    "actual_ranks": act_ranks,
                                    "official_payouts": act_payouts,
                                    "actual_winner": win_no
                                }
                                extra_tag = f" 🎯 確定1着: {win_no}番" if win_no > 0 else ""
                                fetched_details.append(f"**WIN{idx+1}**: {res_data['race_name']} ({len(horses_df)}頭{extra_tag})")

                    if error_details:
                        for ed in error_details:
                            st.error(f"⚠️ {ed}")

                    if success_count > 0:
                        st.session_state["win5_has_run"] = False
                        st.session_state["win5_custom_selections"] = {}
                        st.session_state["win5_data_version"] += 1
                        st.success(f"🎉 **{success_count}レース分の出走馬データを取得し、下の出馬表に反映しました！**\n\n" + " ｜ ".join(fetched_details))
                        st.rerun()

    # ==========================================
    # 📋 各レース（WIN1〜WIN5）の条件設定 ＆ 出走馬エディタ
    # ==========================================
    st.subheader("📋 WIN5 対象5レースの条件設定 ＆ 出馬表")
    st.caption("各レースの競馬場、距離、馬場状態、ペース、出走馬（馬番・人気・オッズ・騎手・脚質）を編集できます。個別のURL取得も可能です。")

    current_ver = st.session_state.get("win5_data_version", 0)
    race_tab_titles = []
    for i in range(5):
        r_d = st.session_state["win5_races_data"][i]
        h_cnt = len(r_d.get("horses", []))
        race_tab_titles.append(f"第{i+1}戦: WIN{i+1} ({h_cnt}頭)")

    race_tabs = st.tabs(race_tab_titles)
    edited_races = []

    for idx, r_tab in enumerate(race_tabs):
        r_info = st.session_state["win5_races_data"][idx]
        with r_tab:
            # 個別URL取得エリア
            c_u1, c_u2 = st.columns([3, 1])
            with c_u1:
                single_url = st.text_input(
                    f"WIN{idx+1} netkeiba URL (個別取得・更新用)",
                    value=r_info.get("url", ""),
                    key=f"win5_single_url_{idx}_{current_ver}",
                    placeholder="https://race.netkeiba.com/race/shutuba.html?race_id=..."
                )
            with c_u2:
                btn_single_fetch = st.button(f"📥 WIN{idx+1}のみ取得", key=f"btn_single_{idx}_{current_ver}", use_container_width=True)

            if btn_single_fetch and single_url.strip() and fetch_netkeiba_func:
                with st.spinner(f"🌐 WIN{idx+1}のレースデータを取得中..."):
                    res_data, err = fetch_netkeiba_func(single_url.strip())
                    if err:
                        st.error(f"⚠️ {err}")
                    else:
                        detected_t = res_data["track_name"]
                        matched_track = next((t for t in track_options if detected_t in t), track_options[0])
                        horses_df = pd.DataFrame(res_data["horses"])
                        if "想定単勝オッズ" not in horses_df.columns or horses_df["想定単勝オッズ"].isna().all():
                            if "単勝オッズ" in horses_df.columns:
                                horses_df["想定単勝オッズ"] = horses_df["単勝オッズ"]
                            else:
                                horses_df["想定単勝オッズ"] = 10.0
                        elif "単勝オッズ" in horses_df.columns:
                            horses_df["想定単勝オッズ"] = horses_df["想定単勝オッズ"].fillna(horses_df["単勝オッズ"])
                        if "斤量" not in horses_df.columns:
                            horses_df["斤量"] = 56.0

                        act_ranks = res_data.get("actual_ranks")
                        act_payouts = res_data.get("official_payouts")
                        win_no = 0
                        if act_ranks and isinstance(act_ranks, dict):
                            for _, h_row in horses_df.iterrows():
                                if act_ranks.get(str(h_row["馬名"]).strip()) == 1:
                                    win_no = int(h_row["馬番"])
                                    st.session_state[f"actual_win_{idx}"] = win_no
                                    break

                        st.session_state["win5_races_data"][idx] = {
                            "id": idx + 1,
                            "name": f"WIN{idx+1}: {res_data['race_name']} ({res_data['course_type']}{res_data['distance']}m)",
                            "track": matched_track,
                            "distance": res_data["distance"],
                            "course_type": res_data["course_type"],
                            "pace": r_info.get("pace", "ミドルペース"),
                            "condition": r_info.get("condition", condition_options[0]),
                            "url": single_url.strip(),
                            "past_records": res_data.get("past_records", {}),
                            "horses": horses_df,
                            "actual_ranks": act_ranks,
                            "official_payouts": act_payouts,
                            "actual_winner": win_no
                        }
                        # バッチURL欄にも反映
                        st.session_state[f"win5_batch_url_{idx}"] = single_url.strip()
                        st.session_state["win5_has_run"] = False
                        st.session_state["win5_data_version"] += 1
                        st.success(f"🎉 WIN{idx+1} ({res_data['race_name']} / {len(horses_df)}頭) を取得し、表に反映しました！")
                        st.rerun()

            # レース条件設定
            c_r1, c_r2, c_r3 = st.columns([2, 1, 1])
            with c_r1:
                r_name = st.text_input(f"レース名 (WIN{idx+1})", value=r_info["name"], key=f"win5_name_{idx}_{current_ver}")
            with c_r2:
                r_dist = st.number_input(f"距離 (m)", value=int(r_info["distance"]), step=100, key=f"win5_dist_{idx}_{current_ver}")
            with c_r3:
                c_idx = 0 if r_info["course_type"] == "芝" else (1 if r_info["course_type"] == "ダート" else 2)
                r_ctype = st.selectbox(f"コース種別", ["芝", "ダート", "ばんえい"], index=c_idx, key=f"win5_ctype_{idx}_{current_ver}")

            c_r4, c_r5, c_r6 = st.columns([1.5, 1, 1])
            with c_r4:
                curr_t = r_info["track"]
                t_idx = 0
                for opt_i, opt_val in enumerate(track_options):
                    if curr_t in opt_val or any(k in opt_val for k in curr_t.split()[:1]):
                        t_idx = opt_i
                        break
                r_track = st.selectbox("🏟️ 競馬場", track_options, index=t_idx, key=f"win5_track_{idx}_{current_ver}")
            with c_r5:
                curr_c = r_info.get("condition", condition_options[0])
                c_cond_idx = 0
                for ci, cv in enumerate(condition_options):
                    if curr_c.split()[0] in cv:
                        c_cond_idx = ci
                        break
                r_cond = st.selectbox("🌦️ 馬場状態", condition_options, index=c_cond_idx, key=f"win5_cond_{idx}_{current_ver}")
            with c_r6:
                pace_opts = ["スローペース", "ミドルペース", "ハイペース"]
                curr_p = r_info.get("pace", "ミドルペース")
                p_idx = 1 if curr_p == "ミドルペース" else (0 if curr_p == "スローペース" else 2)
                r_pace = st.selectbox("⏱️ 想定ペース", pace_opts, index=p_idx, key=f"win5_pace_{idx}_{current_ver}")

            st.caption("出走馬データ (馬番・馬名・人気・オッズ・騎手名・脚質・斤量):")
            
            # DataFrameのカラム整備
            h_df_current = r_info["horses"].copy()
            for col, default_val in [
                ("馬番", 1), ("馬名", "馬"), ("単勝人気", 1),
                ("想定単勝オッズ", 5.0), ("騎手名", "騎手"), ("脚質", "先行"), ("斤量", 56.0)
            ]:
                if col not in h_df_current.columns:
                    h_df_current[col] = default_val

            # 出走馬データエディタ（キーにcurrent_verを組み込むことで、外部取得時に強制リフレッシュ）
            h_df_edited = st.data_editor(
                h_df_current,
                column_config={
                    "馬番": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, required=True),
                    "馬名": st.column_config.TextColumn("馬名", required=True),
                    "単勝人気": st.column_config.NumberColumn("人気", min_value=1, max_value=18, required=True),
                    "想定単勝オッズ": st.column_config.NumberColumn("オッズ", min_value=1.0, max_value=9999.0, step=0.1, format="%.1f倍"),
                    "騎手名": st.column_config.TextColumn("騎手名", required=True),
                    "脚質": st.column_config.SelectboxColumn("脚質", options=["逃げ", "先行", "差し", "追込"], required=True),
                    "斤量": st.column_config.NumberColumn("斤量", min_value=45.0, max_value=65.0, step=0.5, format="%.1fkg")
                },
                num_rows="dynamic",
                use_container_width=True,
                key=f"win5_editor_{idx}_{current_ver}"
            )

            # 過去走カルテ表示（netkeibaから取得されたデータがある場合）
            p_recs = r_info.get("past_records", {})
            if p_recs:
                with st.expander(f"📋 WIN{idx+1} 過去走データカルテ ({len(p_recs)}頭分取得済)", expanded=False):
                    for h_name, p_val in p_recs.items():
                        st.markdown(f"**{h_name}**: {p_val.get('last_race', '')} ｜ {p_val.get('form_trend', '')} ｜ 適性: {p_val.get('course_suitability', '')}")

            clean_horses = h_df_edited.dropna(subset=["馬名"]).copy()
            if "想定単勝オッズ" in clean_horses.columns:
                clean_horses["単勝オッズ"] = clean_horses["想定単勝オッズ"]

            edited_races.append({
                "id": idx + 1,
                "name": r_name,
                "track": r_track,
                "distance": r_dist,
                "course_type": r_ctype,
                "pace": r_pace,
                "condition": r_cond,
                "url": single_url,
                "past_records": p_recs,
                "horses": clean_horses,
                "actual_ranks": r_info.get("actual_ranks"),
                "actual_winner": r_info.get("actual_winner", 0),
                "official_payouts": r_info.get("official_payouts")
            })

    st.session_state["win5_races_data"] = edited_races

    # ==========================================
    # 🚀 AI診断 ＆ 最適フォーメーション自動生成
    # ==========================================
    st.markdown("---")
    btn_calc_win5 = st.button("🚀 WIN5 AI診断 ＆ 最適フォーメーションを自動生成", type="primary", use_container_width=True)

    if btn_calc_win5 or st.session_state.get("win5_has_run", False):
        st.session_state["win5_has_run"] = True

        with st.spinner("🤖 AIが5レースを同時解析し、鉄板度・波乱度・展開から最適買い目を算出中..."):
            race_diagnostics = []

            for r_data in edited_races:
                df = r_data["horses"]
                if df.empty or len(df) < 2:
                    continue

                w_a = learned_weights.get("ability", 1.5)
                w_p = learned_weights.get("pace", 1.2)
                w_j = learned_weights.get("jockey", 0.9)
                w_e = learned_weights.get("escape_dynamic", 1.2)
                w_s = learned_weights.get("synergy", 1.1)

                pred_res, meta = calculate_predictions_func(
                    df, w_a, w_p, w_j, w_e, w_s,
                    race_distance=r_data["distance"],
                    track_name=r_data["track"],
                    course_type=r_data["course_type"],
                    pace_prediction=r_data["pace"],
                    track_condition=r_data["condition"]
                )

                # レース波乱度診断
                vol_info = meta.get("volatility", {})
                if not vol_info and diagnose_volatility_func:
                    esc_cnt = len(df[df["脚質"] == "逃げ"])
                    vol_info = diagnose_volatility_func(pred_res, esc_cnt)

                top1 = pred_res.iloc[0]
                top2 = pred_res.iloc[1] if len(pred_res) >= 2 else None
                top3 = pred_res.iloc[2] if len(pred_res) >= 3 else None
                top4 = pred_res.iloc[3] if len(pred_res) >= 4 else None

                win1 = float(top1["勝率(%)"])
                win2 = float(top2["勝率(%)"]) if top2 is not None else 0.0
                win_gap = win1 - win2

                # 鉄板度スコア (勝率が高く、2位を引き離し、波乱度が低いほど鉄板)
                vol_level = vol_info.get("level", "")
                vol_penalty = 15.0 if "大波乱" in vol_level else (8.0 if "波乱含み" in vol_level else 0.0)
                solid_score = (win1 * 1.5) + (win_gap * 2.0) - vol_penalty

                race_diagnostics.append({
                    "id": r_data["id"],
                    "name": r_data["name"],
                    "track": r_data["track"],
                    "distance": r_data["distance"],
                    "course_type": r_data["course_type"],
                    "pace": r_data["pace"],
                    "condition": r_data["condition"],
                    "pred_df": pred_res,
                    "meta": meta,
                    "vol_info": vol_info,
                    "top1_horse": top1,
                    "top2_horse": top2,
                    "top3_horse": top3,
                    "top4_horse": top4,
                    "win1": win1,
                    "win2": win2,
                    "win_gap": win_gap,
                    "solid_score": solid_score,
                    "past_records": r_data.get("past_records", {}),
                    "url": r_data.get("url", ""),
                    "actual_ranks": r_data.get("actual_ranks"),
                    "actual_winner": r_data.get("actual_winner", 0),
                    "official_payouts": r_data.get("official_payouts")
                })

            if len(race_diagnostics) < 5:
                st.warning("⚠️ 5つのレースすべてに出走馬データを入力してください（各レース最低2頭以上）。")
                return

            # 鉄板度順にソートして、配分頭数を割り振る
            sorted_by_solidity = sorted(race_diagnostics, key=lambda x: x["solid_score"], reverse=True)

            for rank_idx, r_diag in enumerate(sorted_by_solidity):
                r_diag["assigned_quota"] = quota_pattern[rank_idx]
                r_diag["solidity_rank"] = rank_idx + 1

            # レース番号順（WIN1〜WIN5）に戻す
            race_diagnostics = sorted(race_diagnostics, key=lambda x: x["id"])

            # セッションに保持
            st.session_state["win5_diagnostics"] = race_diagnostics

            # ------------------------------------------
            # 🎯 AI最適フォーメーションサマリーカード
            # ------------------------------------------
            st.subheader("🎯 AI最適フォーメーション決定！")
            
            # ユーザーカスタム選択の保持
            if "win5_custom_selections" not in st.session_state:
                st.session_state["win5_custom_selections"] = {}

            # 各レースの初期選択馬 (WIN5専用勝ち切りアルゴリズム適用)
            active_selections = []
            for r_diag in race_diagnostics:
                r_id = r_diag["id"]
                p_df = r_diag["pred_df"]
                q = r_diag["assigned_quota"]

                # 以前にユーザーが選んだものがあればそれを優先、なければWIN5勝ち切りAI推奨q頭
                saved_sel = st.session_state["win5_custom_selections"].get(r_id)
                if saved_sel is not None and len(saved_sel) > 0:
                    chosen_horses = p_df[p_df["馬番"].isin(saved_sel)]
                    if chosen_horses.empty:
                        chosen_horses = select_win5_horses_for_quota(p_df, q)
                else:
                    chosen_horses = select_win5_horses_for_quota(p_df, q)
                active_selections.append(chosen_horses)

            # 点数・資金額の動的計算
            actual_points = int(np.prod([len(sel) for sel in active_selections]))
            actual_cost = actual_points * 100

            st.info(
                f"💡 **戦略プラン**: **{plan_choice.split(':')[0]}** ｜ "
                f"**合計購入点数: {actual_points}点** ｜ **必要資金: {actual_cost:,}円** (1点100円)"
            )

            # 5レース横断カード表示
            col_res_list = st.columns(5)
            for i, r_diag in enumerate(race_diagnostics):
                q = len(active_selections[i])
                p_df = r_diag["pred_df"]
                chosen = active_selections[i]
                s_rank = r_diag["solidity_rank"]

                with col_res_list[i]:
                    if q == 1:
                        badge = "🥇 【1点突破！】"
                        card_bg = "linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)"
                    elif q == 2:
                        badge = "🥈 【2頭カバー】"
                        card_bg = "linear-gradient(135deg, #134e5e 0%, #71b280 100%)"
                    else:
                        badge = "⚠️ 【混戦警戒】"
                        card_bg = "linear-gradient(135deg, #4b134f 0%, #c94b4b 100%)"

                    st.markdown(
                        f"""
                        <div style="background: {card_bg}; padding: 12px 10px; border-radius: 8px; color: white; margin-bottom: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.15);">
                            <b style="font-size: 1.1rem;">WIN{r_diag['id']}</b><br>
                            <span style="font-size: 0.85rem; font-weight: bold;">{badge}</span><br>
                            <span style="font-size: 0.75rem; opacity: 0.9;">鉄板度: 第{s_rank}位</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    r_clean_name = r_diag['name'].split(':')[1] if ':' in r_diag['name'] else r_diag['name']
                    st.caption(f"**{r_clean_name}**")
                    
                    nums_str = ", ".join([f"**{int(r['馬番'])}番**" for _, r in chosen.iterrows()])
                    st.markdown(f"🎯 **選択: {nums_str}** ({q}頭)")
                    for _, h in chosen.iterrows():
                        ai_mark = h.get('AI印', '◎' if _ == chosen.index[0] else '○')
                        st.write(f"- {ai_mark} {int(h['馬番'])}番 {h['馬名']} ({h['勝率(%)']}%)")

            # ------------------------------------------
            # 🛠️ 買い目の自由カスタマイズ（手動微調整）
            # ------------------------------------------
            with st.expander("🛠️ **買い目を自由にカスタマイズ（各レースの購入馬を手動追加・除外）**", expanded=False):
                st.caption("AIが選定した推奨馬に加えて、「この穴馬も抑えたい」「この馬は消したい」などの微調整がワンタップで可能です。変更すると合計点数と必要資金がリアルタイムに更新されます。")
                cust_cols = st.columns(5)
                for i, r_diag in enumerate(race_diagnostics):
                    r_id = r_diag["id"]
                    p_df = r_diag["pred_df"]
                    all_options = [int(r["馬番"]) for _, r in p_df.iterrows()]
                    current_selected = [int(r["馬番"]) for _, r in active_selections[i].iterrows()]

                    with cust_cols[i]:
                        new_sel = st.multiselect(
                            f"WIN{r_id} 選択馬番",
                            options=all_options,
                            default=current_selected,
                            format_func=lambda x, p_df=p_df: f"{x}番 {p_df[p_df['馬番']==x]['馬名'].values[0]} ({p_df[p_df['馬番']==x].get('AI印', pd.Series(['-'])).values[0]})",
                            key=f"win5_multisel_{r_id}"
                        )
                        if set(new_sel) != set(current_selected) and len(new_sel) > 0:
                            st.session_state["win5_custom_selections"][r_id] = new_sel
                            st.rerun()

            # ------------------------------------------
            # 📝 IPAT投票用コード & 総合的中期待率
            # ------------------------------------------
            st.markdown("---")
            col_sum1, col_sum2 = st.columns([2, 1])
            with col_sum1:
                st.subheader("📝 IPAT投票用 買い目まとめ")
                ipat_lines = []
                for i, r_diag in enumerate(race_diagnostics):
                    c_horses = active_selections[i]
                    nums = [str(int(r["馬番"])) for _, r in c_horses.iterrows()]
                    ipat_lines.append(f"WIN{r_diag['id']}: {', '.join(nums)} ({len(nums)}頭)")
                
                ipat_code = "\n".join(ipat_lines)
                st.code(ipat_code, language="text")
                st.caption("※ 上記の馬番を即PATまたはマークカード（フォーメーション）に転記して投票してください。")

            with col_sum2:
                st.subheader("📊 的中確率＆配当期待")
                total_cov = float(np.prod([
                    min(0.98, sum(active_selections[i]["勝率(%)"]) / 100.0) for i in range(5)
                ])) * 100.0
                st.metric("🎯 5レース総合カバー期待率", f"{total_cov:.2f}%")
                st.write(f"💵 **購入資金**: **{actual_cost:,}円**")
                st.write(f"🎫 **購入点数**: **{actual_points}点**")
                st.caption("※ 各レースで選択馬の勝率合計を掛け合わせた総合通過確率の推計値です。")

            # ------------------------------------------
            # 💰 WIN5 推定配当金レンジシミュレーション (購入前想定)
            # ------------------------------------------
            min_odds_list = []
            max_odds_list = []
            main_odds_list = []

            for i in range(5):
                sel_df = active_selections[i]
                odds_vals = [get_horse_effective_odds(r) for _, r in sel_df.iterrows()]
                if not odds_vals:
                    odds_vals = [5.0]
                min_odds_list.append(min(odds_vals))
                max_odds_list.append(max(odds_vals))
                main_odds_list.append(odds_vals[0])

            # JRA払戻率70%に基づく合成配当推計
            min_est_pay = int(min(600_000_000, max(100, float(np.prod(min_odds_list)) * 70)))
            max_est_pay = int(min(600_000_000, max(100, float(np.prod(max_odds_list)) * 70)))
            main_est_pay = int(min(600_000_000, max(100, float(np.prod(main_odds_list)) * 70)))

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #1c2833 0%, #2c3e50 100%); border: 1px solid #4a6572; border-radius: 10px; padding: 14px 18px; margin-top: 10px; margin-bottom: 12px; color: white;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <div>
                            <span style="font-size: 15px; font-weight: bold; color: #f1c40f;">💰 WIN5 推定払戻金シミュレーション (オッズ合成想定)</span>
                        </div>
                        <div style="font-size: 13px; color: #bdc3c7;">
                            ※ JRA払戻率70%換算概算値 (1点100円購入想定)
                        </div>
                    </div>
                    <div style="display: flex; gap: 24px; margin-top: 10px; flex-wrap: wrap; align-items: baseline;">
                        <div>
                            <span style="font-size: 12px; color: #bdc3c7;">🎯 AI本線 想定配当:</span><br>
                            <span style="font-size: 20px; font-weight: bold; color: #2ecc71;">￥{main_est_pay:,} 円</span>
                        </div>
                        <div>
                            <span style="font-size: 12px; color: #bdc3c7;">📉 最低配当想定 (堅着):</span><br>
                            <span style="font-size: 18px; font-weight: bold; color: #e0e0e0;">￥{min_est_pay:,} 円</span>
                        </div>
                        <div>
                            <span style="font-size: 12px; color: #bdc3c7;">📈 最高配当想定 (穴着):</span><br>
                            <span style="font-size: 18px; font-weight: bold; color: #f39c12;">￥{max_est_pay:,} 円</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ==========================================
            # 🌟 各レースの詳細AI診断・推奨馬カード (通常レースと同等！)
            # ==========================================
            st.markdown("---")
            st.subheader("🌟 各レースの詳細AI診断 ＆ 推奨馬カルテ (通常レース完全同期)")
            st.caption("通常レース予測と同じく、各レースのAI波乱度診断・展開シミュレーション・厳選推奨馬（◎○▲☆）・出走全頭ランキング・過去走カルテを詳細に閲覧できます。")

            detail_tabs = st.tabs([f"第{i}戦: WIN{i} 詳細" for i in range(1, 6)])

            for i, d_tab in enumerate(detail_tabs):
                r_diag = race_diagnostics[i]
                p_df = r_diag["pred_df"]
                meta = r_diag["meta"]
                vol_info = r_diag["vol_info"]

                with d_tab:
                    # レース条件バッジ
                    st.info(
                        f"📍 **設定レース条件**: 【{r_diag['name']}】 ｜ コース: **{r_diag['track']} ({r_diag['course_type']} {r_diag['distance']}m)** ｜ "
                        f"馬場: **{r_diag['condition'].split()[0]}** ｜ ペース: **{r_diag['pace']}**"
                    )

                    # 最強AIレース診断カード
                    if vol_info:
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); padding: 14px 20px; border-radius: 8px; color: white; margin-bottom: 12px; border-left: 6px solid {vol_info.get('badge_color', '#0d6efd')};">
                                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                                    <span style="font-size: 16px; font-weight: bold;">
                                        🎯 【最強AIレース診断】 {vol_info.get('level', '')} ｜ 波乱度: {vol_info.get('stars', '★★★☆☆')} ｜ 自信度: <span style="color: #ffeb3b;">{vol_info.get('confidence', '')}</span>
                                    </span>
                                </div>
                                <div style="margin-top: 6px; font-size: 14px; opacity: 0.95;">
                                    {vol_info.get('summary', '')}
                                </div>
                                <div style="margin-top: 6px; font-size: 13px; color: #a5d6a7; font-weight: bold;">
                                    {vol_info.get('bet_strategy', '')}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # AI展開力学シミュレーション
                    pace_msg = meta.get("pace_diagnosis", "")
                    if pace_msg:
                        st.markdown(
                            f"""
                            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 12px 18px; border-radius: 8px; color: white; margin-bottom: 12px;">
                                <b>🤖 AI展開力学シミュレータの診断結果:</b><br>{pace_msg}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # 激走穴馬アラート
                    upset_h = p_df[p_df.get("特注穴馬", False) == True]
                    if len(upset_h) > 0:
                        up_names = [f"**{r['馬番']}番 {r['馬名']}** (単勝{r['単勝人気']}位 → AI予測{r['予想着順']})" for _, r in upset_h.iterrows()]
                        st.warning(f"🔥 **【AI特注・激走穴馬を発掘！】** {' / '.join(up_names)}")

                    # AI推奨馬ピックアップ (◎ ○ ▲ ☆)
                    st.markdown("##### 🌟 AI厳選推奨馬ピックアップ")
                    rec_horses = []
                    if select_recommended_func:
                        rec_horses = select_recommended_func(p_df, max_count=4)
                    else:
                        rec_horses = [p_df.iloc[k] for k in range(min(4, len(p_df)))]

                    mark_colors = {
                        "◎": {"bg": "#dc3545", "role": "本命 (1点突破推奨)"},
                        "○": {"bg": "#0d6efd", "role": "対抗 (連対有力)"},
                        "▲": {"bg": "#198754", "role": "単穴 (一発逆転)"},
                        "☆": {"bg": "#fd7e14", "role": "特注穴馬 (高配当)"},
                        "△": {"bg": "#6f42c1", "role": "連下 (抑え)"},
                        "ー": {"bg": "#6c757d", "role": "注目"}
                    }

                    for rec_row in rec_horses:
                        if generate_comment_func:
                            rec_info = generate_comment_func(rec_row, meta, r_diag["distance"], r_diag["course_type"])
                        else:
                            rec_info = {
                                "mark": rec_row.get("AI印", "◎"),
                                "horse_num": int(rec_row["馬番"]),
                                "horse_name": rec_row["馬名"],
                                "jockey": rec_row["騎手名"],
                                "pop": int(rec_row["単勝人気"]),
                                "odds": rec_row.get("想定単勝オッズ", 5.0),
                                "ev_grade": rec_row.get("期待値ランク", ""),
                                "win_rate": rec_row["勝率(%)"],
                                "score": rec_row["総合スコア"],
                                "reasons": ["AI総合指数上位", f"脚質「{rec_row['脚質']}」展開適合"],
                                "advice": f"WIN5候補として有力です。"
                            }

                        mark = rec_info.get("mark", "◎")
                        cfg = mark_colors.get(mark, mark_colors["ー"])

                        with st.container(border=True):
                            c_head1, c_head2 = st.columns([3, 2])
                            with c_head1:
                                st.markdown(
                                    f"""
                                    <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 4px;">
                                        <span style="background-color: {cfg['bg']}; color: white; padding: 3px 10px; border-radius: 4px; font-weight: bold; font-size: 14px;">
                                            {mark} {cfg['role']}
                                        </span>
                                        <span style="font-size: 18px; font-weight: bold;">
                                            {rec_info['horse_num']}番 {rec_info['horse_name']}
                                        </span>
                                        <span style="color: #666; font-size: 14px;">
                                            ({rec_info['jockey']})
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            with c_head2:
                                st.markdown(
                                    f"""
                                    <div style="text-align: right; font-size: 13px; color: #555; padding-top: 4px;">
                                        単勝 <b>{rec_info['pop']}位</b> ({rec_info.get('odds', 5.0)}倍) ｜ <b>{rec_info.get('ev_grade', '')}</b><br>
                                        勝率 <b>{rec_info['win_rate']}%</b> ｜ 総合スコア <b>{rec_info['score']}</b>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            st.markdown("**💬 AI推奨根拠:**")
                            for r in rec_info.get("reasons", []):
                                st.markdown(f"- {r}")
                            st.info(rec_info.get("advice", ""))

                    # 出走全頭の予測ランキング表
                    st.markdown("##### 📊 出走全頭のAI予測ランキング表")
                    display_cols = ["AI印", "予想着順", "馬番", "馬名", "単勝人気", "想定単勝オッズ", "騎手名", "脚質", "勝率(%)", "総合スコア", "期待値ランク"]
                    avail_cols = [c for c in display_cols if c in p_df.columns]
                    st.dataframe(p_df[avail_cols], use_container_width=True, hide_index=True)

                    # 過去走データカルテ
                    p_perf = meta.get("past_performances", {}) or r_diag.get("past_records", {})
                    if p_perf:
                        with st.expander("📋 このレースの出走馬 過去走詳細カルテ", expanded=False):
                            for h_name, p_info in p_perf.items():
                                st.write(f"- **{h_name}**: {p_info.get('last_race', '')} ｜ {p_info.get('form_trend', '')} ｜ 適性: {p_info.get('course_suitability', '')}")

            # ==========================================
            # 🏁 リアルタイム生還チェックシート
            # ==========================================
            st.markdown("---")
            st.subheader("🏁 リアルタイム生還チェックシート（レース当日のライブ追跡）")
            st.caption("レースが終わるごとに確定1着馬番を入力してください。生き残り点数とWIN5突破状況をリアルタイムで追跡します！")

            # レース結果一括取得ボタン (netkeiba URLが登録されている場合)
            has_urls = any(r_diag.get("url") or (edited_races[i].get("url") if i < len(edited_races) else "") for i, r_diag in enumerate(race_diagnostics))
            if has_urls and fetch_netkeiba_func:
                if st.button("🌐 netkeibaから5レースの確定着順を一括取得する", key="win5_fetch_results_btn"):
                    with st.spinner("🌐 netkeibaの結果ページから確定着順を最新取得中..."):
                        fetched_cnt = 0
                        for idx_f, r_diag in enumerate(race_diagnostics):
                            target_url = r_diag.get("url") or (edited_races[idx_f].get("url") if idx_f < len(edited_races) else "")
                            if target_url and "netkeiba.com" in target_url:
                                try:
                                    res_data, _ = fetch_netkeiba_func(target_url)
                                    if res_data and res_data.get("actual_ranks"):
                                        r_diag["meta"]["actual_ranks"] = res_data["actual_ranks"]
                                        if res_data.get("official_payouts"):
                                            r_diag["official_payouts"] = res_data["official_payouts"]
                                        for h_obj in res_data.get("horses", []):
                                            if res_data["actual_ranks"].get(h_obj["馬名"]) == 1:
                                                w_num = int(h_obj["馬番"])
                                                st.session_state[f"actual_win_{idx_f}"] = w_num
                                                r_diag["actual_winner"] = w_num
                                                if idx_f < len(st.session_state.get("win5_races_data", [])):
                                                    st.session_state["win5_races_data"][idx_f]["actual_winner"] = w_num
                                                fetched_cnt += 1
                                                break
                                except Exception:
                                    pass
                        if fetched_cnt > 0:
                            st.success(f"🎉 {fetched_cnt} レースの確定1着馬番を自動取得しました！")
                            st.rerun()
                        else:
                            st.info("確定着順データがまだ反映されていないか、レース開催前です。下記に手動で確定1着馬番を入力してください。")

            c_chk = st.columns(5)
            actual_win_nums = []
            for i in range(5):
                p_df = race_diagnostics[i]["pred_df"]
                detected_w = race_diagnostics[i].get("actual_winner", 0)
                init_val = int(st.session_state.get(f"actual_win_{i}", detected_w))
                with c_chk[i]:
                    val = st.number_input(
                        f"WIN{i+1} 確定1着馬番",
                        min_value=0,
                        max_value=18,
                        value=init_val,
                        key=f"actual_win_{i}",
                        help=f"WIN{i+1}の確定1着馬番。netkeibaから結果が取得された場合は自動セットされます。"
                    )
                    actual_win_nums.append(val)
                    if val > 0:
                        matched_h = p_df[p_df["馬番"] == val]
                        if not matched_h.empty:
                            h_row = matched_h.iloc[0]
                            st.caption(f"🏁 **{h_row['馬名']}** ({h_row.get('単勝人気', '-')}人気 ｜ 単勝{get_horse_effective_odds(h_row)}倍)")
                        else:
                            st.caption(f"🏁 {val}番")

            # 生還判定
            survived_points = actual_points
            alive_status = []
            is_dead = False

            for i in range(5):
                act_num = actual_win_nums[i]
                c_horses = active_selections[i]
                selected_nums = [int(r["馬番"]) for _, r in c_horses.iterrows()]
                sel_str = "・".join(str(n) for n in selected_nums) + "番"

                if act_num == 0:
                    alive_status.append("⏳ 待機中")
                elif act_num in selected_nums:
                    alive_status.append(f"⭕ 突破！ ({act_num}番的中)")
                    survived_points = max(1, int(survived_points / max(1, len(selected_nums))))
                else:
                    alive_status.append(f"❌ 脱落 (勝馬:{act_num}番 ｜ 買目:{sel_str})")
                    is_dead = True
                    survived_points = 0

            col_res_cards = st.columns(5)
            for i in range(5):
                with col_res_cards[i]:
                    status_text = alive_status[i]
                    if "⭕" in status_text:
                        st.success(f"**WIN{i+1}**: {status_text}")
                    elif "❌" in status_text:
                        st.error(f"**WIN{i+1}**: {status_text}")
                    else:
                        st.info(f"**WIN{i+1}**: {status_text}")

            passed_cnt = sum(1 for s in alive_status if "⭕" in s)
            is_win5_hit = (passed_cnt == 5 and all("⭕" in s for s in alive_status))
            all_races_entered = all(num > 0 for num in actual_win_nums)

            if is_win5_hit:
                st.balloons()
                st.success("🎉🎉🎉 **おめでとうございます！！！ WIN5 的中達成です！！！ 下の払戻金計算エリアで結果を確認してください！** 🎉🎉🎉")
            elif not is_dead and any("⭕" in s for s in alive_status):
                st.info(f"🔥 **現在 {passed_cnt} レース突破中！ 生き残り点数: {survived_points}点！ 次のレースへ進みます！**")

            # ------------------------------------------
            # 🔍 【過去検証モード】他プラン生還比較 ＆ 敗因勝因カルテ
            # ------------------------------------------
            if all_races_entered:
                with st.expander("🔍 **【過去レース検証カルテ】なぜ当たったか／外れたかの詳細分析 ＆ 他プランでの生還比較**", expanded=True):
                    st.markdown("##### 📊 各レースの確定勝馬とAI予想の対比分析")
                    diag_rows = []
                    for i, r_diag in enumerate(race_diagnostics):
                        act_w = actual_win_nums[i]
                        p_df = r_diag["pred_df"]
                        c_horses = active_selections[i]
                        sel_nums = [int(r["馬番"]) for _, r in c_horses.iterrows()]
                        w_match = p_df[p_df["馬番"] == act_w]
                        
                        if not w_match.empty:
                            w_row = w_match.iloc[0]
                            w_name = w_row["馬名"]
                            try:
                                w_pop = int(w_row.get("単勝人気", 99))
                            except Exception:
                                w_pop = 99
                            w_odds = get_horse_effective_odds(w_row)
                            raw_rank_val = w_row.get("予想着順数値")
                            if raw_rank_val is not None and not pd.isna(raw_rank_val):
                                try:
                                    w_rank = int(raw_rank_val)
                                except Exception:
                                    w_rank = 99
                            else:
                                raw_rank_str = str(w_row.get("予想着順", "99")).replace("着", "").strip()
                                try:
                                    w_rank = int(raw_rank_str)
                                except Exception:
                                    w_rank = 99
                            w_mark = w_row.get("AI印", "-")
                            try:
                                w_prob = float(w_row.get("勝率(%)", 0.0))
                            except Exception:
                                w_prob = 0.0
                        else:
                            w_name = f"{act_w}番"
                            w_pop = "-"
                            w_odds = "-"
                            w_rank = "-"
                            w_mark = "-"
                            w_prob = 0.0

                        is_hit = act_w in sel_nums
                        status_str = "⭕ 的中突破" if is_hit else "❌ 買い目漏れ"
                        quota_len = len(sel_nums)

                        # 要因コメント
                        if is_hit:
                            reason_note = f"AIが推奨した上位{quota_len}頭に合致。見事に1着的中！"
                        else:
                            if quota_len == 1:
                                reason_note = f"⚠️ 1頭固定レースで本命が敗退（勝馬は{w_pop}番人気/AI{w_rank}位）。1頭固定の突然死が敗因。"
                            else:
                                reason_note = f"波乱決着（勝馬は{w_pop}番人気/AI{w_rank}位）。{quota_len}頭カバーの範囲外でした。"

                        diag_rows.append({
                            "レース": f"WIN{i+1}",
                            "確定1着馬": f"{act_w}番 {w_name}",
                            "単勝人気/オッズ": f"{w_pop}人気 ({w_odds}倍)",
                            "AI評価": f"{w_mark} (AI{w_rank}位 / 勝率{w_prob}%)",
                            "今回選択頭数": f"{quota_len}頭",
                            "結果": status_str,
                            "分析コメント": reason_note
                        })

                    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)

                    # 他プランでの生還比較シミュレーション
                    st.markdown("##### 💡 もし他の買い目プランを選んでいたら？ 生還シミュレーション比較")
                    comp_plans = [
                        ("🛡️ 3,200円プラン (2×2×2×2×2)", [2, 2, 2, 2, 2], 3200, 32),
                        ("🎯 3,600円プラン (1×2×2×3×3)", [1, 2, 2, 3, 3], 3600, 36),
                        ("🏆 7,200円プラン (2×2×2×3×3)", [2, 2, 2, 3, 3], 7200, 72),
                        ("🚀 4,800円プラン (1×2×2×3×4)", [1, 2, 2, 3, 4], 4800, 48),
                        ("⚡ 2,400円プラン (1×2×2×2×3)", [1, 2, 2, 2, 3], 2400, 24)
                    ]

                    sorted_by_sol = sorted(race_diagnostics, key=lambda x: x["solid_score"], reverse=True)
                    comp_cols = st.columns(len(comp_plans))

                    for p_i, (p_name, p_quota, p_cost, p_pts) in enumerate(comp_plans):
                        # 各レースへのquota割り振り
                        sim_passed = 0
                        for r_rank_idx, r_d in enumerate(sorted_by_sol):
                            sim_q = p_quota[r_rank_idx]
                            sim_chosen = select_win5_horses_for_quota(r_d["pred_df"], sim_q)
                            sim_nums = [int(r["馬番"]) for _, r in sim_chosen.iterrows()]
                            act_winner_num = actual_win_nums[r_d["id"] - 1]
                            if act_winner_num in sim_nums:
                                sim_passed += 1

                        with comp_cols[p_i]:
                            plan_short = p_name.split()[0] + " " + p_name.split()[1]
                            if sim_passed == 5:
                                st.success(f"**{plan_short}**\n\n🎉 **5/5 的中！**\n({p_cost:,}円 / {p_pts}点)")
                            else:
                                st.warning(f"**{plan_short}**\n\n**{sim_passed}/5 レース突破**\n({p_cost:,}円 / {p_pts}点)")

                    # ワンクリックで確定1着馬を買い目に適用するボタン
                    if not is_win5_hit:
                        st.caption("💡 過去の検証・払戻金計算のために、確定1着馬を買い目にワンクリックで追加できます:")
                        if st.button("🪄 確定1着馬を買い目に自動追加して払戻金をシミュレーションする", key="btn_apply_winners_to_sel"):
                            for i, r_diag in enumerate(race_diagnostics):
                                w_num = actual_win_nums[i]
                                if w_num > 0:
                                    curr_sel = list(st.session_state.get("win5_custom_selections", {}).get(r_diag["id"], []))
                                    if not curr_sel:
                                        curr_sel = [int(r["馬番"]) for _, r in active_selections[i].iterrows()]
                                    if w_num not in curr_sel:
                                        curr_sel.append(w_num)
                                    st.session_state["win5_custom_selections"][r_diag["id"]] = sorted(curr_sel)
                            st.success("🎉 確定1着馬を各レースの買い目に自動追加しました！")
                            st.rerun()

            # ==========================================
            # 💰 WIN5 確定払戻金・回収率の自動計算
            # ==========================================
            st.markdown("---")
            st.subheader("💰 WIN5 確定払戻金・回収率の自動計算")
            st.caption("全5レース終了後、JRA公式発表の確定払戻金（100円あたり）を入力して、総払戻金額・純損益・回収率(ROI)を自動計算します。")

            # 単勝オッズ合成による参考配当の算出
            est_payout_synthetic = 0
            if all_races_entered:
                prod_odds = 1.0
                for i_w, r_diag in enumerate(race_diagnostics):
                    act_num = actual_win_nums[i_w]
                    p_df = r_diag["pred_df"]
                    m_row = p_df[p_df["馬番"] == act_num]
                    if not m_row.empty:
                        prod_odds *= get_horse_effective_odds(m_row.iloc[0])
                    else:
                        prod_odds *= 5.0
                est_payout_synthetic = int(min(600_000_000, max(100, prod_odds * 70)))

            c_pay_in1, c_pay_in2 = st.columns([2, 1])
            with c_pay_in1:
                official_win5_payout = st.number_input(
                    "WIN5 確定払戻金 (100円あたり / 円)",
                    min_value=0,
                    max_value=600_000_000,
                    value=int(st.session_state.get("win5_official_payout_val", 0)),
                    step=100,
                    key="win5_official_payout_input",
                    help="JRA公式結果のWIN5払戻金額（100円につき）を入力してください。"
                )
                st.session_state["win5_official_payout_val"] = official_win5_payout

                if est_payout_synthetic > 0 and official_win5_payout == 0:
                    st.caption(f"💡 5レースの確定オッズから推計した合成配当: **￥{est_payout_synthetic:,} 円**")
                    if st.button(f"👆 推計配当（￥{est_payout_synthetic:,} 円）を配当金に自動入力する", key="btn_apply_synth_payout"):
                        st.session_state["win5_official_payout_val"] = est_payout_synthetic
                        st.rerun()

            with c_pay_in2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                if is_win5_hit:
                    st.success("🎯 **WIN5 的中判定: 成立！** (1口的中)")
                elif all_races_entered:
                    st.error(f"❌ **WIN5 的中判定: 不的中** ({passed_cnt}/5レース突破)")
                else:
                    st.info(f"⏳ **WIN5 的中判定: 進行中** ({passed_cnt}レース突破)")

            # 回収パフォーマンス計算
            invest_cost = actual_cost  # 購入点数 * 100円
            hit_units = 1 if is_win5_hit else 0
            total_return = official_win5_payout * hit_units if is_win5_hit else 0
            net_profit = total_return - invest_cost
            roi_ratio = (total_return / invest_cost * 100.0) if invest_cost > 0 else 0.0

            # 的中時: 豪華ゴールド/グリーンカードでサマリー表示 (通常レース完全同等)
            if is_win5_hit and official_win5_payout > 0:
                roi_badge = "🔥 特大万馬券的中！超絶プラス！" if roi_ratio >= 1000.0 else ("🎯 プラス収支達成！" if roi_ratio >= 100.0 else "⚠️ トリガミ（投資額を下回りました）")
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #0d321d 0%, #1a5336 50%, #206b43 100%); border: 2px solid #2ecc71; border-radius: 12px; padding: 22px; box-shadow: 0 4px 15px rgba(46,204,113,0.3); margin-top: 10px; margin-bottom: 16px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-bottom: 8px;">
                            <span style="font-size: 22px; font-weight: bold; color: #f1c40f;">
                                👑 【WIN5 的中達成！】 総回収額: <span style="font-size: 32px; color: #2ecc71; text-shadow: 0 0 10px rgba(46,204,113,0.5);">￥{total_return:,} 円</span>
                            </span>
                            <span style="background-color: #2ecc71; color: #041b0f; font-weight: bold; padding: 6px 16px; border-radius: 20px; font-size: 14px;">
                                最高回収率 {roi_ratio:,.1f}% （純利益 +{net_profit:,}円）
                            </span>
                        </div>
                        <div style="font-size: 15px; margin-top: 10px; color: #ffffff; line-height: 1.6;">
                            🏆 <b>的中戦略</b>: WIN5 メリハリフォーメーション（購入点数: <b>{actual_points}点</b> ｜ 投資資金: <b>{invest_cost:,}円</b>）<br>
                            💎 <b>確定配当</b>: 100円あたり <b>￥{official_win5_payout:,} 円</b> ｜ <b>的中口数</b>: 1口<br>
                            📊 <b>判定評価</b>: {roi_badge}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                with c_m1:
                    st.metric("🥇 総回収額 (払戻金)", f"{total_return:,}円", delta=f"純利益 +{net_profit:,}円", delta_color="normal")
                with c_m2:
                    st.metric("💵 投資金額", f"{invest_cost:,}円", delta=f"{actual_points}点 × 100円")
                with c_m3:
                    st.metric("📈 最大回収率 (ROI)", f"{roi_ratio:,.1f}%", delta="投資リターン")
                with c_m4:
                    st.metric("💎 純利益", f"+{net_profit:,}円", delta="手元利益")
            elif all_races_entered and not is_win5_hit:
                st.warning(
                    f"⚠️ **【結果検証】 今回のWIN5は不的中となりました（{passed_cnt}/5レース突破 ｜ 投資額: {invest_cost:,}円 ｜ 払戻: 0円 ｜ 純損失: -{invest_cost:,}円）。**\n\n"
                    f"波乱や出遅れ等の5レースの確定着順を「競走馬メモリー」に蓄積し、次回レースでの精度向上・リベンジに繋げましょう。"
                )

            # ==========================================
            # 💾 5レース結果の自己学習・競走馬メモリー蓄積 (通常レース完全同期)
            # ==========================================
            st.markdown("---")
            st.subheader("💾 レース結果の記憶・競走馬メモリー蓄積 (通常レース完全同期)")
            st.markdown(
                """
                5レースの確定結果をもとに、**【Tier 1: 過去レース完全記憶】**と**【Tier 2: 出走全頭の競走馬メモリー（経験値・実力補正）】**を学習・永久保存します。  
                ※ 通常レースのAI予想と馬メモリー・過去レース記憶が**完全に共有**されるため、次回以降通常レースでもWIN5でも学習成果が自動発揮されます！
                """
            )

            # 各レースの2着・3着馬番（任意補足入力）アコーディオン
            with st.expander("📋 各レースの確定着順・補足設定（2着・3着馬番の指定）", expanded=False):
                st.caption("1着馬番は生還チェックシートと連動しています。2着・3着馬番も設定すると、連対・複勝好走ボーナスが馬メモリーに加算され、さらに高精度な学習が可能です。")
                col_sub_ranks = st.columns(5)
                extra_race_ranks = []
                for i, r_diag in enumerate(race_diagnostics):
                    p_df = r_diag["pred_df"]
                    p_horses = {int(r["馬番"]): str(r["馬名"]).strip() for _, r in p_df.iterrows()}
                    with col_sub_ranks[i]:
                        st.markdown(f"**WIN{r_diag['id']}**")
                        w_num = actual_win_nums[i]
                        w_name = p_horses.get(w_num, "未確定") if w_num > 0 else "未確定"
                        st.caption(f"1着: {w_num}番 {w_name}")
                        rank2 = st.selectbox(
                            f"WIN{r_diag['id']} 2着馬番",
                            options=[0] + list(p_horses.keys()),
                            format_func=lambda x, p_h=p_horses: "未設定" if x == 0 else f"{x}番 {p_h.get(x, '')}",
                            key=f"win5_rank2_{r_diag['id']}"
                        )
                        rank3 = st.selectbox(
                            f"WIN{r_diag['id']} 3着馬番",
                            options=[0] + list(p_horses.keys()),
                            format_func=lambda x, p_h=p_horses: "未設定" if x == 0 else f"{x}番 {p_h.get(x, '')}",
                            key=f"win5_rank3_{r_diag['id']}"
                        )
                        extra_race_ranks.append({"win": w_num, "second": rank2, "third": rank3})

            # 学習実行ボタン
            can_learn = any(num > 0 for num in actual_win_nums)
            if st.button("💾 5レースの確定結果をAIに自己学習させる（馬メモリー蓄積＆永久保存）", type="primary", disabled=not can_learn):
                with st.spinner("🤖 AIが5レースの結果を解析し、Tier 1完全記憶と出走全頭の競走馬メモリーを更新中..."):
                    if "race_database" not in st.session_state:
                        st.session_state["race_database"] = {}
                    if "horse_memory" not in st.session_state:
                        st.session_state["horse_memory"] = {}
                    if "training_history" not in st.session_state:
                        st.session_state["training_history"] = []

                    race_db = st.session_state["race_database"]
                    horse_mem = st.session_state["horse_memory"]
                    total_horses_learned = 0
                    races_learned_count = 0

                    for i, r_diag in enumerate(race_diagnostics):
                        p_df = r_diag["pred_df"]
                        w_num = actual_win_nums[i]
                        if w_num == 0:
                            continue

                        races_learned_count += 1
                        track_name = r_diag["track"]
                        race_dist = r_diag["distance"]
                        c_type = r_diag["course_type"]
                        pace_val = r_diag["pace"]
                        cond_val = r_diag["condition"]

                        # 確定着順辞書
                        actual_ranks_dict = {}
                        r_meta = r_diag.get("meta", {})
                        scraped_ranks = r_diag.get("actual_ranks") or (r_meta.get("actual_ranks") if isinstance(r_meta, dict) else None)

                        if scraped_ranks and isinstance(scraped_ranks, dict) and len(scraped_ranks) > 0:
                            actual_ranks_dict = scraped_ranks.copy()
                        else:
                            # 手動入力データから着順構築
                            sub_r = extra_race_ranks[i]
                            r2_num = sub_r["second"]
                            r3_num = sub_r["third"]

                            # 馬番から馬名マッピング
                            num_to_name = {int(r["馬番"]): str(r["馬名"]).strip() for _, r in p_df.iterrows()}
                            for h_n, h_nm in num_to_name.items():
                                if h_n == w_num:
                                    actual_ranks_dict[h_nm] = 1
                                elif r2_num > 0 and h_n == r2_num:
                                    actual_ranks_dict[h_nm] = 2
                                elif r3_num > 0 and h_n == r3_num:
                                    actual_ranks_dict[h_nm] = 3
                                else:
                                    actual_ranks_dict[h_nm] = 8

                        # 確定勝馬が確実に1着として登録されるよう安全補正
                        num_to_name_all = {int(r["馬番"]): str(r["馬名"]).strip() for _, r in p_df.iterrows()}
                        winner_horse_name = num_to_name_all.get(w_num)
                        if winner_horse_name:
                            actual_ranks_dict[winner_horse_name] = 1

                        # 【Tier 1】 過去レース完全記憶登録
                        if generate_race_sig_func:
                            race_sig = generate_race_sig_func(p_df, track_name, race_dist)
                        else:
                            clean_names = sorted([str(n).strip() for n in p_df["馬名"].dropna() if str(n).strip()])
                            track_key = str(track_name).split()[0]
                            race_sig = f"{track_key}_{race_dist}m_" + "_".join(clean_names)

                        race_label = f"{track_name.split()[0]} {race_dist}m ({c_type}) [WIN{r_diag['id']}]"
                        race_db[race_sig] = {
                            "race_name": race_label,
                            "track": track_name,
                            "track_clean": track_name.split()[0],
                            "distance": race_dist,
                            "course_type": c_type,
                            "condition": cond_val,
                            "pace": pace_val,
                            "actual_ranks": actual_ranks_dict,
                            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                            "horse_count": len(p_df)
                        }

                        # 【Tier 2】 出走馬ごとの競走馬メモリー更新
                        pop_map = dict(zip(p_df["馬番"], p_df["単勝人気"]))
                        for _, r_horse in p_df.iterrows():
                            h_name = str(r_horse["馬名"]).strip()
                            if not h_name:
                                continue
                            total_horses_learned += 1
                            act_rank = actual_ranks_dict.get(h_name, 8)
                            h_num = r_horse.get("馬番")
                            pop = int(pop_map.get(h_num, 5)) if not pd.isna(pop_map.get(h_num)) else 5

                            h_data = horse_mem.get(h_name, {
                                "starts": 0,
                                "wins": 0,
                                "top3": 0,
                                "ability_adj": 0.0,
                                "last_ranks": []
                            })

                            h_data["starts"] += 1
                            if act_rank == 1:
                                h_data["wins"] += 1
                            if act_rank <= 3:
                                h_data["top3"] += 1

                            ranks_list = h_data.get("last_ranks", [])
                            ranks_list.append(act_rank)
                            h_data["last_ranks"] = ranks_list[-5:]

                            # 能力補正の算出: 1着+3.0, 2着+1.8, 3着+1.0, 8着以下-1.0
                            rank_bonus = 0.0
                            if act_rank == 1:
                                rank_bonus += 3.0
                            elif act_rank == 2:
                                rank_bonus += 1.8
                            elif act_rank == 3:
                                rank_bonus += 1.0
                            elif act_rank >= 8:
                                rank_bonus -= 1.0

                            # 人気着順ギャップ補正（人気以上に好走した実力馬を底上げ、大敗を下方修正）
                            gap = pop - act_rank
                            gap_bonus = gap * 0.5

                            delta_ab = rank_bonus + gap_bonus
                            new_adj = h_data.get("ability_adj", 0.0) + (delta_ab * 0.4)
                            h_data["ability_adj"] = round(float(np.clip(new_adj, -6.0, 10.0)), 2)

                            horse_mem[h_name] = h_data

                    # 学習履歴への記録
                    cur_w = learned_weights
                    w_a = float(cur_w.get("ability", 1.5))
                    w_p = float(cur_w.get("pace", 1.0))
                    w_j = float(cur_w.get("jockey", 1.0))
                    w_e = float(cur_w.get("escape_dynamic", 1.2))
                    w_s = float(cur_w.get("synergy", 1.1))

                    hist_entry = {
                        "レース番号": f"第{len(st.session_state['training_history']) + 1}戦 (WIN5)",
                        "学習日時": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        "コース・距離": "WIN5 (5レース一括)",
                        "想定ペース": f"{passed_cnt}/5レース突破",
                        "最大回収額": f"{total_return:,}円" if is_win5_hit else "0円",
                        "最高回収プラン": "WIN5 メリハリフォーメーション",
                        "最大回収率": f"{roi_ratio:.1f}%" if is_win5_hit else "0.0%",
                        "3連複": "―",
                        "馬連BOX(3点)": "―",
                        "ワイドBOX(3点)": "―",
                        "単勝的中": f"{passed_cnt}/5レース的中",
                        "3着内好走": f"勝馬確定{races_learned_count}頭",
                        "モデル重み(黄金比率固定)": f"{w_a:.2f}/{w_p:.2f}/{w_j:.2f}/{w_e:.2f}/{w_s:.2f}"
                    }
                    st.session_state["training_history"].append(hist_entry)

                    # モデルファイルへの永続化保存
                    if save_ai_model_func:
                        save_ai_model_func(
                            st.session_state["learned_weights"],
                            st.session_state["training_history"],
                            st.session_state["race_database"],
                            st.session_state["horse_memory"]
                        )

                    # AI分析解説インサイト
                    insights = []
                    insights.append(f"🧠 **【過去レース完全記憶（Tier 1）】** 確定結果を入力した全{races_learned_count}レースの着順データを永久保存しました。次回同一レースを入力した際は100%完全再現して的中させます。")
                    insights.append(f"🐎 **【個別競走馬メモリー更新（Tier 2）】** 出走全{total_horses_learned}頭の実績・好走・人気着順ギャップを馬メモリーに反映しました。別レース（通常レース含む）に出走した際にも実力差として自動適用されます。")
                    if is_win5_hit and official_win5_payout > 0:
                        insights.append(f"💰 **【WIN5 回収達成！】** 払戻金 **￥{total_return:,} 円**（回収率 **{roi_ratio:.1f}%** / 純利益 **+{net_profit:,} 円**）の特大勝利を記録しました！")
                    else:
                        insights.append(f"📊 **【次戦への蓄積】** {passed_cnt}/5 レース突破の実績と波乱データを学習。次回開催のWIN5・通常レースの予想精度が底上げされました。")
                    insights.append("🛡️ **【黄金比率セーフティ】** 重みパラメータは実績十分の黄金比率（能力1.5 / 展開1.0 / 騎手1.0 / 力学1.2 / シナジー1.1）を強固に維持。波乱によるAIのブレ・過学習を完全に防止しています。")

                    st.session_state["win5_learning_success"] = {
                        "weights": (w_a, w_p, w_j, w_e, w_s),
                        "insights": insights,
                        "races_count": races_learned_count,
                        "horses_count": total_horses_learned
                    }

                    st.rerun()

            # 学習完了インサイトの常時表示
            if st.session_state.get("win5_learning_success"):
                succ_data = st.session_state["win5_learning_success"]
                w_a, w_p, w_j, w_e, w_s = succ_data["weights"]
                insights = succ_data["insights"]

                st.success(f"🎉 **WIN5 5レースのAI学習が完了しました！出走全{succ_data.get('horses_count', 0)}頭の競走馬メモリーと確定着順が安全に永久保存されました。**")
                col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
                with col_w1:
                    st.metric("1️⃣ 基礎能力", f"{w_a:.2f}", delta="黄金比率固定", delta_color="normal")
                with col_w2:
                    st.metric("2️⃣ ペース・展開", f"{w_p:.2f}", delta="黄金比率固定", delta_color="normal")
                with col_w3:
                    st.metric("3️⃣ 騎手実績", f"{w_j:.2f}", delta="黄金比率固定", delta_color="normal")
                with col_w4:
                    st.metric("4️⃣ 脚質力学", f"{w_e:.2f}", delta="黄金比率固定", delta_color="normal")
                with col_w5:
                    st.metric("5️⃣ 血統シナジー", f"{w_s:.2f}", delta="黄金比率固定", delta_color="normal")

                st.markdown("##### 💡 AI分析・学習インサイト")
                for ins in insights:
                    st.markdown(f"- {ins}")
