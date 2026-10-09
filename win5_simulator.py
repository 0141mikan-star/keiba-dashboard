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

def render_win5_simulator(
    calculate_predictions_func,
    learned_weights: dict,
    fetch_netkeiba_func=None,
    select_recommended_func=None,
    generate_comment_func=None,
    diagnose_volatility_func=None,
    track_options: list = None
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
                "🎯 【本命・王道】3,600円プラン (36点) : 1頭 × 2頭 × 2頭 × 3頭 × 3頭 (おすすめ！)",
                "🚀 【手広く安心】4,800円プラン (48点) : 1頭 × 2頭 × 2頭 × 3頭 × 4頭",
                "⚡ 【超エコノミー】2,400円プラン (24点) : 1頭 × 2頭 × 2頭 × 2頭 × 3頭",
                "🛠️ 【カスタム配分】各レースの頭数を自由指定"
            ],
            index=0
        )
    with plan_col2:
        if "3,600円" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 3]
            base_cost = 3600
            base_points = 36
            st.info("💡 **おすすめNo.1**\n鉄板1レースを本命◎1点で突破し、他を2〜3頭でカバーする最もバランスの良い王道買い目！")
        elif "4,800円" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 4]
            base_cost = 4800
            base_points = 48
            st.info("💡 **カバー重視**\n大荒れが予想されるメイン戦や混戦レースを4頭まで手広く抑える安心設計！")
        elif "2,400円" in plan_choice:
            quota_pattern = [1, 2, 2, 2, 3]
            base_cost = 2400
            base_points = 24
            st.info("💡 **超節約エコノミー**\nわずか2,400円（24点）で毎週気軽にWIN5に挑戦できる高回収率特化プラン！")
        else:
            quota_pattern = [1, 2, 2, 3, 3]
            base_cost = 3600
            base_points = 36
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
                                if "想定単勝オッズ" not in horses_df.columns:
                                    horses_df["想定単勝オッズ"] = 10.0
                                if "斤量" not in horses_df.columns:
                                    horses_df["斤量"] = 56.0

                                r_name_formatted = f"WIN{idx+1}: {res_data['race_name']} ({res_data['course_type']}{res_data['distance']}m)"
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
                                    "actual_winner": 0
                                }
                                fetched_details.append(f"**WIN{idx+1}**: {res_data['race_name']} ({len(horses_df)}頭)")

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
                        if "想定単勝オッズ" not in horses_df.columns:
                            horses_df["想定単勝オッズ"] = 10.0
                        if "斤量" not in horses_df.columns:
                            horses_df["斤量"] = 56.0

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
                            "actual_winner": 0
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
                "horses": h_df_edited.dropna(subset=["馬名"])
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
                    "past_records": r_data.get("past_records", {})
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

            # 各レースの初期選択馬
            active_selections = []
            for r_diag in race_diagnostics:
                r_id = r_diag["id"]
                p_df = r_diag["pred_df"]
                q = r_diag["assigned_quota"]

                # 以前にユーザーが選んだものがあればそれを優先、なければAI推奨上位q頭
                saved_sel = st.session_state["win5_custom_selections"].get(r_id)
                if saved_sel is not None and len(saved_sel) > 0:
                    chosen_horses = p_df[p_df["馬番"].isin(saved_sel)]
                    if chosen_horses.empty:
                        chosen_horses = p_df.iloc[:q]
                else:
                    chosen_horses = p_df.iloc[:q]
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

            c_chk = st.columns(5)
            actual_win_nums = []
            for i in range(5):
                with c_chk[i]:
                    val = st.number_input(f"WIN{i+1} 確定1着馬番", min_value=0, max_value=18, value=0, key=f"actual_win_{i}")
                    actual_win_nums.append(val)

            # 生還判定
            survived_points = actual_points
            alive_status = []
            is_dead = False

            for i in range(5):
                act_num = actual_win_nums[i]
                c_horses = active_selections[i]
                selected_nums = [int(r["馬番"]) for _, r in c_horses.iterrows()]

                if act_num == 0:
                    alive_status.append("⏳ 待機中")
                elif act_num in selected_nums:
                    alive_status.append("⭕ 突破！")
                    survived_points = max(1, int(survived_points / max(1, len(selected_nums))))
                else:
                    alive_status.append(f"❌ 脱落 ({act_num}番)")
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

            if all("⭕" in s for s in alive_status):
                st.balloons()
                st.success("🎉🎉🎉 **おめでとうございます！！！ WIN5 的中達成です！！！ 払戻金を確認してください！** 🎉🎉🎉")
            elif not is_dead and any("⭕" in s for s in alive_status):
                passed_cnt = sum(1 for s in alive_status if "⭕" in s)
                st.info(f"🔥 **現在 {passed_cnt} レース突破中！ 生き残り点数: {survived_points}点！ 次のレースへ進みます！**")
