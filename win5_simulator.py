import streamlit as st
import pandas as pd
import numpy as np

def get_default_win5_races() -> list[dict]:
    """WIN5用のリアルなサンプル5レースデータ"""
    return [
        {
            "id": 1,
            "name": "WIN1: 中山10R 外房S (ダ1200m)",
            "track": "中山 (直線310m・急坂小回り前残り)",
            "distance": 1200,
            "course_type": "ダート",
            "pace": "ハイペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "エスカル", "単勝人気": 1, "騎手名": "戸崎圭太", "脚質": "逃げ"},
                {"馬番": 2, "馬名": "コパノパサディナ", "単勝人気": 2, "騎手名": "横山武史", "脚質": "先行"},
                {"馬番": 3, "馬名": "スマートフォルス", "単勝人気": 3, "騎手名": "ルメール", "脚質": "先行"},
                {"馬番": 4, "馬名": "サザンエルフ", "単勝人気": 4, "騎手名": "津村明秀", "脚質": "差し"},
                {"馬番": 5, "馬名": "アイアムユウシュン", "単勝人気": 5, "騎手名": "佐々木大輔", "脚質": "差し"},
                {"馬番": 6, "馬名": "ショウナンカブト", "単勝人気": 6, "騎手名": "石川裕紀人", "脚質": "追込"},
            ])
        },
        {
            "id": 2,
            "name": "WIN2: 中京10R 甲南S (ダ1800m)",
            "track": "中京 (直線412m・急坂タフ)",
            "distance": 1800,
            "course_type": "ダート",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "バハルダール", "単勝人気": 1, "騎手名": "川田将雅", "脚質": "先行"},
                {"馬番": 2, "馬名": "クールミラボー", "単勝人気": 2, "騎手名": "松山弘平", "脚質": "差し"},
                {"馬番": 3, "馬名": "サンライズアリオン", "単勝人気": 3, "騎手名": "坂井瑠星", "脚質": "先行"},
                {"馬番": 4, "馬名": "カズペトシーン", "単勝人気": 4, "騎手名": "西村淳也", "脚質": "差し"},
                {"馬番": 5, "馬名": "オメガタキシード", "単勝人気": 5, "騎手名": "岩田望来", "脚質": "追込"},
            ])
        },
        {
            "id": 3,
            "name": "WIN3: 中山11R オールカマー (芝2200m)",
            "track": "中山 (直線310m・急坂小回り前残り)",
            "distance": 2200,
            "course_type": "芝",
            "pace": "スローペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "レーベンスティール", "単勝人気": 1, "騎手名": "ルメール", "脚質": "先行"},
                {"馬番": 2, "馬名": "サヴォーナ", "単勝人気": 2, "騎手名": "池添謙一", "脚質": "先行"},
                {"馬番": 3, "馬名": "ステラヴェローチェ", "単勝人気": 3, "騎手名": "横山典弘", "脚質": "差し"},
                {"馬番": 4, "馬名": "リカンカブール", "単勝人気": 4, "騎手名": "津村明秀", "脚質": "先行"},
                {"馬番": 5, "馬名": "キラーアビリティ", "単勝人気": 5, "騎手名": "北村友一", "脚質": "差し"},
            ])
        },
        {
            "id": 4,
            "name": "WIN4: 中京11R 神戸新聞杯 (芝2200m)",
            "track": "中京 (直線412m・急坂タフ)",
            "distance": 2200,
            "course_type": "芝",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "メイショウタバル", "単勝人気": 2, "騎手名": "浜中俊", "脚質": "逃げ"},
                {"馬番": 2, "馬名": "ジューンテイク", "単勝人気": 3, "騎手名": "藤岡佑介", "脚質": "先行"},
                {"馬番": 3, "馬名": "メリオーレム", "単勝人気": 1, "騎手名": "川田将雅", "脚質": "差し"},
                {"馬番": 4, "馬名": "ショウナンラプンタ", "単勝人気": 4, "騎手名": "鮫島克駿", "脚質": "差し"},
                {"馬番": 5, "馬名": "オールセインツ", "単勝人気": 5, "騎手名": "坂井瑠星", "脚質": "追込"},
            ])
        },
        {
            "id": 5,
            "name": "WIN5: 東京11R 天皇賞(秋) / メインG1 (芝2000m)",
            "track": "東京 (直線525m・末脚炸裂)",
            "distance": 2000,
            "course_type": "芝",
            "pace": "ミドルペース",
            "condition": "良馬場 (晴・乾燥 / 高速スピード＆キレ味)",
            "horses": pd.DataFrame([
                {"馬番": 1, "馬名": "ドウデュース", "単勝人気": 2, "騎手名": "武豊", "脚質": "差し"},
                {"馬番": 2, "馬名": "リバティアイランド", "単勝人気": 1, "騎手名": "川田将雅", "脚質": "先行"},
                {"馬番": 3, "馬名": "ベラジオオペラ", "単勝人気": 3, "騎手名": "横山和生", "脚質": "先行"},
                {"馬番": 4, "馬名": "ジャスティンパレス", "単勝人気": 4, "騎手名": "坂井瑠星", "脚質": "追込"},
                {"馬番": 5, "馬名": "レーベンスティール", "単勝人気": 5, "騎手名": "ルメール", "脚質": "先行"},
                {"馬番": 6, "馬名": "ソールオリエンス", "単勝人気": 6, "騎手名": "横山武史", "脚質": "追込"},
            ])
        }
    ]

def render_win5_simulator(calculate_predictions_func, learned_weights: dict):
    """WIN5攻略シミュレーターのメインUI"""
    st.title("🏆 WIN5 攻略シミュレーター")
    st.markdown(
        """
        **低予算（3,000円〜5,000円前後）でWIN5の高額配当を射止めるAIメリハリ戦略システム**です。
        全レースを手広く買うのではなく、**「鉄板レースを1頭で突破し、浮いた予算を混戦レースに回す」**ことで、
        わずか**36点（3,600円）〜48点（4,800円）**で最大のカバー率と高回収率を実現します！
        """
    )

    # 予算プランの選択
    st.markdown("### 💰 予算プランの選択")
    plan_choice = st.radio(
        "目的に合った買い目プランを選択してください:",
        [
            "🎯 【本命・王道】3,600円プラン (36点) : 1頭 × 2頭 × 2頭 × 3頭 × 3頭",
            "🚀 【手広く安心】4,800円プラン (48点) : 1頭 × 2頭 × 2頭 × 3頭 × 4頭",
            "⚡ 【超エコノミー】2,400円プラン (24点) : 1頭 × 2頭 × 2頭 × 2頭 × 3頭"
        ],
        index=0
    )

    if "3,600円" in plan_choice:
        quota_pattern = [1, 2, 2, 3, 3]
        plan_cost = 3600
        plan_points = 36
    elif "4,800円" in plan_choice:
        quota_pattern = [1, 2, 2, 3, 4]
        plan_cost = 4800
        plan_points = 48
    else:
        quota_pattern = [1, 2, 2, 2, 3]
        plan_cost = 2400
        plan_points = 24

    # セッション状態にWIN5レースデータを保持
    if "win5_races_data" not in st.session_state:
        st.session_state["win5_races_data"] = get_default_win5_races()

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🎲 サンプル5レースを一括ロード (即座にお試し)", use_container_width=True):
            st.session_state["win5_races_data"] = get_default_win5_races()
            st.success("サンプル5レースをセットしました！下のボタンでAI診断を実行できます。")
            st.rerun()

    # 5レースの出馬表設定エリア
    st.markdown("---")
    st.subheader("📋 WIN5 対象5レースの出走馬データ設定")
    st.caption("各レース名や出走馬情報を編集できます（新聞やネット競馬を見ながら入力可能）。")

    race_tabs = st.tabs([f"第{i}戦: WIN{i}" for i in range(1, 6)])
    edited_races = []

    for idx, r_tab in enumerate(race_tabs):
        r_info = st.session_state["win5_races_data"][idx]
        with r_tab:
            c_r1, c_r2, c_r3 = st.columns([2, 1, 1])
            with c_r1:
                r_name = st.text_input(f"レース名 (WIN{idx+1})", value=r_info["name"], key=f"win5_name_{idx}")
            with c_r2:
                r_dist = st.number_input(f"距離 (m)", value=int(r_info["distance"]), step=100, key=f"win5_dist_{idx}")
            with c_r3:
                r_ctype = st.selectbox(f"コース種別", ["芝", "ダート", "ばんえい"], index=0 if r_info["course_type"]=="芝" else 1, key=f"win5_ctype_{idx}")

            c_r4, c_r5 = st.columns([1, 1])
            with c_r4:
                r_track = st.selectbox(
                    "競馬場",
                    [
                        "東京 (直線525m・末脚炸裂)", "中山 (直線310m・急坂小回り前残り)",
                        "阪神 (直線474m・急坂タフ)", "京都 (直線404m・平坦高速持続)",
                        "中京 (直線412m・急坂タフ)", "新潟 (直線659m・日本最長直線強襲)",
                        "小回り平坦 (前残り絶対有利)", "大井 (右回り / 直線386m・南関王道・外差し届く)",
                        "盛岡 (左回り / 起伏4.4mタフ・地方唯一芝併設)", "金沢 (右回り / 直線236m・超小回り前残り鉄則)",
                        "帯広（ば） (直線200m / 世界唯一ばんえい・障害登坂)"
                    ],
                    index=0 if "東京" in r_info["track"] else (1 if "中山" in r_info["track"] else 4),
                    key=f"win5_track_{idx}"
                )
            with c_r5:
                r_pace = st.selectbox("想定ペース", ["スローペース", "ミドルペース", "ハイペース"], index=1 if r_info["pace"]=="ミドルペース" else (0 if r_info["pace"]=="スローペース" else 2), key=f"win5_pace_{idx}")

            st.caption("出走馬データ (馬番・馬名・単勝人気・騎手名・脚質):")
            h_df = st.data_editor(
                r_info["horses"],
                column_config={
                    "馬番": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, required=True),
                    "馬名": st.column_config.TextColumn("馬名", required=True),
                    "単勝人気": st.column_config.NumberColumn("人気", min_value=1, max_value=18, required=True),
                    "騎手名": st.column_config.TextColumn("騎手名", required=True),
                    "脚質": st.column_config.SelectboxColumn("脚質", options=["逃げ", "先行", "差し", "追込"], required=True)
                },
                num_rows="dynamic",
                use_container_width=True,
                key=f"win5_editor_{idx}"
            )

            edited_races.append({
                "id": idx + 1,
                "name": r_name,
                "track": r_track,
                "distance": r_dist,
                "course_type": r_ctype,
                "pace": r_pace,
                "condition": r_info.get("condition", "良馬場 (晴・乾燥 / 高速スピード＆キレ味)"),
                "horses": h_df.dropna(subset=["馬名"])
            })

    st.session_state["win5_races_data"] = edited_races

    # ==========================================
    # AI診断 & フォーメーション生成実行
    # ==========================================
    st.markdown("---")
    btn_calc_win5 = st.button("🚀 WIN5 AI診断 ＆ 最適フォーメーションを自動生成", type="primary", use_container_width=True)

    if btn_calc_win5 or st.session_state.get("has_run_win5", False):
        st.session_state["has_run_win5"] = True

        with st.spinner("🤖 AIが5レースを同時解析し、鉄板度・波乱度から最適買い目を算出中..."):
            race_diagnostics = []

            for r_data in edited_races:
                df = r_data["horses"]
                if df.empty or len(df) < 2:
                    continue

                # 予測計算
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

                # 勝率上位馬
                top1 = pred_res.iloc[0]
                top2 = pred_res.iloc[1] if len(pred_res) >= 2 else None
                top3 = pred_res.iloc[2] if len(pred_res) >= 3 else None
                top4 = pred_res.iloc[3] if len(pred_res) >= 4 else None

                win1 = float(top1["勝率(%)"])
                win2 = float(top2["勝率(%)"]) if top2 is not None else 0.0
                win_gap = win1 - win2  # 1位と2位の勝率差

                # 鉄板度スコア (勝率が高く、2位を大きく引き離しているほど鉄板)
                solid_score = (win1 * 1.5) + (win_gap * 2.0)

                race_diagnostics.append({
                    "id": r_data["id"],
                    "name": r_data["name"],
                    "pred_df": pred_res,
                    "top1_horse": top1,
                    "top2_horse": top2,
                    "top3_horse": top3,
                    "top4_horse": top4,
                    "win1": win1,
                    "win2": win2,
                    "win_gap": win_gap,
                    "solid_score": solid_score
                })

            if len(race_diagnostics) < 5:
                st.warning("⚠️ 5つのレースすべてに出走馬データを入力してください。")
                return

            # 鉄板度順にソートして、配分頭数を割り振る
            # 最も鉄板度が高いレースに 1頭、次いで 2頭、混戦レースに 3〜4頭を割り当て
            sorted_by_solidity = sorted(race_diagnostics, key=lambda x: x["solid_score"], reverse=True)

            # quota_pattern = [1, 2, 2, 3, 3] など
            for rank_idx, r_diag in enumerate(sorted_by_solidity):
                r_diag["assigned_quota"] = quota_pattern[rank_idx]
                r_diag["solidity_rank"] = rank_idx + 1

            # レース番号順（WIN1〜WIN5）に戻す
            race_diagnostics = sorted(race_diagnostics, key=lambda x: x["id"])

            # ------------------------------------------
            # 買い目フォーメーションサマリー
            # ------------------------------------------
            st.subheader("🎯 AI最適フォーメーション決定！")
            st.info(
                f"💡 **戦略プラン**: **{plan_choice.split(':')[0]}** ｜ "
                f"**合計購入点数: {plan_points}点** ｜ **必要資金: {plan_cost:,}円** (1点100円)"
            )

            col_res_list = st.columns(5)
            final_selections = []

            for i, r_diag in enumerate(race_diagnostics):
                q = r_diag["assigned_quota"]
                p_df = r_diag["pred_df"]
                chosen_horses = p_df.iloc[:q]
                final_selections.append(chosen_horses)

                with col_res_list[i]:
                    badge = "🥇 【1点突破！】" if q == 1 else ("🥈 【2頭カバー】" if q == 2 else "⚠️ 【混戦警戒】")
                    card_color = "#1e3c72" if q == 1 else ("#2a5298" if q == 2 else "#4a2a72")
                    st.markdown(
                        f"""
                        <div style="background: {card_color}; padding: 10px; border-radius: 8px; color: white; margin-bottom: 8px; text-align: center;">
                            <b>WIN{r_diag['id']}</b><br>
                            <span style="font-size: 0.85rem;">{badge}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.caption(f"**{r_diag['name'].split(':')[1] if ':' in r_diag['name'] else r_diag['name']}**")
                    
                    nums_str = ", ".join([f"**{int(r['馬番'])}番**" for _, r in chosen_horses.iterrows()])
                    st.markdown(f"🎯 **選択: {nums_str}** ({q}頭)")
                    for _, h in chosen_horses.iterrows():
                        st.write(f"- {int(h['馬番'])}番 {h['馬名']} ({h['勝率(%)']}%)")

            # 買い目詳細＆IPAT用コード
            st.markdown("---")
            col_sum1, col_sum2 = st.columns([2, 1])
            with col_sum1:
                st.subheader("📝 IPAT投票用 買い目まとめ")
                ipat_lines = []
                for i, r_diag in enumerate(race_diagnostics):
                    q = r_diag["assigned_quota"]
                    c_horses = final_selections[i]
                    nums = [str(int(r["馬番"])) for _, r in c_horses.iterrows()]
                    ipat_lines.append(f"WIN{r_diag['id']}: {', '.join(nums)} ({q}頭)")
                
                st.code("\n".join(ipat_lines), language="text")
                st.caption("※ 上記の馬番を即PATまたはマークカード（フォーメーション）に転記して投票してください。")

            with col_sum2:
                st.subheader("📊 的中確率＆配当期待")
                total_cov = np.prod([
                    min(0.95, sum(final_selections[i]["勝率(%)"]) / 100.0) for i in range(5)
                ]) * 100.0
                st.metric("🎯 5レース総合カバー期待率", f"{total_cov:.2f}%")
                st.write(f"💵 **購入資金**: **{plan_cost:,}円**")
                st.write(f"🎫 **購入点数**: **{plan_points}点**")
                st.caption("※ 各レースでAI上位の勝率合計を掛け合わせた総合通過確率の推計値です。")

            # ------------------------------------------
            # 🏁 リアルタイム生還チェックシート
            # ------------------------------------------
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
            survived_points = plan_points
            alive_status = []
            is_dead = False

            for i in range(5):
                act_num = actual_win_nums[i]
                c_horses = final_selections[i]
                selected_nums = [int(r["馬番"]) for _, r in c_horses.iterrows()]

                if act_num == 0:
                    alive_status.append("⏳ 待機中")
                elif act_num in selected_nums:
                    alive_status.append("⭕ 突破！")
                    survived_points = int(survived_points / len(selected_nums))
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
                st.info(f"🔥 **現在 {passed_cnt} レース突破中！ 生き残り点数: {max(1, survived_points)}点！ 次のレースへ進みます！**")
