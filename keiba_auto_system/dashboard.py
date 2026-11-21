import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬AI自動予測ダッシュボード", layout="wide")
st.title("🏇 競馬AI 自動予測 ＆ 収支ダッシュボード")

DB_PATH = Path(__file__).resolve().parent / "data" / "keiba.db"

# Supabase接続チェック（Streamlit CloudのSecretsまたは環境変数）
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

sb_client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        sb_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        st.caption("☁️ データソース: Supabase (クラウド同期)")
    except Exception:
        st.caption("💻 データソース: ローカル SQLite")
else:
    st.caption("💻 データソース: ローカル SQLite")

if st.button("🔄 最新状態に更新"):
    st.rerun()

def load_table(table_name: str) -> pd.DataFrame:
    if sb_client is not None:
        res = sb_client.table(table_name).select("*").execute()
        return pd.DataFrame(res.data)
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

tab1, tab2, tab3 = st.tabs(["📊 当日レース ＆ 期待値", "🏆 WIN5 フォーメーション", "💰 自動収支レポート"])

# ==========================================
# TAB 1: 当日レース
# ==========================================
with tab1:
    races_df = load_table("races")
    if races_df.empty:
        st.info("まだレースデータがありません。`scheduler_backend.py` を起動してください。")
    else:
        races_df = races_df.sort_values(["max_ev", "start_time"], ascending=[False, True])
        races_df["当たりやすさ"] = races_df["difficulty_star"].apply(lambda x: "★" * int(x) + "☆" * (5 - int(x)))
        st.dataframe(
            races_df[["race_name", "start_time", "当たりやすさ", "entropy", "max_ev", "status"]],
            use_container_width=True
        )

        race_options = dict(zip(races_df["race_id"], races_df["race_name"] + " (" + races_df["race_id"] + ")"))
        selected = st.selectbox("詳細を見るレースを選択", list(race_options.keys()), format_func=lambda x: race_options[x])

        preds_df = load_table("predictions")
        if not preds_df.empty:
            sub = preds_df[preds_df["race_id"] == selected].sort_values("ev", ascending=False).copy()
            sub["予測勝率(%)"] = (sub["win_prob"] * 100).round(1)
            sub["期待値"] = sub["ev"].round(2)
            st.subheader(f"📊 {race_options[selected]} 買い目詳細")
            st.dataframe(
                sub[["umaban", "bamei", "予測勝率(%)", "odds", "期待値", "recommended_bet"]].rename(
                    columns={"umaban": "馬番", "bamei": "馬名", "odds": "オッズ", "recommended_bet": "推奨額(円)"}
                ),
                use_container_width=True
            )

# ==========================================
# TAB 2: WIN5
# ==========================================
with tab2:
    win5_df = load_table("win5_plans")
    if win5_df.empty:
        st.info("まだWIN5の買い目プランがありません（日曜日の1R目発走15分前に自動生成されます）。")
    else:
        win5_df = win5_df.sort_values("date_str", ascending=False)
        selected_date = st.selectbox("開催日を選択", win5_df["date_str"].tolist())
        row = win5_df[win5_df["date_str"] == selected_date].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("総点数", f"{int(row['total_points'])} 点")
        c2.metric("投資合計", f"{int(row['total_cost']):,} 円")
        c3.metric("想定的中率", f"{float(row['hit_prob'])*100:.2f} %")

        plans = json.loads(row["plan_json"])
        for leg in plans:
            stars = "★" * int(leg.get("stars", 3)) + "☆" * (5 - int(leg.get("stars", 3)))
            st.markdown(f"### {leg['leg']}R目：{leg['race_name']} （{leg['count']}頭 / {stars} H={leg['entropy']}）")
            picks_df = pd.DataFrame(leg["picks"])
            picks_df["予測勝率(%)"] = (picks_df["win_prob"] * 100).round(1)
            picks_df["期待値"] = picks_df["ev"].round(2)
            st.dataframe(
                picks_df[["umaban", "bamei", "予測勝率(%)", "odds", "期待値"]].rename(
                    columns={"umaban": "馬番", "bamei": "馬名", "odds": "オッズ"}
                ),
                use_container_width=True
            )

# ==========================================
# TAB 3: 自動収支レポート
# ==========================================
with tab3:
    res_df = load_table("results")
    if res_df.empty:
        st.info("まだ確定した収支データがありません。")
    else:
        total_bet = int(res_df["total_bet"].sum())
        total_ret = int(res_df["total_return"].sum())
        total_prof = int(res_df["profit"].sum())
        roi = (total_ret / total_bet * 100) if total_bet > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累計投資", f"{total_bet:,} 円")
        c2.metric("累計払戻", f"{total_ret:,} 円")
        c3.metric("累計収支", f"{total_prof:+,} 円")
        c4.metric("回収率", f"{roi:.1f} %")

        st.dataframe(
            res_df.sort_values("recorded_at", ascending=False)[
                ["race_name", "winner_umaban", "tansho_payout", "total_bet", "total_return", "profit", "recorded_at"]
            ].rename(columns={
                "race_name": "レース", "winner_umaban": "1着馬番", "tansho_payout": "単勝払戻",
                "total_bet": "投資額", "total_return": "払戻額", "profit": "収支", "recorded_at": "記録時刻"
            }),
            use_container_width=True
        )
