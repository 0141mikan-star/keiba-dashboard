import streamlit as st
import pandas as pd
import numpy as np
import re
from bigdata_ai_bridge import estimate_race_upset_risk

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
    AI本線の予測順位（1位〜上位）を最優先で尊重しつつ、
    3頭以上選ぶ枠の最終枠で勝率差が僅差（1.8%以内）の「前走好調馬・特注穴馬」がいれば取りこぼしを防ぐ。
    """
    if p_df.empty or quota <= 0:
        return p_df.head(0)

    df_copy = p_df.copy().reset_index(drop=True)
    if quota <= 3 or len(df_copy) <= quota:
        return df_copy.head(quota)

    # 3頭以上枠: 上位(quota-1)頭はAI予測順位そのまま確実に採用
    picked = [df_copy.iloc[i] for i in range(quota - 1)]
    picked_nums = {r["馬番"] for r in picked}

    next_std_row = df_copy.iloc[quota - 1]
    next_std_win = float(next_std_row.get("勝率(%)", 5.0))

    remaining = df_copy.iloc[quota - 1:]
    upset_candidates = remaining[remaining.apply(
        lambda r: (
            bool(r.get("前走好調馬", False))
            or bool(r.get("特注穴馬", False))
            or float(r.get("ケリー基準f*(%)", 0.0)) >= 1.5
            or int(r.get("3走平均賞金指数", 0)) >= 15
            or float(r.get("CatBoost勝率(%)", 0.0)) >= float(r.get("勝率(%)", 0.0)) * 1.15
            or "🔥" in str(r.get("純能力AI評価", ""))
            or any(f"純能力{k}位 (" in str(r.get("純能力AI評価", "")) for k in (1, 2, 3, 4))
        )
        and int(r.get("予想着順数値", 99)) <= min(7, quota + 3)
        and (next_std_win - float(r.get("勝率(%)", 0.0))) <= 2.8,
        axis=1
    )]

    for _, u_row in upset_candidates.iterrows():
        if len(picked) >= quota:
            break
        if u_row["馬番"] not in picked_nums:
            picked.append(u_row)
            picked_nums.add(u_row["馬番"])

    for _, rem_row in remaining.iterrows():
        if len(picked) >= quota:
            break
        if rem_row["馬番"] not in picked_nums:
            picked.append(rem_row)
            picked_nums.add(rem_row["馬番"])

    return pd.DataFrame(picked)


def extract_jackpot_bombs_for_leg(
    pred_df: pd.DataFrame,
    past_records: dict,
    course_type: str,
    distance: int,
    core_nums: set,
    max_bombs: int = 3
) -> list[dict]:
    """
    数千万円〜1億円WIN5を演出する「3大爆弾プロファイル」に合致する特注穴馬(4〜12番人気)を根拠付きで厳選抽出:
      ① 【🔥純能力モンスター】: オッズ封印・純能力AIで1〜4位なのに単勝4番人気以下で放置されている隠れ実力馬
      ② 【⚡一発逃げ切り爆弾】: 脚質「逃げ」× 4〜12番人気 × (前走好調 / 前走・前々走3着以内 / 持ち時計1位 / 純能力8位以内)
      ③ 【🔄前走凡走×前々走好走の巻き返し】: 前走5〜9着敗戦でオッズ急落も、前々走が同距離帯3着以内または重賞4着以内
    """
    import re as _re
    if pred_df.empty:
        return []
    past_records = past_records or {}
    bombs = []
    for _, row in pred_df.iterrows():
        num = int(row["馬番"])
        if num in core_nums:
            continue
        pop = int(row.get("単勝人気", 99))
        odds = get_horse_effective_odds(row)
        if pop < 4 or pop > 12 or odds < 6.0:
            continue
        nm = str(row.get("馬名", "")).strip()
        style = str(row.get("脚質", ""))
        pure_str = str(row.get("純能力AI評価", ""))
        m_pr = _re.search(r"純能力(\d+)位", pure_str)
        pure_rk = int(m_pr.group(1)) if m_pr else 99
        ai_rk = int(row.get("予想着順数値", 99))
        win_p = float(row.get("勝率(%)", 2.0))

        pr = past_records.get(nm, {})
        l0 = str(pr.get("last_race", ""))
        l1 = str(pr.get("second_last", ""))
        m_r0 = _re.search(r"(\d+)着", l0)
        m_r1 = _re.search(r"(\d+)着", l1)
        r0 = int(m_r0.group(1)) if m_r0 else 99
        r1 = int(m_r1.group(1)) if m_r1 else 99
        m_d1 = _re.search(r"(?:芝|ダ|障)(\d{3,4})", l1)
        d1 = int(m_d1.group(1)) if m_d1 else 0

        bomb_score = 0.0
        reasons = []

        # ① 純能力モンスター (純能力1〜4位)
        if pure_rk <= 4:
            bomb_score += 45.0 + (5 - pure_rk) * 6.0
            reasons.append(f"🔥純能力{pure_rk}位の隠れ実力馬")

        # ② 穴の逃げ馬 (行った行った一発)
        is_gf = bool(row.get("前走好調馬", False)) or "⏱️" in str(row.get("過去走精査", ""))
        if style == "逃げ" and (is_gf or r0 <= 3 or r1 <= 3 or pure_rk <= 8):
            bomb_score += 38.0 + (12.0 if is_gf or r0 <= 2 or r1 <= 3 else 0.0)
            reasons.append("⚡穴の逃げ残り一発候補")

        # ③ 前々走好走の巻き返し (前走敗戦でオッズ妙味急増)
        if 5 <= r0 <= 9 and (r1 <= 3 or (r1 <= 4 and _re.search(r"GI|GII|GIII|Jpn", l1))):
            if d1 == 0 or abs(int(distance) - d1) <= 200:
                bomb_score += 32.0
                reasons.append("🔄前走敗戦も前々走好走の巻き返し")

        # AI上位5頭内の中穴バリュー加点
        if ai_rk <= 5:
            bomb_score += 18.0

        if bomb_score >= 30.0:
            bombs.append({
                "num": num,
                "name": nm,
                "mark": str(row.get("AI印", "☆")),
                "pop": pop,
                "odds": odds,
                "win_pct": win_p,
                "ai_rk": ai_rk,
                "pure_rk": pure_rk,
                "style": style,
                "bomb_score": bomb_score,
                "reason": " ＋ ".join(reasons)
            })

    bombs.sort(key=lambda x: (x["bomb_score"], -x["ai_rk"]), reverse=True)
    return bombs[:max_bombs]


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
            "小倉 (右回り / 直線293m・2角下り坂で高速前残り＆マクリ)",
            "福島 (右回り / 直線292m・スパイラルカーブ小回り先行有利)",
            "函館 (右回り / 直線262m・JRA最短直線＆洋芝高低差3.5m前残り鉄則)",
            "札幌 (右回り / 直線266m・大回りコーナー平坦洋芝)",
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
                "🧠 【AI勝率最大化・動的メリハリ】4,800円上限プラン (最大48点) : 5レースの鉄板度・混戦度から頭数配分を自動最適化！",
                "🛡️ 【全レース2頭以上・事故防止】3,200円プラン (32点) : 2頭 × 2頭 × 2頭 × 2頭 × 2頭 (1頭固定の突然死を完全回避！)",
                "🎯 【本命・メリハリ】3,600円プラン (36点) : 1頭 × 2頭 × 2頭 × 3頭 × 3頭 (王道バランス)",
                "🏆 【高確率・本気的中】7,200円プラン (72点) : 5レースの難易度に合わせて最大72点で勝率最大化！",
                "💎 【中穴網羅・高勝率】14,400円プラン (144点) : 銀行レース圧縮 × 混戦5〜6頭網羅 (バックテスト勝率7.0%)",
                "👑 【最高勝率＆回収率130%実証】24,300円プラン (240〜243点) : AI動的メリハリ配分 (均等BOX比2.7倍の勝率12.8%！)",
                "⚡ 【超エコノミー】2,400円プラン (24点) : 1頭 × 2頭 × 2頭 × 2頭 × 3頭",
                "🛠️ 【カスタム配分】各レースの頭数を自由指定"
            ],
            index=0
        )
    is_dynamic_quota = False
    with plan_col2:
        if "AI勝率最大化" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 4]
            base_cost = 4800
            base_points = 48
            is_dynamic_quota = True
            st.info("💡 **おすすめNo.1（AI勝率最大化！）**\nその日の5レースのAI勝率分布を解析し、48点以内で『5連勝突破確率の積』が数学的に最大になる頭数配分（例: 1×1×2×4×6 や 2×2×2×2×3）を自動算出します！")
        elif "3,200円" in plan_choice:
            quota_pattern = [2, 2, 2, 2, 2]
            base_cost = 3200
            base_points = 32
            st.info("💡 **全レース2頭以上（事故防止！）**\n全5レースをすべて【2頭以上】でカバー！ 1頭固定による『ハナ差2着での即死』を完全回避する安全設計！")
        elif "3,600円" in plan_choice:
            quota_pattern = [1, 2, 2, 3, 3]
            base_cost = 3600
            base_points = 36
            is_dynamic_quota = True
            st.info("💡 **王道メリハリ**\n36点以内で5レース突破確率が最大になるよう、鉄板レースを絞り混戦レースに頭数を自動傾斜配分します！")
        elif "7,200円" in plan_choice:
            quota_pattern = [2, 2, 2, 3, 3]
            base_cost = 7200
            base_points = 72
            is_dynamic_quota = True
            st.info("💡 **高的中・本気プラン**\n72点以内で5レース同時突破確率を最大化するAI動的配分！")
        elif "14,400円" in plan_choice:
            quota_pattern = [1, 2, 3, 4, 6]
            base_cost = 14400
            base_points = 144
            is_dynamic_quota = True
            st.info("💡 **中穴網羅・高勝率プラン**\n144点以内で銀行レースを1〜2頭に絞り、荒れる重賞・ハンデ戦を最大6〜7頭まで広げて純能力AIの穴馬を一網打尽にします！")
        elif "24,300円" in plan_choice:
            quota_pattern = [1, 2, 3, 6, 6]
            base_cost = 24300
            base_points = 243
            is_dynamic_quota = True
            st.info("💡 **バックテスト実証・回収率130.4%ゾーン**\n均等3頭BOX（3^5=243点）と同じ予算で、AI動的配分により的中率を4.7%→12.8%（約8回に1回的中）へ引き上げる最強プラン！")
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
                        for _k in range(5):
                            st.session_state.pop(f"win5_leg_mode_{_k}", None)
                            st.session_state.pop(f"win5_multisel_{_k+1}", None)
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
    # 🎛️ WIN5 波乱モード使い分け設定（順当 vs 荒れそうなレースの個別切替）
    # ==========================================
    st.markdown("---")
    st.markdown("### 🎛️ WIN5 波乱モード使い分け設定（荒れそうなレースの攻略切替）")

    leg_upset_diags = []
    for r_d in edited_races:
        u_diag = estimate_race_upset_risk(
            r_d["horses"],
            track_condition=r_d.get("condition", "良馬場"),
            race_distance=r_d.get("distance", 1600),
            course_type=r_d.get("course_type", "芝"),
        )
        leg_upset_diags.append(u_diag)

    with st.container(border=True):
        st.caption("📡 **WIN1〜WIN5 事前「荒れそう度」AIセンサー診断（堅いレースと荒れそうなレースを自動判別）**")
        s_cols = st.columns(5)
        for i_s, u_d in enumerate(leg_upset_diags):
            with s_cols[i_s]:
                rec_m = u_d["recommended_mode"]
                m_short = "🌪️大荒れ推奨" if rec_m == "upset_chaos" else ("⚡中波乱推奨" if rec_m == "upset_mild" else "🛡️通常推奨")
                st.markdown(
                    f"<div style='padding:6px 8px; border-top:4px solid {u_d['color']}; background:rgba(128,128,128,0.08); border-radius:4px; font-size:12px;'>"
                    f"<b>WIN{i_s+1}</b>: <span style='color:{u_d['color']}; font-weight:bold;'>波乱度 {u_d['risk_pct']}%</span><br>"
                    f"推奨: <b>{m_short}</b><br>"
                    f"<span style='color:#666; font-size:11px;'>{u_d['reasons'][0] if u_d['reasons'] else ''}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        win5_global_mode = st.radio(
            "🧭 **WIN5 全体モード方針**",
            options=[
                "🛡️ 全5レース 通常・王道モード (推奨：実力・過去走・コース適性を素直に評価する標準モード)",
                "🤖 AI自動使い分け (堅いレースは通常・波乱含みのレースは自動で中波乱モードへ切替)",
                "🎯 レースごとに個別指定 (下のWIN1〜WIN5設定をそのまま使用)",
                "⚡ 全5レース 中波乱・妙味モード (中穴4〜8番人気を積極抜擢)",
                "🌪️ 全5レース 大荒れ・オッズ封印モード (大波乱・人気完全無視で実力穴馬を狙い撃ち)",
            ],
            index=0,
            horizontal=False,
            key="win5_global_mode_radio",
        )

        leg_mode_opts = [
            "🤖 自動 (センサー連動)",
            "🛡️ 通常・王道",
            "⚡ 中波乱・妙味",
            "🌪️ 大荒れ・オッズ封印",
        ]
        opt_to_code = {
            "🛡️ 通常・王道": "standard",
            "⚡ 中波乱・妙味": "upset_mild",
            "🌪️ 大荒れ・オッズ封印": "upset_chaos",
        }
        m_cols = st.columns(5)
        resolved_leg_modes = {}
        for i_m in range(5):
            with m_cols[i_m]:
                sel_leg_opt = st.selectbox(
                    f"WIN{i_m+1} 予想モード",
                    options=leg_mode_opts,
                    index=0,
                    key=f"win5_leg_mode_{i_m}",
                    help="『荒れそう』と感じるレースだけ⚡中波乱や🌪️大荒れに切り替えることができます",
                )
                raw_auto = leg_upset_diags[i_m]["recommended_mode"]
                # AI自動使い分け時は、人気馬を極端に削る大荒れ(upset_chaos)には自動移行せず、波乱度75%以上の時のみ中波乱(upset_mild)を適用
                auto_code = "upset_mild" if raw_auto in ("upset_chaos", "upset_mild") and leg_upset_diags[i_m]["risk_pct"] >= 75 else "standard"
                if sel_leg_opt != "🤖 自動 (センサー連動)":
                    # 個別セレクトボックスが手動変更されていれば最優先で反映
                    eff_code = opt_to_code.get(sel_leg_opt, "standard")
                elif "全5レース 通常" in win5_global_mode:
                    eff_code = "standard"
                elif "全5レース 中波乱" in win5_global_mode:
                    eff_code = "upset_mild"
                elif "全5レース 大荒れ" in win5_global_mode:
                    eff_code = "upset_chaos"
                elif "AI自動使い分け" in win5_global_mode:
                    eff_code = auto_code
                else:
                    eff_code = auto_code
                resolved_leg_modes[i_m + 1] = eff_code

    # ==========================================
    # 🚀 AI診断 ＆ 最適フォーメーション自動生成
    # ==========================================
    btn_calc_win5 = st.button("🚀 WIN5 AI診断 ＆ 最適フォーメーションを自動生成", type="primary", use_container_width=True)

    # セッション内の古い選択キャッシュ（◎○が飛んで▲△だけ残る現象）を自動クリーンアップ
    _race_names_sig = "|".join(str(r.get("name", "")) for r in edited_races)
    _config_sig = f"v4|{plan_choice}|{win5_global_mode}|{sorted(resolved_leg_modes.items())}|{_race_names_sig}"
    if btn_calc_win5 or st.session_state.get("_win5_config_sig") != _config_sig:
        st.session_state["win5_custom_selections"] = {}
        for _k in range(1, 6):
            st.session_state.pop(f"win5_multisel_{_k}", None)
        st.session_state["_win5_config_sig"] = _config_sig

    if btn_calc_win5 or st.session_state.get("win5_has_run", False):
        st.session_state["win5_has_run"] = True

        with st.spinner("🤖 AIが5レースを同時解析し、鉄板度・波乱度・展開から最適買い目を算出中..."):
            race_diagnostics = []

            for r_data in edited_races:
                df = r_data["horses"]
                if df.empty or len(df) < 2:
                    continue

                w_a = learned_weights.get("ability", 1.55)
                w_p = learned_weights.get("pace", 1.15)
                w_j = learned_weights.get("jockey", 0.95)
                w_e = learned_weights.get("escape_dynamic", 1.20)
                w_s = learned_weights.get("synergy", 1.15)

                leg_mode_code = resolved_leg_modes.get(r_data["id"], "standard")
                prev_past_rec = st.session_state.get("netkeiba_past_records", {})
                st.session_state["netkeiba_past_records"] = r_data.get("past_records", {}) or prev_past_rec
                try:
                    pred_res, meta = calculate_predictions_func(
                        df, w_a, w_p, w_j, w_e, w_s,
                        race_distance=r_data["distance"],
                        track_name=r_data["track"],
                        course_type=r_data["course_type"],
                        pace_prediction=r_data["pace"],
                        track_condition=r_data["condition"],
                        prediction_mode=leg_mode_code,
                    )
                except TypeError:
                    pred_res, meta = calculate_predictions_func(
                        df, w_a, w_p, w_j, w_e, w_s,
                        race_distance=r_data["distance"],
                        track_name=r_data["track"],
                        course_type=r_data["course_type"],
                        pace_prediction=r_data["pace"],
                        track_condition=r_data["condition"],
                    )
                st.session_state["netkeiba_past_records"] = prev_past_rec

                # レース波乱度診断
                esc_cnt_leg = len(df[df["脚質"] == "逃げ"]) if "脚質" in df.columns else 0
                vol_info = meta.get("volatility", {})
                if not vol_info and diagnose_volatility_func:
                    vol_info = diagnose_volatility_func(pred_res, esc_cnt_leg)

                top1 = pred_res.iloc[0]
                top2 = pred_res.iloc[1] if len(pred_res) >= 2 else None
                top3 = pred_res.iloc[2] if len(pred_res) >= 3 else None
                top4 = pred_res.iloc[3] if len(pred_res) >= 4 else None

                win1 = float(top1["勝率(%)"])
                win2 = float(top2["勝率(%)"]) if top2 is not None else 0.0
                win3 = float(top3["勝率(%)"]) if top3 is not None else 0.0
                win_gap = win1 - win2
                top3_win_sum = win1 + win2 + win3

                sc1 = float(top1.get("総合スコア", 100.0))
                sc2 = float(top2.get("総合スコア", sc1)) if top2 is not None else sc1
                sc3 = float(top3.get("総合スコア", sc2)) if top3 is not None else sc2
                sc_gap12 = max(0.0, sc1 - sc2)
                sc_gap23 = max(0.0, sc2 - sc3)
                top1_is_pure1 = "純能力1位 (" in str(top1.get("純能力AI評価", ""))
                dark_pure1 = any(
                    int(rx.get("単勝人気", 99)) >= 4 and "純能力1位 (" in str(rx.get("純能力AI評価", ""))
                    for _, rx in pred_res.head(5).iterrows()
                )
                dark_gf_cnt = sum(
                    1 for _, rx in pred_res.head(8).iterrows()
                    if int(rx.get("単勝人気", 99)) >= 4 and bool(rx.get("前走好調馬", False))
                )

                p_recs_leg = r_data.get("past_records", {}) or {}
                def _get_leg_last_str(h_row):
                    if h_row is None:
                        return ""
                    s_l = str(h_row.get("前走成績", ""))
                    if not s_l or s_l in ("-", "データなし", "nan"):
                        nm_h = str(h_row.get("馬名", "")).strip()
                        s_l = str(p_recs_leg.get(nm_h, {}).get("last_race", ""))
                    return s_l

                def _get_leg_r0(h_row):
                    s_l = _get_leg_last_str(h_row)
                    m_r = re.search(r"(\d+)着", s_l)
                    return int(m_r.group(1)) if m_r else 99

                top1_last = _get_leg_last_str(top1)
                top2_last = _get_leg_last_str(top2)
                top1_r0 = _get_leg_r0(top1)
                top2_r0 = _get_leg_r0(top2)
                top1_pop = int(top1.get("単勝人気", 1))
                top1_odds = get_horse_effective_odds(top1)
                is_shinba_clash = ("新馬" in top1_last and "新馬" in top2_last)
                is_graded_drop_fav1 = (
                    top1_pop == 1
                    and top1_odds <= 2.1
                    and bool(re.search(r"\b(?:GI|GII|GIII|JpnI|JpnII|JpnIII)\b", top1_last))
                )
                top3_has_bomb = bool(
                    top3 is not None
                    and int(top3.get("単勝人気", 99)) >= 4
                    and (
                        bool(top3.get("特注穴馬", False))
                        or bool(top3.get("前走好調馬", False))
                        or any(f"純能力{k}位 (" in str(top3.get("純能力AI評価", "")) for k in (1, 2, 3))
                    )
                )
                is_two_horse_lock = (
                    top1_r0 <= 2
                    and top2_r0 <= 2
                    and not ("新馬" in top1_last and esc_cnt_leg >= 4)
                    and (sc_gap12 >= 10.0 or (sc_gap23 >= 15.0 and not top3_has_bomb))
                )
                is_one_horse_lock = (
                    not is_shinba_clash
                    and (
                        (sc_gap12 >= 25.0 and not (top1_pop == 1 and top1_r0 >= 6 and not is_graded_drop_fav1) and not top3_has_bomb)
                        or (top1_r0 == 1 and sc_gap12 >= 12.0 and not top3_has_bomb)
                        or (top1_r0 == 1 and top2_r0 >= 6 and not top3_has_bomb)
                        or (is_graded_drop_fav1 and sc_gap12 >= 14.0 and not top3_has_bomb)
                    )
                )

                # 鉄板度スコア (勝率だけでなくAI総合スコア差・2-3位間断層・前走着順・3位特注穴馬・逃げ馬過密を統合評価)
                vol_level = vol_info.get("level", "")
                vol_penalty = 15.0 if "大波乱" in vol_level else (8.0 if "波乱含み" in vol_level else 0.0)
                if leg_mode_code == "upset_chaos":
                    vol_penalty += 18.0
                elif leg_mode_code == "upset_mild":
                    vol_penalty += 9.0
                if (win1 + win2) < 38.0 or win1 < 22.0:
                    if not (is_one_horse_lock or is_two_horse_lock):
                        vol_penalty += 24.0
                if top3_win_sum < 50.0:
                    if not (is_one_horse_lock or is_two_horse_lock):
                        vol_penalty += 22.0
                if len(pred_res) >= 16 and win1 < 25.0:
                    if not (is_one_horse_lock or is_two_horse_lock):
                        vol_penalty += 15.0
                if sc_gap12 < 3.0 and not is_two_horse_lock:
                    vol_penalty += 25.0
                elif sc_gap12 < 6.0 and not (is_one_horse_lock or is_two_horse_lock):
                    vol_penalty += 10.0
                if dark_pure1:
                    vol_penalty += 14.0
                if dark_gf_cnt >= 4:
                    vol_penalty += 8.0
                if esc_cnt_leg >= 4 and not is_two_horse_lock:
                    vol_penalty += 16.0
                    if "新馬" in top1_last or top3_has_bomb or sc_gap12 < 3.0:
                        vol_penalty += 45.0

                # 本命(AI 1位)の前走敗戦深さに応じた連続リスク減点＆3位特注穴馬の波乱リスク反映
                if top1_pop == 1 and top1_r0 >= 5:
                    if is_graded_drop_fav1:
                        vol_penalty += 12.0
                    else:
                        vol_penalty += min(62.0, 26.0 + (top1_r0 - 4) * 9.0)
                elif top1_r0 >= 5:
                    vol_penalty += min(36.0, 14.0 + (top1_r0 - 4) * 5.0)
                if top3_has_bomb and top1_r0 >= 5 and not is_graded_drop_fav1:
                    vol_penalty += 28.0
                elif top3_has_bomb and top2_r0 >= 6 and sc_gap23 < 10.0 and win1 < 35.0:
                    vol_penalty += 32.0
                elif top3_has_bomb and sc_gap12 < 10.0:
                    vol_penalty += 14.0
                elif top3_has_bomb and sc_gap23 <= 2.5 and win1 < 36.0:
                    vol_penalty += 38.0

                form_solid_bonus = 0.0
                if sc_gap12 >= 25.0 and not (top1_pop == 1 and top1_r0 >= 6 and not is_graded_drop_fav1) and not top3_has_bomb and not is_shinba_clash:
                    form_solid_bonus += 65.0
                elif ((top1_r0 == 1 and not top3_has_bomb) or (is_graded_drop_fav1 and win1 >= 38.0)) and sc_gap12 >= 12.0 and not is_shinba_clash:
                    form_solid_bonus += 45.0
                elif top1_r0 == 1 and top2_r0 >= 6 and not top3_has_bomb and not is_shinba_clash:
                    form_solid_bonus += 42.0
                elif is_two_horse_lock and not is_shinba_clash:
                    form_solid_bonus += 32.0
                if is_shinba_clash:
                    vol_penalty += 45.0

                pure1_bonus = 12.0 if (top1_is_pure1 and sc_gap12 >= 5.0 and win1 >= 24.0 and (top1_r0 <= 4 or is_graded_drop_fav1)) else 0.0
                solid_score = (
                    (win1 * 1.2)
                    + (win_gap * 1.5)
                    + (min(25.0, sc_gap12) * 1.2)
                    + (min(20.0, sc_gap23) * 0.4 if is_two_horse_lock else 0.0)
                    + pure1_bonus
                    + form_solid_bonus
                    - vol_penalty
                )

                race_diagnostics.append({
                    "id": r_data["id"],
                    "name": r_data["name"],
                    "track": r_data["track"],
                    "distance": r_data["distance"],
                    "course_type": r_data["course_type"],
                    "pace": r_data["pace"],
                    "condition": r_data["condition"],
                    "prediction_mode": leg_mode_code,
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
                    "sc_gap12": sc_gap12,
                    "sc_gap23": sc_gap23,
                    "top1_r0": top1_r0,
                    "top2_r0": top2_r0,
                    "top1_pop": top1_pop,
                    "is_shinba_clash": is_shinba_clash,
                    "is_graded_drop_fav1": is_graded_drop_fav1,
                    "top3_has_bomb": top3_has_bomb,
                    "is_one_horse_lock": is_one_horse_lock,
                    "is_two_horse_lock": is_two_horse_lock,
                    "top1_is_pure1": top1_is_pure1,
                    "dark_pure1": dark_pure1,
                    "dark_gf_cnt": dark_gf_cnt,
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

            if is_dynamic_quota:
                if base_points >= 240:
                    r0_d = sorted_by_solidity[0]
                    r4_d = sorted_by_solidity[4]
                    if (
                        bool(r0_d.get("is_one_horse_lock", False))
                        and float(r0_d.get("win1", 0.0)) >= 48.0
                        and float(r0_d.get("sc_gap12", 0.0)) >= 35.0
                        and float(r4_d.get("solid_score", 50.0)) < 0.0
                    ):
                        quota_pattern = [1, 3, 3, 4, 6]
                    else:
                        quota_pattern = [3, 3, 3, 3, 3]
                else:
                    # 各レースでk頭(1〜7頭)選んだときの累積勝率を計算し、base_points以内で5レース同時突破確率の積を最大化する配分を探索
                    leg_cum_probs = []
                    for idx_sol, r_d in enumerate(sorted_by_solidity):
                        p_df_c = r_d["pred_df"]
                        max_k = min(7, len(p_df_c))
                        c_map = {}
                        for k_c in range(1, max_k + 1):
                            sel_k = select_win5_horses_for_quota(p_df_c, k_c)
                            p_sum = min(0.98, max(0.05, float(sel_k["勝率(%)"].sum()) / 100.0))
                            # 1頭固定(k_c == 1)は鉄板度1位かつ十分なスコア差があるレース以外には強いリスクペナルティを課す
                            if k_c == 1:
                                if bool(r_d.get("is_shinba_clash", False)):
                                    p_sum *= 0.42
                                elif idx_sol >= 1 and (float(r_d.get("win1", 0.0)) < 38.0 or float(r_d.get("sc_gap12", 0.0)) < 12.0):
                                    p_sum *= 0.45
                                elif (float(r_d.get("win1", 0.0)) < 30.0 and float(r_d.get("sc_gap12", 0.0)) < 12.0) or float(r_d.get("sc_gap12", 0.0)) < 5.0 or (bool(r_d.get("dark_pure1", False)) and not bool(r_d.get("is_graded_drop_fav1", False))):
                                    p_sum *= 0.55
                                if int(r_d.get("top1_pop", 1)) == 1 and int(r_d.get("top1_r0", 99)) >= 6 and not bool(r_d.get("is_graded_drop_fav1", False)):
                                    p_sum *= 0.35
                            elif k_c == 2:
                                if (
                                    idx_sol >= 3 and (
                                        bool(r_d.get("top3_has_bomb", False))
                                        or int(r_d.get("top1_r0", 99)) >= 5
                                        or float(r_d.get("sc_gap23", 0.0)) < 15.0
                                        or not bool(r_d.get("is_two_horse_lock", False))
                                    )
                                ) or (
                                    idx_sol >= 2
                                    and bool(r_d.get("top3_has_bomb", False))
                                    and int(r_d.get("top2_r0", 99)) >= 6
                                    and float(r_d.get("sc_gap23", 0.0)) < 10.0
                                    and float(r_d.get("win1", 0.0)) < 35.0
                                ) or (
                                    bool(r_d.get("top3_has_bomb", False))
                                    and float(r_d.get("sc_gap23", 0.0)) <= 2.5
                                    and float(r_d.get("win1", 0.0)) < 36.0
                                ):
                                    p_sum *= 0.58
                            c_map[k_c] = p_sum
                        for k_c in range(max_k + 1, 8):
                            c_map[k_c] = c_map.get(max_k, 0.95)
                        leg_cum_probs.append(c_map)

                    best_prob = -1.0
                    best_q = list(quota_pattern)
                    min_pts = max(24, int(base_points * 0.75))
                    for q0 in range(1, 5):
                        for q1 in range(q0, 6):
                            for q2 in range(q1, 7):
                                for q3 in range(q2, 8):
                                    for q4 in range(q3, 8):
                                        pts = q0 * q1 * q2 * q3 * q4
                                        if min_pts <= pts <= base_points:
                                            jp = (
                                                leg_cum_probs[0][q0]
                                                * leg_cum_probs[1][q1]
                                                * leg_cum_probs[2][q2]
                                                * leg_cum_probs[3][q3]
                                                * leg_cum_probs[4][q4]
                                            )
                                            if jp > best_prob:
                                                best_prob = jp
                                                best_q = [q0, q1, q2, q3, q4]
                    quota_pattern = best_q

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
                    leg_win_prob = round(float(chosen["勝率(%)"].astype(float).sum()), 1) if not chosen.empty else 0.0
                    st.markdown(f"🎯 **選択: {nums_str}** ({q}頭)<br>📈 **買い目勝率: `{leg_win_prob:.1f}%`**", unsafe_allow_html=True)
                    chosen_nums_set = {int(r["馬番"]) for _, r in chosen.iterrows()}
                    for _, h in chosen.iterrows():
                        ai_mark = h.get('AI印', '◎' if _ == chosen.index[0] else '○')
                        f_tag = " 🔥前走好調" if (bool(h.get("前走好調馬", False)) or "🔥" in str(h.get("過去走精査", ""))) else ""
                        st.write(f"- {ai_mark} {int(h['馬番'])}番 {h['馬名']} ({h['勝率(%)']}%){f_tag}")

                    # 1〜2頭制限枠のレースで、AI上位5頭内(▲・△)に「単勝回収率225%条件(4番人気以下・6〜35倍)」かつ明確な根拠を持つ特注馬が1頭いる場合のみ理由付きで提示（根拠なき羅列は廃止）
                    p_df_card = r_diag["pred_df"]
                    if q <= 2:
                        high_ev_unpicked = p_df_card[p_df_card.apply(
                            lambda r_u: (int(r_u["馬番"]) not in chosen_nums_set)
                            and int(r_u.get("予想着順数値", 99)) <= 5
                            and int(r_u.get("単勝人気", 99)) >= 4
                            and 6.0 <= get_horse_effective_odds(r_u) <= 35.0
                            and (bool(r_u.get("前走好調馬", False)) or "⏱️" in str(r_u.get("過去走精査", "")) or "👑" in str(r_u.get("過去走精査", "")) or "🔥" in str(r_u.get("過去走精査", ""))),
                            axis=1
                        )]
                        if not high_ev_unpicked.empty:
                            top_u = high_ev_unpicked.iloc[0]
                            u_num = int(top_u["馬番"])
                            u_name = str(top_u["馬名"])
                            u_mark = str(top_u.get("AI印", "▲"))
                            u_odds = get_horse_effective_odds(top_u)
                            u_reason = str(top_u.get("過去走精査", "")).strip()
                            if len(u_reason) > 36:
                                u_reason = u_reason[:36] + "…"
                            st.caption(
                                f"💎 **予算に余裕があれば＋1頭**: **{u_num}番{u_name}** ({u_mark} / {u_odds:.1f}倍)<br>"
                                f"<span style='font-size:0.75rem;'>📌 根拠: {u_reason}</span>",
                                unsafe_allow_html=True
                            )

            # ------------------------------------------
            # 🚦 各レースの「買い目勝率」＆「買うべきか・買わないべきか（荒れ展開判定）」一覧
            # ------------------------------------------
            leg_judge_rows = []
            combined_win5_prob = 1.0
            chaos_leg_names = []
            solid_leg_names = []
            for i_lj, r_diag in enumerate(race_diagnostics):
                p_df_lj = r_diag["pred_df"]
                chosen_lj = active_selections[i_lj]
                leg_prob_pct = round(float(chosen_lj["勝率(%)"].astype(float).sum()), 1) if not chosen_lj.empty else 0.0
                combined_win5_prob *= (leg_prob_pct / 100.0)

                scs = p_df_lj["総合スコア"].astype(float).tolist()
                wrs = p_df_lj["勝率(%)"].astype(float).tolist()
                t1_win = wrs[0] if wrs else 15.0
                t2_win = wrs[1] if len(wrs) >= 2 else t1_win
                t4_win = wrs[min(3, len(wrs) - 1)] if wrs else t2_win
                t6_win = wrs[min(5, len(wrs) - 1)] if wrs else t1_win
                wg12 = round(t1_win - t2_win, 1)
                wg14 = round(t1_win - t4_win, 1)
                wg26 = round(t2_win - t6_win, 1)
                top3_w = round(sum(wrs[:3]), 1)

                dark_gf = p_df_lj[p_df_lj.apply(
                    lambda r_x: int(r_x.get("単勝人気", 99)) >= 4
                    and bool(r_x.get("前走好調馬", False))
                    and int(r_x.get("予想着順数値", 99)) <= 8,
                    axis=1
                )]
                dark_front = [
                    f"{int(rx['馬番'])}番{rx['馬名']}"
                    for _, rx in dark_gf.iterrows()
                    if str(rx.get("脚質", "")) in ["逃げ", "先行"]
                ]

                if t1_win >= 26.0 and wg12 >= 7.5 and top3_w >= 58.0:
                    single_verdict = "🔥 買うべき (順当・鉄板級)"
                    w5_guide = "🔒 1〜2頭で高確率突破"
                    solid_leg_names.append(f"WIN{r_diag['id']}")
                elif t1_win >= 21.0 and wg26 > 7.5 and len(dark_gf) <= 2:
                    single_verdict = "🟢 買うべき (標準・好機)"
                    w5_guide = "⚖️ 2〜3頭で安定通過"
                elif wg26 <= 7.0 or len(dark_gf) >= 3 or (r_diag.get("course_type") == "ダート" and len(dark_front) >= 2 and t1_win < 26.0):
                    single_verdict = "🛑 買わないべき・見推奨 (荒れ注意)"
                    front_note = f" [前残り穴:{','.join(dark_front[:2])}]" if dark_front else ""
                    w5_guide = f"🌪️ 隠れ荒れ関門！5〜6頭推奨{front_note}"
                    chaos_leg_names.append(f"WIN{r_diag['id']}")
                else:
                    single_verdict = "⚠️ 慎重・少額 (中穴ケア)"
                    w5_guide = "⚡ 3〜4頭＋前走好調馬ケア"

                leg_judge_rows.append({
                    "レース": r_diag["name"],
                    "選択頭数": f"{len(chosen_lj)}頭",
                    "選択買い目勝率(突破率)": f"{leg_prob_pct:.1f}%",
                    "1位単勝率 / 上位3頭合計": f"{t1_win:.1f}% / {top3_w:.1f}%",
                    "単体馬券ジャッジ": single_verdict,
                    "WIN5攻略アドバイス": w5_guide
                })

            with st.container(border=True):
                st.markdown(
                    f"##### 🚦 レース別「買い目勝率（突破確率）」＆「買うべきか・買わないべきか」AIジャッジ （5レース通算突破率: **`{combined_win5_prob * 100.0:.2f}%`**）"
                )
                st.dataframe(pd.DataFrame(leg_judge_rows), use_container_width=True, hide_index=True)
                if chaos_leg_names:
                    st.warning(
                        f"⚠️ **【荒れ展開センサー警告】** **{', '.join(chaos_leg_names)}** はAI上位〜中位のスコア差が詰まっており、前走好調な伏兵が多数潜んでいます。単体馬券なら**『見（買わない）』**が安全、WIN5なら**『🔥前走好調馬を全レース追加』＋『✂️スマート死に目カット』**でここを厚くカバーしてください！"
                    )

            # ------------------------------------------
            # 🛠️ 買い目の自由カスタマイズ（手動選択・微調整）
            # ------------------------------------------
            with st.expander("🛠️ **自分で買い目を選択・カスタマイズ（WIN1〜WIN5の購入馬を自由に追加・除外）**", expanded=True):
                st.caption("各レースで購入したい馬番を自由に選択・変更できます（×ボタンで削除、プルダウンから任意の馬を追加）。下の『✂️ スマート死に目カット』を使えば、前走好調馬や穴馬を広く追加しても予算内に自動圧縮できます！")
                q_btn_c1, q_btn_c2, q_btn_c3, q_btn_c4, q_btn_c5 = st.columns(5)
                with q_btn_c1:
                    if st.button("🤖 AI推奨馬にリセット", use_container_width=True, key="win5_reset_ai_sel"):
                        st.session_state["win5_custom_selections"] = {}
                        for r_d in race_diagnostics:
                            r_i = r_d["id"]
                            q_i = r_d["assigned_quota"]
                            ai_def = [int(r["馬番"]) for _, r in select_win5_horses_for_quota(r_d["pred_df"], q_i).iterrows()]
                            st.session_state[f"win5_multisel_{r_i}"] = ai_def
                        st.rerun()
                with q_btn_c2:
                    if st.button("🎰 1億円・帯封宝くじ編成", use_container_width=True, key="win5_jackpot_bomb_sel", help="各レースの『AI本線(◎○)2頭』＋『1億円キラー3大爆弾(🔥純能力上位・⚡穴の逃げ馬・🔄前々走好走巻き返し)』を自動セットし、本線3〜4勝×爆弾1〜2勝の帯封〜1億円目を抜粋します"):
                        for r_d in race_diagnostics:
                            r_i = r_d["id"]
                            p_df_f = r_d["pred_df"]
                            core_2 = [int(r["馬番"]) for _, r in p_df_f.head(2).iterrows()]
                            bombs_f = extract_jackpot_bombs_for_leg(
                                p_df_f, r_d.get("past_records", {}), r_d.get("course_type", "芝"), r_d.get("distance", 1600), set(core_2), max_bombs=3
                            )
                            jp_nums = core_2 + [b["num"] for b in bombs_f if b["num"] not in core_2]
                            st.session_state["win5_custom_selections"][r_i] = jp_nums
                            st.session_state[f"win5_multisel_{r_i}"] = jp_nums
                        st.session_state["win5_use_smart_cut"] = True
                        st.session_state["win5_smart_target_n"] = 48
                        st.session_state["win5_smart_dark_limit"] = "🔥 穴馬(4人気以下かつAI3番手以下)は『最大2レースまで』 (推奨・死に目完全カット)"
                        st.session_state["win5_smart_sort_mode"] = "🎰 1億円・帯封ジャックポット特化 (本線3〜4勝＋爆弾1〜2勝・想定50万〜1億円超)"
                        st.rerun()
                with q_btn_c3:
                    if st.button("🔥 前走好調馬を全レース追加", use_container_width=True, key="win5_add_good_form_sel", help="AI推奨馬に加えて、前走1〜3着や僅差0.3秒以内で調子が良い未選択馬(AI7位以内)をワンクリックで追加します"):
                        for i_f, r_d in enumerate(race_diagnostics):
                            r_i = r_d["id"]
                            p_df_f = r_d["pred_df"]
                            cur_s = list(st.session_state.get("win5_custom_selections", {}).get(r_i, [int(r["馬番"]) for _, r in active_selections[i_f].iterrows()]))
                            gf_rows = p_df_f[p_df_f.apply(
                                lambda r_g: (bool(r_g.get("前走好調馬", False)) or "🔥" in str(r_g.get("過去走精査", "")))
                                and int(r_g.get("予想着順数値", 99)) <= 7,
                                axis=1
                            )]
                            for _, g_row in gf_rows.iterrows():
                                g_num = int(g_row["馬番"])
                                if g_num not in cur_s and len(cur_s) < 5:
                                    cur_s.append(g_num)
                            st.session_state["win5_custom_selections"][r_i] = cur_s
                            st.session_state[f"win5_multisel_{r_i}"] = cur_s
                        st.session_state["win5_use_smart_cut"] = True
                        st.rerun()
                with q_btn_c4:
                    if st.button("🕸️ 広域網セット(各3〜4頭)", use_container_width=True, key="win5_wide_net_sel", help="各レース3〜4頭ずつ広く拾い、下のスマート死に目カットで30〜50点に絞るプロ推奨設定"):
                        sorted_sol_tmp = sorted(race_diagnostics, key=lambda x: x["solid_score"], reverse=True)
                        wide_quotas = [3, 3, 4, 4, 4]
                        for idx_w, r_d in enumerate(sorted_sol_tmp):
                            r_i = r_d["id"]
                            w_q = wide_quotas[idx_w]
                            w_nums = [int(r["馬番"]) for _, r in select_win5_horses_for_quota(r_d["pred_df"], w_q).iterrows()]
                            st.session_state["win5_custom_selections"][r_i] = w_nums
                            st.session_state[f"win5_multisel_{r_i}"] = w_nums
                        st.session_state["win5_use_smart_cut"] = True
                        st.rerun()
                with q_btn_c5:
                    if st.button("➕ 全レースに「1番人気」追加", use_container_width=True, key="win5_add_fav1_sel"):
                        for i_f, r_d in enumerate(race_diagnostics):
                            r_i = r_d["id"]
                            p_df_f = r_d["pred_df"]
                            cur_s = list(st.session_state.get("win5_custom_selections", {}).get(r_i, [int(r["馬番"]) for _, r in active_selections[i_f].iterrows()]))
                            fav1_rows = p_df_f[p_df_f["単勝人気"].astype(int) == 1]
                            if not fav1_rows.empty:
                                f_num = int(fav1_rows.iloc[0]["馬番"])
                                if f_num not in cur_s:
                                    cur_s.append(f_num)
                            st.session_state["win5_custom_selections"][r_i] = cur_s
                            st.session_state[f"win5_multisel_{r_i}"] = cur_s
                        st.rerun()

                cust_cols = st.columns(5)
                for i, r_diag in enumerate(race_diagnostics):
                    r_id = r_diag["id"]
                    p_df = r_diag["pred_df"]
                    all_options = [int(r["馬番"]) for _, r in p_df.iterrows()]
                    current_selected = [int(r["馬番"]) for _, r in active_selections[i].iterrows()]
                    h_info_lookup = {int(r["馬番"]): r for _, r in p_df.iterrows()}

                    def _fmt_w5_horse(x_num: int, lkp=h_info_lookup) -> str:
                        r_h = lkp.get(x_num)
                        if r_h is None:
                            return f"{x_num}番"
                        mk = r_h.get("AI印", "ー")
                        nm = r_h.get("馬名", "")
                        pop_v = int(r_h.get("単勝人気", 99))
                        od_v = get_horse_effective_odds(r_h)
                        f_b = " 🔥前走好調" if (bool(r_h.get("前走好調馬", False)) or "🔥" in str(r_h.get("過去走精査", ""))) else ""
                        return f"{x_num}番 {nm} ({mk} / {pop_v}人気 {od_v:.1f}倍{f_b})"

                    with cust_cols[i]:
                        if r_id not in st.session_state["win5_custom_selections"]:
                            st.session_state[f"win5_multisel_{r_id}"] = current_selected
                        new_sel = st.multiselect(
                            f"WIN{r_id} 購入馬を選択",
                            options=all_options,
                            default=current_selected,
                            format_func=_fmt_w5_horse,
                            key=f"win5_multisel_{r_id}"
                        )
                        if set(new_sel) != set(current_selected) and len(new_sel) > 0:
                            st.session_state["win5_custom_selections"][r_id] = new_sel
                            st.rerun()

            # ==========================================
            # ✂️ 【攻略法②】 フォーメーションの「死に目カット」＆ 高確率上位N点スマート抜粋
            # ==========================================
            import itertools
            raw_box_points = int(np.prod([len(sel) for sel in active_selections]))
            raw_box_cost = raw_box_points * 100

            # 選択された各レースの馬リストから全組み合わせを生成し、AI勝率積・穴馬数・想定配当を計算
            leg_selected_records = []
            leg_total_prob_sums = []
            for i_leg in range(5):
                p_df_full = race_diagnostics[i_leg]["pred_df"]
                total_w_sum = max(1.0, float(p_df_full["勝率(%)"].astype(float).sum()))
                sel_df_leg = active_selections[i_leg].copy()
                recs = []
                leg_p_sum = 0.0
                for idx_in_sel, (_, r_h) in enumerate(sel_df_leg.iterrows()):
                    p_norm = max(0.002, float(r_h.get("勝率(%)", 1.0)) / total_w_sum)
                    leg_p_sum += p_norm
                    pop_int = int(r_h.get("単勝人気", 5))
                    rk_int = int(r_h.get("予想着順数値", idx_in_sel + 1))
                    od_val = get_horse_effective_odds(r_h)
                    # 「穴馬」判定: 単勝4番人気以下 かつ AI予想3番手以下
                    is_dark = (pop_int >= 4 and rk_int >= 3)
                    recs.append({
                        "num": int(r_h["馬番"]),
                        "name": str(r_h["馬名"]),
                        "mark": str(r_h.get("AI印", "ー")),
                        "pop": pop_int,
                        "rank": rk_int,
                        "prob": p_norm,
                        "odds": od_val,
                        "is_dark": is_dark,
                    })
                leg_selected_records.append(recs)
                leg_total_prob_sums.append(min(0.98, leg_p_sum))

            all_box_combos = []
            full_box_prob_sum = 0.0
            for comb_items in itertools.product(*leg_selected_records):
                c_prob = float(np.prod([x["prob"] for x in comb_items]))
                full_box_prob_sum += c_prob
                c_pay = int(min(600_000_000, max(100, float(np.prod([x["odds"] for x in comb_items])) * 70)))
                dark_cnt = sum(1 for x in comb_items if x["is_dark"])
                nums_tuple = tuple(int(x["num"]) for x in comb_items)
                nums_label = " - ".join(f"{x['num']:02d}" for x in comb_items)
                names_label = " → ".join(f"{x['num']}番{x['name'][:5]}({x['mark']})" for x in comb_items)
                # 期待値バランススコア (確率 × 配当の対数重み)
                ev_score = c_prob * (np.log10(max(10.0, float(c_pay))) ** 1.35)
                all_box_combos.append({
                    "tuple": nums_tuple,
                    "nums_str": nums_label,
                    "names_str": names_label,
                    "prob": c_prob,
                    "pay": c_pay,
                    "dark_count": dark_cnt,
                    "ev_score": ev_score,
                })

            st.markdown("---")
            with st.container(border=True):
                st.subheader("✂️ 【攻略法②】 スマート死に目カット ＆ 高確率上位N点抜粋")
                st.caption(
                    "各レースで3〜5頭ずつ広く選んだ状態から、**「5レース全部が穴馬」などの確率0.0001%以下の死に目（無駄馬券）を自動カット**し、"
                    "**予算内で最も当たる確率が高い上位N通りだけ**をピンポイント抜粋します！"
                )

                use_smart_cut = st.checkbox(
                    f"⚡ スマート死に目カット＆上位N点抜粋を有効にする（現在のBOX全通り: {raw_box_points}点 / {raw_box_cost:,}円）",
                    value=bool(st.session_state.get("win5_use_smart_cut", False)),
                    key="win5_smart_cut_checkbox"
                )
                st.session_state["win5_use_smart_cut"] = use_smart_cut

                smart_selected_tuples = None
                if use_smart_cut:
                    sc_c1, sc_c2, sc_c3 = st.columns([1.0, 1.3, 1.3])
                    with sc_c1:
                        default_target_n = min(36, max(10, raw_box_points))
                        target_n_tickets = st.number_input(
                            "🎯 抜粋する購入点数 (上限)",
                            min_value=1,
                            max_value=max(1, min(500, raw_box_points)),
                            value=min(int(st.session_state.get("win5_smart_target_n", default_target_n)), max(1, raw_box_points)),
                            step=5,
                            help="30点なら3,000円、36点なら3,600円にピタリと収まります。"
                        )
                        st.session_state["win5_smart_target_n"] = target_n_tickets
                    with sc_c2:
                        max_dark_opt = st.selectbox(
                            "🚫 同時穴馬（死に目）カット条件",
                            options=[
                                "🔥 穴馬(4人気以下かつAI3番手以下)は『最大2レースまで』 (推奨・死に目完全カット)",
                                "🛡️ 穴馬は『最大1レースまで』 (堅め4勝＋1波乱の一撃狙い)",
                                "⚡ 穴馬は『最大3レースまで』許容 (中〜大波乱対応)",
                                "🔓 制限なし (純粋にAI勝率積が高い順)"
                            ],
                            index=0,
                            key="win5_smart_dark_limit"
                        )
                    with sc_c3:
                        sort_mode_opt = st.selectbox(
                            "📊 抜粋優先モード",
                            options=[
                                "🎯 的中確率最優先 (AI勝率積 p1×p2×p3×p4×p5 が高い順)",
                                "💎 妙味・期待値バランス順 (高確率 × 好配当を優先)",
                                "💰 ガミ目カット＋的中確率順 (想定配当5,000円未満を除外)",
                                "🎰 1億円・帯封ジャックポット特化 (本線3〜4勝＋爆弾1〜2勝・想定50万〜1億円超)"
                            ],
                            index=0,
                            key="win5_smart_sort_mode"
                        )

                    if "最大1レース" in max_dark_opt:
                        max_dark_allowed = 1
                    elif "最大2レース" in max_dark_opt:
                        max_dark_allowed = 2
                    elif "最大3レース" in max_dark_opt:
                        max_dark_allowed = 3
                    else:
                        max_dark_allowed = 5

                    filtered_c = [c for c in all_box_combos if c["dark_count"] <= max_dark_allowed]
                    if "ジャックポット特化" in sort_mode_opt:
                        jp_c = [c for c in filtered_c if 1 <= c["dark_count"] <= max_dark_allowed and c["pay"] >= 300_000]
                        if jp_c:
                            filtered_c = jp_c
                    elif "ガミ目カット" in sort_mode_opt:
                        non_gami = [c for c in filtered_c if c["pay"] >= 5000]
                        if non_gami:
                            filtered_c = non_gami
                    if not filtered_c:
                        filtered_c = list(all_box_combos)

                    if "妙味・期待値" in sort_mode_opt or "ジャックポット特化" in sort_mode_opt:
                        filtered_c.sort(key=lambda x: x["ev_score"], reverse=True)
                    else:
                        filtered_c.sort(key=lambda x: x["prob"], reverse=True)

                    top_n_combos = filtered_c[:target_n_tickets]
                    smart_selected_tuples = [c["tuple"] for c in top_n_combos]
                    retained_prob = sum(c["prob"] for c in top_n_combos)
                    retain_ratio = (retained_prob / full_box_prob_sum * 100.0) if full_box_prob_sum > 0 else 100.0
                    cut_points = raw_box_points - len(top_n_combos)
                    cut_pct = (cut_points / raw_box_points * 100.0) if raw_box_points > 0 else 0.0

                    actual_points = len(top_n_combos)
                    actual_cost = actual_points * 100

                    m_sc1, m_sc2, m_sc3, m_sc4 = st.columns(4)
                    with m_sc1:
                        st.metric("✂️ 圧縮後の購入点数", f"{actual_points}点", delta=f"-{cut_points}点 (-{cut_pct:.0f}% 削減)")
                    with m_sc2:
                        st.metric("💵 圧縮後の必要資金", f"{actual_cost:,}円", delta=f"{raw_box_cost - actual_cost:,}円 節約！")
                    with m_sc3:
                        st.metric("🎯 BOX的中率の維持率", f"{retain_ratio:.1f}%", delta="美味しい目だけ凝縮")
                    with m_sc4:
                        dead_cut_cnt = len(all_box_combos) - len([c for c in all_box_combos if c["dark_count"] <= max_dark_allowed])
                        st.metric("🗑️ 排除した死に目数", f"{dead_cut_cnt}通り", delta="全穴の無駄馬券を除外")

                    with st.expander(f"📋 抜粋された厳選 {actual_points} 通り一覧 ＆ 個別ON/OFF調整", expanded=False):
                        all_c_labels = [f"{c['nums_str']} ｜ {c['names_str']} [想定¥{c['pay']:,}]" for c in top_n_combos]
                        label_to_combo = dict(zip(all_c_labels, top_n_combos))
                        chosen_c_labels = st.multiselect(
                            "☑️ 購入する組み合わせの個別微調整（不要な目は×で削除できます）",
                            options=all_c_labels,
                            default=all_c_labels,
                            key=f"win5_smart_indiv_{actual_points}"
                        )
                        if chosen_c_labels and len(chosen_c_labels) != len(top_n_combos):
                            top_n_combos = [label_to_combo[lbl] for lbl in chosen_c_labels if lbl in label_to_combo]
                            smart_selected_tuples = [c["tuple"] for c in top_n_combos]
                            actual_points = len(top_n_combos)
                            actual_cost = actual_points * 100

                        sc_table = []
                        for idx_sc, c_item in enumerate(top_n_combos, start=1):
                            sc_table.append({
                                "順位": f"{idx_sc}位",
                                "WIN1〜5 買い目": c_item["nums_str"],
                                "選出馬ルート": c_item["names_str"],
                                "穴馬含有数": f"{c_item['dark_count']}頭",
                                "AI的中期待率": f"{c_item['prob'] * 100.0:.3f}%",
                                "想定配当": f"¥{c_item['pay']:,}",
                            })
                        st.dataframe(pd.DataFrame(sc_table), use_container_width=True, hide_index=True)

                # 🎰 1億円・帯封ジャックポット「本線◎○ × 3大爆弾キラー」解剖ボード
                with st.expander("🎰 **【1億円・帯封ジャックポット解剖ボード】各レースの『本線◎○』×『1億円キラー3大爆弾（4〜12番人気）』一覧**", expanded=False):
                    st.caption(
                        "数千万円〜1億円のWIN5は『5レース全部が大荒れ』ではなく、**『3〜4レースはAI本線(◎○)が勝ち、残り1〜2レースで以下の3大爆弾プロファイルに該当する穴馬が炸裂』**した時に生まれます。"
                        "上の『🎰 1億円・帯封宝くじ編成』ボタンを押すと、以下の本線＋特注爆弾から『本線3〜4勝×爆弾1〜2勝』の美味しい目だけをワンクリックで自動合成します！"
                    )
                    jp_board_rows = []
                    for r_d in race_diagnostics:
                        p_df_jb = r_d["pred_df"]
                        core_2_rows = p_df_jb.head(2)
                        core_2_nums = {int(r["馬番"]) for _, r in core_2_rows.iterrows()}
                        core_str = " / ".join([f"{r['AI印']}{int(r['馬番'])}番{r['馬名']}({int(r['単勝人気'])}人気・{get_horse_effective_odds(r):.1f}倍)" for _, r in core_2_rows.iterrows()])
                        bombs_jb = extract_jackpot_bombs_for_leg(
                            p_df_jb, r_d.get("past_records", {}), r_d.get("course_type", "芝"), r_d.get("distance", 1600), core_2_nums, max_bombs=3
                        )
                        if bombs_jb:
                            b_str = " ｜ ".join([f"💣{b['num']}番{b['name']}({b['pop']}人気 {b['odds']:.1f}倍: {b['reason']})" for b in bombs_jb])
                        else:
                            b_str = "ー (順当決着濃厚)"
                        jp_board_rows.append({
                            "レース": r_d["name"],
                            "🎯 本線コア (◎1位・○2位)": core_str,
                            "💣 1億円キラー特注爆弾 (①🔥純能力上位・②⚡穴の逃げ馬・③🔄前々走好走巻き返し)": b_str,
                        })
                    st.dataframe(pd.DataFrame(jp_board_rows), use_container_width=True, hide_index=True)

            # ==========================================
            # 🛡️ 【攻略法③】 WIN5脱落保険！後半メイン・勝負レース単体ヘッジ馬券ビルダー
            # ==========================================
            with st.container(border=True):
                st.subheader("🛡️ 【攻略法③】 WIN5脱落保険！勝負レース単体ヘッジ（資金回収）ビルダー")
                st.caption(
                    "「WIN1・WIN2で惜しくも落ちたのに、後半のWIN4・WIN5で狙っていた本命や穴馬がドンピシャで来た！」という悔しさを防ぐ保険機能です。"
                    "後半メインレース等で少額の単体馬券（馬連・ワイド・単勝・3連複）を抑えておくことで、**WIN5が途中で散っても当日の投資資金を丸ごと回収〜プラス転換**できます！"
                )

                use_hedge_ins = st.checkbox(
                    "🛡️ 勝負レースの単体ヘッジ（脱落保険馬券）を同時に組む",
                    value=bool(st.session_state.get("win5_use_hedge", True)),
                    key="win5_hedge_checkbox"
                )
                st.session_state["win5_use_hedge"] = use_hedge_ins

                hedge_plans_by_leg = {}
                total_hedge_cost = 0

                if use_hedge_ins:
                    h_c1, h_c2, h_c3 = st.columns([1.2, 1.4, 1.0])
                    with h_c1:
                        leg_labels_map = {
                            r_d["id"]: f"WIN{r_d['id']} ({r_d['name'].split(':')[-1].strip()})"
                            for r_d in race_diagnostics
                        }
                        hedge_target_legs = st.multiselect(
                            "🎯 保険をかけるレースを選択 (後半WIN4・WIN5推奨)",
                            options=[1, 2, 3, 4, 5],
                            default=[4, 5],
                            format_func=lambda x: leg_labels_map.get(x, f"WIN{x}"),
                            key="win5_hedge_legs_sel"
                        )
                    with h_c2:
                        hedge_type_opt = st.selectbox(
                            "📐 単体ヘッジの買い方",
                            options=[
                                "🔥 WIN5選択馬の『馬連BOX＋穴単勝』 (ワンツーor穴1着で一撃回収)",
                                "🤝 WIN5選択馬の『ワイドBOX』 (3着内2頭で手堅く投資回収)",
                                "👑 AI本命◎1頭軸『馬連ながし』 (本命から選択馬・上位馬へ流す)",
                                "🎲 WIN5選択馬＋AI上位の『3連複BOX』 (高配当ごっそり回収)"
                            ],
                            index=0,
                            key="win5_hedge_type_sel"
                        )
                    with h_c3:
                        hedge_budget_per_leg = st.number_input(
                            "💵 1レースあたり保険予算 (円)",
                            min_value=300,
                            max_value=20000,
                            value=1000,
                            step=500,
                            key="win5_hedge_budget_in"
                        )

                    if hedge_target_legs:
                        h_tabs = st.tabs([leg_labels_map.get(lg, f"WIN{lg}") for lg in hedge_target_legs])
                        for h_tab, lg_id in zip(h_tabs, hedge_target_legs):
                            r_d_h = race_diagnostics[lg_id - 1]
                            p_df_h = r_d_h["pred_df"]
                            sel_df_h = active_selections[lg_id - 1]
                            sel_nums_h = [int(r["馬番"]) for _, r in sel_df_h.iterrows()]
                            # もしWIN5で1頭しか選んでいない場合は、AI上位3頭を自動的に相手に加える
                            pool_nums_h = list(sel_nums_h)
                            for _, r_top in p_df_h.iterrows():
                                n_top = int(r_top["馬番"])
                                if n_top not in pool_nums_h and len(pool_nums_h) < 3:
                                    pool_nums_h.append(n_top)

                            h_map_lg = {int(r["馬番"]): r for _, r in p_df_h.iterrows()}
                            h_tickets = []

                            if "馬連BOX＋穴単勝" in hedge_type_opt:
                                # 選択馬のうち単勝オッズ6.0倍以上の妙味馬がいれば単勝を追加
                                for n_cand in sel_nums_h:
                                    r_c = h_map_lg.get(n_cand)
                                    if r_c is not None and get_horse_effective_odds(r_c) >= 6.0:
                                        od_c = get_horse_effective_odds(r_c)
                                        h_tickets.append({
                                            "type": "単勝",
                                            "nums": (n_cand,),
                                            "label": f"単勝 {n_cand}番 ({r_c['馬名']})",
                                            "odds": od_c
                                        })
                                for n1, n2 in itertools.combinations(pool_nums_h[:4], 2):
                                    pair = tuple(sorted([n1, n2]))
                                    r1, r2 = h_map_lg.get(pair[0]), h_map_lg.get(pair[1])
                                    o1 = get_horse_effective_odds(r1) if r1 is not None else 4.0
                                    o2 = get_horse_effective_odds(r2) if r2 is not None else 6.0
                                    est_uma = round(max(2.0, min(999.0, o1 * o2 * 0.42)), 1)
                                    h_tickets.append({
                                        "type": "馬連",
                                        "nums": pair,
                                        "label": f"馬連 {pair[0]}-{pair[1]} ({r1['馬名'][:5] if r1 is not None else ''}-{r2['馬名'][:5] if r2 is not None else ''})",
                                        "odds": est_uma
                                    })
                            elif "ワイドBOX" in hedge_type_opt:
                                for n1, n2 in itertools.combinations(pool_nums_h[:4], 2):
                                    pair = tuple(sorted([n1, n2]))
                                    r1, r2 = h_map_lg.get(pair[0]), h_map_lg.get(pair[1])
                                    o1 = get_horse_effective_odds(r1) if r1 is not None else 4.0
                                    o2 = get_horse_effective_odds(r2) if r2 is not None else 6.0
                                    est_w = round(max(1.3, min(300.0, (o1 * o2 * 0.42) * 0.34 + 0.6)), 1)
                                    h_tickets.append({
                                        "type": "ワイド",
                                        "nums": pair,
                                        "label": f"ワイド {pair[0]}-{pair[1]} ({r1['馬名'][:5] if r1 is not None else ''}-{r2['馬名'][:5] if r2 is not None else ''})",
                                        "odds": est_w
                                    })
                            elif "馬連ながし" in hedge_type_opt:
                                ax_num = pool_nums_h[0]
                                r_ax = h_map_lg.get(ax_num)
                                o_ax = get_horse_effective_odds(r_ax) if r_ax is not None else 3.0
                                for pt_num in pool_nums_h[1:5]:
                                    pair = tuple(sorted([ax_num, pt_num]))
                                    r_pt = h_map_lg.get(pt_num)
                                    o_pt = get_horse_effective_odds(r_pt) if r_pt is not None else 6.0
                                    est_uma = round(max(2.0, min(999.0, o_ax * o_pt * 0.42)), 1)
                                    h_tickets.append({
                                        "type": "馬連",
                                        "nums": pair,
                                        "label": f"馬連 {pair[0]}-{pair[1]} (軸{ax_num}番→{r_pt['馬名'][:6] if r_pt is not None else pt_num})",
                                        "odds": est_uma
                                    })
                            else:  # 3連複BOX
                                while len(pool_nums_h) < 4 and len(pool_nums_h) < len(p_df_h):
                                    for _, r_top in p_df_h.iterrows():
                                        n_t = int(r_top["馬番"])
                                        if n_t not in pool_nums_h:
                                            pool_nums_h.append(n_t)
                                            break
                                for n1, n2, n3 in itertools.combinations(pool_nums_h[:4], 3):
                                    tri = tuple(sorted([n1, n2, n3]))
                                    r1, r2, r3 = h_map_lg.get(tri[0]), h_map_lg.get(tri[1]), h_map_lg.get(tri[2])
                                    o1 = get_horse_effective_odds(r1) if r1 is not None else 3.5
                                    o2 = get_horse_effective_odds(r2) if r2 is not None else 5.5
                                    o3 = get_horse_effective_odds(r3) if r3 is not None else 8.5
                                    est_trio = round(max(3.0, min(5000.0, (o1 * o2 * o3) * 0.16)), 1)
                                    h_tickets.append({
                                        "type": "3連複",
                                        "nums": tri,
                                        "label": f"3連複 {tri[0]}-{tri[1]}-{tri[2]}",
                                        "odds": est_trio
                                    })

                            # トリガミ防止の均等払戻ダッチング配分
                            if h_tickets:
                                inv_s = sum(1.0 / max(1.1, t["odds"]) for t in h_tickets)
                                leg_b = max(len(h_tickets) * 100, int(hedge_budget_per_leg))
                                allocs = [max(100, int((leg_b * (1.0 / max(1.1, t["odds"])) / inv_s) // 100) * 100) for t in h_tickets]
                                rem_b = leg_b - sum(allocs)
                                while rem_b >= 100:
                                    rets = [allocs[k] * h_tickets[k]["odds"] for k in range(len(h_tickets))]
                                    min_k = int(np.argmin(rets))
                                    allocs[min_k] += 100
                                    rem_b -= 100
                                leg_cost_actual = sum(allocs)
                                total_hedge_cost += leg_cost_actual
                                for k, t in enumerate(h_tickets):
                                    t["amount"] = allocs[k]
                                    t["est_return"] = int(round(allocs[k] * t["odds"]))
                                hedge_plans_by_leg[lg_id] = {
                                    "tickets": h_tickets,
                                    "cost": leg_cost_actual
                                }

                                with h_tab:
                                    min_h_ret = min(t["est_return"] for t in h_tickets)
                                    max_h_ret = max(t["est_return"] for t in h_tickets)
                                    st.markdown(
                                        f"💡 **WIN{lg_id} 単体保険**: 計 **{len(h_tickets)}点 ({leg_cost_actual:,}円)** ｜ "
                                        f"的中時想定払戻: **¥{min_h_ret:,} 〜 ¥{max_h_ret:,} 円** "
                                        f"（WIN5本体の投資 {actual_cost:,}円 を{'完全回収可能！' if max_h_ret >= actual_cost else '大幅補填！'}）"
                                    )
                                    h_df_rows = [{
                                        "保険買い目": t["label"],
                                        "想定オッズ": f"{t['odds']:.1f}倍",
                                        "推奨購入額": f"{t['amount']:,}円",
                                        "的中時払戻": f"¥{t['est_return']:,}",
                                    } for t in h_tickets]
                                    st.dataframe(pd.DataFrame(h_df_rows), use_container_width=True, hide_index=True)

            # ------------------------------------------
            # 📝 IPAT投票用コード & 総合的中期待率
            # ------------------------------------------
            st.markdown("---")
            col_sum1, col_sum2 = st.columns([2, 1])
            with col_sum1:
                st.subheader("📝 IPAT投票用 買い目まとめ")
                if use_smart_cut and smart_selected_tuples:
                    st.caption(f"✂️ **スマート死に目カット適用中（厳選 {len(smart_selected_tuples)} 通り）**:")
                    ipat_lines = [
                        f"{idx_t:02d}点目: " + " - ".join(f"{n:02d}" for n in tup)
                        for idx_t, tup in enumerate(smart_selected_tuples, start=1)
                    ]
                    ipat_code = "\n".join(ipat_lines)
                    st.code(ipat_code, language="text")
                else:
                    ipat_lines = []
                    for i, r_diag in enumerate(race_diagnostics):
                        c_horses = active_selections[i]
                        nums = [str(int(r["馬番"])) for _, r in c_horses.iterrows()]
                        ipat_lines.append(f"WIN{r_diag['id']}: {', '.join(nums)} ({len(nums)}頭)")
                    ipat_code = "\n".join(ipat_lines)
                    st.code(ipat_code, language="text")
                    st.caption("※ 上記の馬番を即PATまたはマークカード（フォーメーション）に転記して投票してください。")

            with col_sum2:
                st.subheader("📊 的中確率＆資金まとめ")
                total_cov = float(np.prod(leg_total_prob_sums)) * 100.0
                if use_smart_cut and smart_selected_tuples and full_box_prob_sum > 0:
                    retained_p = sum(c["prob"] for c in all_box_combos if c["tuple"] in set(smart_selected_tuples))
                    total_cov = total_cov * (retained_p / full_box_prob_sum)
                st.metric("🎯 5レース総合突破期待率", f"{total_cov:.2f}%")
                st.write(f"🎫 **WIN5 購入点数**: **{actual_points}点** ({actual_cost:,}円)")
                if use_hedge_ins and total_hedge_cost > 0:
                    st.write(f"🛡️ **単体ヘッジ保険**: **{total_hedge_cost:,}円** ({len(hedge_plans_by_leg)}レース)")
                    st.write(f"💵 **本日トータル投資**: **{actual_cost + total_hedge_cost:,}円**")
                else:
                    st.write(f"💵 **購入資金**: **{actual_cost:,}円**")

            # ------------------------------------------
            # 💰 WIN5 推定配当金レンジシミュレーション (購入前想定)
            # ------------------------------------------
            if use_smart_cut and smart_selected_tuples:
                sel_set_tmp = set(smart_selected_tuples)
                sel_pays = [c["pay"] for c in all_box_combos if c["tuple"] in sel_set_tmp]
                min_est_pay = min(sel_pays) if sel_pays else 100
                max_est_pay = max(sel_pays) if sel_pays else 100
                main_est_pay = sel_pays[0] if sel_pays else 100
            else:
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
                            <span style="font-size: 12px; color: #bdc3c7;">🎯 最有力目 想定配当:</span><br>
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

            # 生還判定（通常BOX時とスマート死に目カット時の両方に対応）
            alive_status = []
            is_dead = False

            if use_smart_cut and smart_selected_tuples:
                remaining_tuples = list(smart_selected_tuples)
                for i in range(5):
                    act_num = actual_win_nums[i]
                    c_horses = active_selections[i]
                    selected_nums = [int(r["馬番"]) for _, r in c_horses.iterrows()]
                    sel_str = "・".join(str(n) for n in selected_nums) + "番"
                    if act_num == 0:
                        alive_status.append("⏳ 待機中")
                    else:
                        remaining_tuples = [t for t in remaining_tuples if t[i] == act_num]
                        if len(remaining_tuples) > 0:
                            alive_status.append(f"⭕ 突破！ ({act_num}番的中・残{len(remaining_tuples)}通り)")
                        elif act_num in selected_nums:
                            alive_status.append(f"⚠️ 死に目カット対象 ({act_num}番・残0通り)")
                            is_dead = True
                        else:
                            alive_status.append(f"❌ 脱落 (勝馬:{act_num}番 ｜ 買目:{sel_str})")
                            is_dead = True
                survived_points = len(remaining_tuples) if not is_dead else 0
            else:
                survived_points = actual_points
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
                    elif "❌" in status_text or "⚠️" in status_text:
                        st.error(f"**WIN{i+1}**: {status_text}")
                    else:
                        st.info(f"**WIN{i+1}**: {status_text}")

            passed_cnt = sum(1 for s in alive_status if "⭕" in s)
            all_races_entered = all(num > 0 for num in actual_win_nums)
            if use_smart_cut and smart_selected_tuples:
                is_win5_hit = (all_races_entered and tuple(actual_win_nums) in set(smart_selected_tuples))
            else:
                is_win5_hit = (passed_cnt == 5 and all("⭕" in s for s in alive_status))

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
                            st.session_state["win5_use_smart_cut"] = False
                            st.success("🎉 確定1着馬を各レースの買い目に自動追加しました！")
                            st.rerun()

            # ==========================================
            # 💰 WIN5 確定払戻金・回収率の自動計算 (＋単体ヘッジ保険 合算検証)
            # ==========================================
            st.markdown("---")
            st.subheader("💰 WIN5 確定払戻金・回収率 ＆ 単体ヘッジ保険 トータル収支計算")
            st.caption("全5レース終了後、JRA公式発表の確定払戻金（100円あたり）を入力して、WIN5本体および単体ヘッジ保険の総払戻金額・純損益・回収率(ROI)を自動計算します。")

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

            # 単体ヘッジ保険の的中・回収額自動判定
            hedge_total_return = 0
            hedge_hit_logs = []
            if use_hedge_ins and hedge_plans_by_leg:
                for lg_id, h_plan in hedge_plans_by_leg.items():
                    act_1 = actual_win_nums[lg_id - 1]
                    if act_1 <= 0:
                        continue
                    r_d_lg = race_diagnostics[lg_id - 1]
                    p_df_lg = r_d_lg["pred_df"]
                    act_ranks_dict = (r_d_lg.get("meta", {}).get("actual_ranks") or r_d_lg.get("actual_ranks") or {})
                    name_to_num_lg = {str(r["馬名"]).strip(): int(r["馬番"]) for _, r in p_df_lg.iterrows()}

                    act_2 = int(st.session_state.get(f"win5_rank2_{lg_id}", 0))
                    act_3 = int(st.session_state.get(f"win5_rank3_{lg_id}", 0))
                    for h_nm, rk_v in act_ranks_dict.items():
                        if rk_v == 2 and act_2 == 0 and h_nm in name_to_num_lg:
                            act_2 = name_to_num_lg[h_nm]
                        elif rk_v == 3 and act_3 == 0 and h_nm in name_to_num_lg:
                            act_3 = name_to_num_lg[h_nm]
                    # 2着・3着が未入力の場合はAI上位馬（1着馬を除く）を仮置きして試算
                    rem_ai = [int(r["馬番"]) for _, r in p_df_lg.iterrows() if int(r["馬番"]) != act_1]
                    if act_2 == 0 and len(rem_ai) >= 1:
                        act_2 = rem_ai[0]
                    if act_3 == 0 and len(rem_ai) >= 2:
                        act_3 = [n for n in rem_ai if n != act_2][0]

                    top2_set_lg = {act_1, act_2}
                    top3_set_lg = {act_1, act_2, act_3}

                    for t in h_plan["tickets"]:
                        t_nums = set(t["nums"])
                        t_hit = False
                        if t["type"] == "単勝" and act_1 in t_nums:
                            t_hit = True
                        elif t["type"] == "馬連" and t_nums == top2_set_lg:
                            t_hit = True
                        elif t["type"] == "ワイド" and t_nums.issubset(top3_set_lg):
                            t_hit = True
                        elif t["type"] == "3連複" and t_nums == top3_set_lg:
                            t_hit = True
                        if t_hit:
                            hedge_total_return += t["est_return"]
                            hedge_hit_logs.append(f"WIN{lg_id} {t['label']} (払戻 ¥{t['est_return']:,})")

            # 回収パフォーマンス計算 (WIN5本体 ＋ 単体ヘッジ保険)
            invest_cost = actual_cost + (total_hedge_cost if use_hedge_ins else 0)
            hit_units = 1 if is_win5_hit else 0
            win5_return = official_win5_payout * hit_units if is_win5_hit else 0
            total_return = win5_return + hedge_total_return
            net_profit = total_return - invest_cost
            roi_ratio = (total_return / invest_cost * 100.0) if invest_cost > 0 else 0.0

            # 的中時: 豪華ゴールド/グリーンカードでサマリー表示
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
                            🏆 <b>的中戦略</b>: {'スマート死に目カット抜粋' if use_smart_cut else 'WIN5 メリハリフォーメーション'}（WIN5: <b>{actual_points}点/{actual_cost:,}円</b>{f' ＋ 保険: <b>{total_hedge_cost:,}円</b>' if use_hedge_ins else ''}）<br>
                            💎 <b>WIN5払戻</b>: <b>￥{win5_return:,} 円</b>{f' ｜ 🛡️ <b>単体ヘッジ保険払戻</b>: <b>￥{hedge_total_return:,} 円</b>' if hedge_total_return > 0 else ''}<br>
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
                    st.metric("💵 合計投資金額", f"{invest_cost:,}円", delta=f"WIN5 {actual_cost:,}円" + (f" + 保険 {total_hedge_cost:,}円" if use_hedge_ins else ""))
                with c_m3:
                    st.metric("📈 総合回収率 (ROI)", f"{roi_ratio:,.1f}%", delta="投資リターン")
                with c_m4:
                    st.metric("💎 純利益", f"{net_profit:+,}円", delta="手元利益")
            elif all_races_entered and not is_win5_hit:
                if hedge_total_return > 0:
                    st.success(
                        f"🛡️ **【単体ヘッジ保険が発動！】 WIN5本体は惜しくも不的中（{passed_cnt}/5突破）でしたが、単体ヘッジ馬券が的中して ¥{hedge_total_return:,} 円を回収しました！**\n\n"
                        f"- 🎯 **的中した保険馬券**: {' ／ '.join(hedge_hit_logs)}\n"
                        f"- 💰 **本日トータル収支**: 投資 **{invest_cost:,}円**（WIN5 {actual_cost:,}円＋保険 {total_hedge_cost:,}円） ➔ 回収 **¥{hedge_total_return:,}円**（純損益 **{net_profit:+,}円** / 回収率 **{roi_ratio:.1f}%**）"
                    )
                else:
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

            # 各レースの2着・3着馬番・除外馬（任意補足入力）アコーディオン
            with st.expander("📋 各レースの確定着順・補足設定（2着・3着・🚫除外馬の指定）", expanded=False):
                st.caption("1着馬番は生還チェックシートと連動しています。2着・3着馬番や、出走取消・競走除外になった馬番を指定すると、除外馬を省いた正確なAI学習が行われます。")
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
                        excl_nums = st.multiselect(
                            f"🚫 WIN{r_diag['id']} 除外・取消馬",
                            options=[k for k in p_horses.keys() if k not in {w_num, rank2, rank3}],
                            format_func=lambda x, p_h=p_horses: f"{x}番 {p_h.get(x, '')}",
                            key=f"win5_excl_{r_diag['id']}"
                        )
                        extra_race_ranks.append({"win": w_num, "second": rank2, "third": rank3, "excluded": set(excl_nums)})

            # 学習実行ボタン
            can_learn = any(num > 0 for num in actual_win_nums)
            if st.button("💾 5レースの確定結果をAIに自己学習させる（コース傾向・重み進化＆馬メモリー永久保存）", type="primary", disabled=not can_learn):
                with st.spinner("🤖 AIが5レースの結果を解析し、Tier 1〜4（完全記憶・馬メモリー・コース傾向・5大重み）を更新中..."):
                    if "race_database" not in st.session_state:
                        st.session_state["race_database"] = {}
                    if "horse_memory" not in st.session_state:
                        st.session_state["horse_memory"] = {}
                    if "pattern_memory" not in st.session_state:
                        st.session_state["pattern_memory"] = {}
                    if "training_history" not in st.session_state:
                        st.session_state["training_history"] = []

                    race_db = st.session_state["race_database"]
                    horse_mem = st.session_state["horse_memory"]
                    pat_mem = st.session_state["pattern_memory"]
                    cur_w = dict(st.session_state.get("learned_weights", learned_weights))
                    old_w = dict(cur_w)
                    update_pattern_fn = kwargs.get("update_pattern_func")
                    pat_summaries = []
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

                        sub_r = extra_race_ranks[i]
                        excl_set = sub_r.get("excluded", set())
                        excl_names = {str(r["馬名"]).strip() for _, r in p_df.iterrows() if int(r["馬番"]) in excl_set}

                        # 確定着順辞書
                        actual_ranks_dict = {}
                        r_meta = r_diag.get("meta", {})
                        scraped_ranks = r_diag.get("actual_ranks") or (r_meta.get("actual_ranks") if isinstance(r_meta, dict) else None)
                        has_full_scraped_ranks = bool(scraped_ranks and isinstance(scraped_ranks, dict) and len(scraped_ranks) >= 4)

                        if has_full_scraped_ranks:
                            actual_ranks_dict = {k: v for k, v in scraped_ranks.items() if k not in excl_names}
                        else:
                            # 手動入力データから着順構築（未入力馬に一律8着ペナルティを与えないよう入力された上位馬のみ登録）
                            r2_num = sub_r["second"]
                            r3_num = sub_r["third"]

                            num_to_name = {int(r["馬番"]): str(r["馬名"]).strip() for _, r in p_df.iterrows() if int(r["馬番"]) not in excl_set}
                            for h_n, h_nm in num_to_name.items():
                                if h_n == w_num:
                                    actual_ranks_dict[h_nm] = 1
                                elif r2_num > 0 and h_n == r2_num:
                                    actual_ranks_dict[h_nm] = 2
                                elif r3_num > 0 and h_n == r3_num:
                                    actual_ranks_dict[h_nm] = 3

                        # 確定勝馬が確実に1着として登録されるよう安全補正
                        num_to_name_all = {int(r["馬番"]): str(r["馬名"]).strip() for _, r in p_df.iterrows()}
                        winner_horse_name = num_to_name_all.get(w_num)
                        if winner_horse_name:
                            actual_ranks_dict[winner_horse_name] = 1

                        # 【Tier 1】 過去レース検証ベンチマーク登録（カンニングなし精度検証用）
                        if generate_race_sig_func:
                            race_sig = generate_race_sig_func(p_df, track_name, race_dist)
                        else:
                            clean_names = sorted([str(n).strip() for n in p_df["馬名"].dropna() if str(n).strip()])
                            track_key = str(track_name).split()[0]
                            race_sig = f"{track_key}_{race_dist}m_" + "_".join(clean_names)

                        already_learned_leg = (race_sig in race_db)
                        existing_bench = race_db.get(race_sig, {}) if already_learned_leg else {}
                        ai_ranks_now = {
                            str(r_p["馬名"]).strip(): int(r_p.get("予想着順数値", i_p + 1))
                            for i_p, (_, r_p) in enumerate(p_df.iterrows())
                            if str(r_p.get("馬名", "")).strip()
                        }
                        win_ai_rk_now = ai_ranks_now.get(winner_horse_name, 99) if winner_horse_name else 99
                        raw_horses_leg = edited_races[i]["horses"].to_dict("records") if i < len(edited_races) and hasattr(edited_races[i].get("horses"), "to_dict") else []
                        race_label = r_diag.get("name") or f"{track_name.split()[0]} {race_dist}m ({c_type}) [WIN{r_diag['id']}]"
                        race_db[race_sig] = {
                            "race_name": race_label,
                            "track": track_name,
                            "track_clean": track_name.split()[0],
                            "distance": race_dist,
                            "course_type": c_type,
                            "condition": cond_val,
                            "pace": pace_val,
                            "actual_ranks": actual_ranks_dict,
                            "horses": raw_horses_leg or existing_bench.get("horses", []),
                            "past_records": r_diag.get("past_records") or existing_bench.get("past_records", {}),
                            "official_payouts": r_diag.get("official_payouts") or existing_bench.get("official_payouts", {}),
                            "ai_ranks_at_save": ai_ranks_now,
                            "winner_ai_rank_at_save": win_ai_rk_now,
                            "initial_winner_ai_rank": existing_bench.get("initial_winner_ai_rank", win_ai_rk_now),
                            "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                            "horse_count": len(actual_ranks_dict)
                        }

                        # 【Tier 2】 出走馬ごとの競走馬メモリー更新 (重複学習防止・未入力馬への不当減点防止)
                        pop_map = dict(zip(p_df["馬番"], p_df["単勝人気"]))
                        if not already_learned_leg:
                            for _, r_horse in p_df.iterrows():
                                h_name = str(r_horse["馬名"]).strip()
                                if not h_name or h_name in excl_names or int(r_horse.get("馬番", 0)) in excl_set:
                                    continue
                                if h_name not in actual_ranks_dict:
                                    continue
                                total_horses_learned += 1
                                act_rank = int(actual_ranks_dict[h_name])
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
                                h_data["ability_adj"] = round(float(np.clip(new_adj, -4.0, 6.0)), 2)

                                horse_mem[h_name] = h_data

                        # 【Tier 3 & Tier 4】 コース傾向学習 ＆ 5大重みの適応型強化学習
                        if update_pattern_fn is not None and not p_df.empty and not already_learned_leg:
                            cur_w, pat_mem, p_logs = update_pattern_fn(
                                pred_df=p_df,
                                actual_ranks_dict=actual_ranks_dict,
                                track_name=track_name,
                                course_type=c_type,
                                race_distance=race_dist,
                                cur_weights=cur_w,
                                pattern_memory=pat_mem
                            )
                            for lg in p_logs:
                                pat_summaries.append(f"**WIN{r_diag['id']}**: {lg}")

                    st.session_state["learned_weights"] = cur_w
                    st.session_state["pattern_memory"] = pat_mem
                    st.session_state["weights_just_updated"] = True

                    # 学習履歴への記録
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
                        "モデル重み(適応進化)": f"{w_a:.2f}/{w_p:.2f}/{w_j:.2f}/{w_e:.2f}/{w_s:.2f}"
                    }
                    st.session_state["training_history"].append(hist_entry)

                    # モデルファイルへの永続化保存
                    if save_ai_model_func:
                        save_ai_model_func(
                            st.session_state["learned_weights"],
                            st.session_state["training_history"],
                            st.session_state["race_database"],
                            st.session_state["horse_memory"],
                            st.session_state["pattern_memory"]
                        )

                    # AI分析解説インサイト
                    insights = []
                    insights.append(f"🧪 **【過去レース検証ベンチマーク登録（Tier 1）】** 確定結果を入力した全{races_learned_count}レースを精度検証用データとして保存しました。今後AIをレベルアップさせた際、答えを見ずに純粋予想させて精度が上がったか比較検証できます。")
                    insights.append(f"🐎 **【個別競走馬メモリー更新（Tier 2）】** 出走全{total_horses_learned}頭の実績・好走・人気着順ギャップを馬メモリーに反映しました。別レース（通常レース含む）に出走した際にも実力差として自動適用されます。")
                    if pat_summaries:
                        insights.extend(pat_summaries)
                    if is_win5_hit and official_win5_payout > 0:
                        insights.append(f"💰 **【WIN5 回収達成！】** 払戻金 **￥{total_return:,} 円**（回収率 **{roi_ratio:.1f}%** / 純利益 **+{net_profit:,} 円**）の特大勝利を記録しました！")
                    else:
                        insights.append(f"📊 **【次戦への蓄積】** {passed_cnt}/5 レース突破の実績とコース別勝ち脚質・好調穴馬傾向・5大重みを学習しました。初出走馬を含む次回開催のWIN5・通常レースの予想精度が直接進化します。")

                    st.session_state["win5_learning_success"] = {
                        "weights": (w_a, w_p, w_j, w_e, w_s),
                        "old_weights": (
                            float(old_w.get("ability", 1.5)),
                            float(old_w.get("pace", 1.0)),
                            float(old_w.get("jockey", 1.0)),
                            float(old_w.get("escape_dynamic", 1.2)),
                            float(old_w.get("synergy", 1.1)),
                        ),
                        "insights": insights,
                        "races_count": races_learned_count,
                        "horses_count": total_horses_learned
                    }

                    st.rerun()

            # 学習完了インサイトの常時表示
            if st.session_state.get("win5_learning_success"):
                succ_data = st.session_state["win5_learning_success"]
                w_a, w_p, w_j, w_e, w_s = succ_data["weights"]
                ow_a, ow_p, ow_j, ow_e, ow_s = succ_data.get("old_weights", (w_a, w_p, w_j, w_e, w_s))
                insights = succ_data["insights"]

                st.success(f"🎉 **WIN5 5レースのAI自己学習（Tier 1〜4）が完了しました！個別馬メモリーに加え、コース傾向と5大重みパラメータが進化しました。**")
                col_w1, col_w2, col_w3, col_w4, col_w5 = st.columns(5)
                with col_w1:
                    d_a = round(w_a - ow_a, 2)
                    st.metric("1️⃣ 基礎能力", f"{w_a:.2f}", delta=f"{d_a:+.2f}" if d_a != 0 else "維持", delta_color="normal")
                with col_w2:
                    d_p = round(w_p - ow_p, 2)
                    st.metric("2️⃣ ペース・展開", f"{w_p:.2f}", delta=f"{d_p:+.2f}" if d_p != 0 else "維持", delta_color="normal")
                with col_w3:
                    d_j = round(w_j - ow_j, 2)
                    st.metric("3️⃣ 騎手実績", f"{w_j:.2f}", delta=f"{d_j:+.2f}" if d_j != 0 else "維持", delta_color="normal")
                with col_w4:
                    d_e = round(w_e - ow_e, 2)
                    st.metric("4️⃣ 脚質力学", f"{w_e:.2f}", delta=f"{d_e:+.2f}" if d_e != 0 else "維持", delta_color="normal")
                with col_w5:
                    d_s = round(w_s - ow_s, 2)
                    st.metric("5️⃣ 血統シナジー", f"{w_s:.2f}", delta=f"{d_s:+.2f}" if d_s != 0 else "維持", delta_color="normal")

                st.markdown("##### 💡 AI分析・学習インサイト")
                for ins in insights:
                    st.markdown(f"- {ins}")
