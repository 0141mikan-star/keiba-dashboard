"""
bigdata_ai_bridge.py
33.8万件の中央(JRA)・地方(NAR)ビッグデータから学習した
「3大アンサンブルAIモデル (市場統合 × オッズなし純能力 × LambdaMARTランキング)」および
「騎手×競馬場別勝率辞書・競走馬能力プロファイル辞書」を
メイン予想アプリ (app.py / win5_simulator.py) に直結する推論ブリッジモジュール。
"""

import os
import re
import pickle
import numpy as np
import pandas as pd

_CACHED_KB = None
_CACHED_KB_MTIME = 0.0
_CACHED_ENS = None
_CACHED_ENS_MTIME = 0.0


def load_bigdata_resources(base_dir: str = None):
    """知識辞書 (bigdata_knowledge.pkl) と 3大アンサンブルモデル (ensemble_models.pkl) を高速ロード"""
    global _CACHED_KB, _CACHED_KB_MTIME, _CACHED_ENS, _CACHED_ENS_MTIME

    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    kb_path = os.path.join(base_dir, "models", "bigdata_knowledge.pkl")
    ens_path = os.path.join(base_dir, "models", "ensemble_models.pkl")

    if os.path.exists(kb_path):
        mtime = os.path.getmtime(kb_path)
        if _CACHED_KB is None or mtime != _CACHED_KB_MTIME:
            try:
                with open(kb_path, "rb") as f:
                    _CACHED_KB = pickle.load(f)
                _CACHED_KB_MTIME = mtime
            except Exception:
                pass

    if os.path.exists(ens_path):
        mtime = os.path.getmtime(ens_path)
        if _CACHED_ENS is None or mtime != _CACHED_ENS_MTIME:
            try:
                with open(ens_path, "rb") as f:
                    _CACHED_ENS = pickle.load(f)
                _CACHED_ENS_MTIME = mtime
            except Exception:
                pass

    return _CACHED_KB, _CACHED_ENS


def extract_clean_track_name(track_str: str) -> str:
    """'東京 (左回り / ...)' のようなUI文字列から純粋な競馬場名 ('東京', '帯広ば' 等) を抽出"""
    if not track_str:
        return "東京"
    s = str(track_str).strip()
    if "帯広" in s:
        return "帯広ば"
    for t in ["東京", "中山", "阪神", "京都", "中京", "新潟", "小倉", "福島", "函館", "札幌", "盛岡", "水沢", "大井", "川崎", "船橋", "浦和", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "門別"]:
        if t in s:
            return t
    return s.split()[0]


def get_bigdata_jockey_rate(jockey_name: str, track_str: str = "", fallback_rate: float = 9.0) -> tuple[float, float]:
    """
    ビッグデータ辞書から (コース別勝率(%), 総合勝率(%)) を取得。
    該当騎手が辞書にある場合は33.8万件の実績値を返し、なければfallback_rateを返す。
    """
    kb, _ = load_bigdata_resources()
    if not kb or not jockey_name:
        return fallback_rate, fallback_rate

    j_clean = re.sub(r"[\s▲△★☆◇◆]", "", str(jockey_name)).strip()
    t_clean = extract_clean_track_name(track_str)

    j_map = kb.get("jockey_win_pct_map", {})
    jc_map = kb.get("jockey_course_pct_map", {})

    # 完全一致または部分一致検索
    total_rate = j_map.get(j_clean)
    matched_j_key = j_clean
    if total_rate is None:
        # 姓のみ・短縮名対応
        for k, v in j_map.items():
            if j_clean in k or k in j_clean:
                total_rate = v
                matched_j_key = k
                break

    if total_rate is None:
        return fallback_rate, fallback_rate

    # Fallback辞書(トップ騎手最新補正)とビッグデータ実績のハイブリッド統合
    eff_total = max(float(total_rate), float(fallback_rate) * 0.92) if fallback_rate > 12.0 else float(total_rate)
    course_rate = jc_map.get(f"{matched_j_key}_{t_clean}", eff_total)
    # コース実績と総合実績のブレンド
    blended_course = round(0.65 * float(course_rate) + 0.35 * eff_total, 1)
    return blended_course, round(eff_total, 1)


def _estimate_race_prize_from_text(race_text: str, rank_val: float) -> float:
    """過去走テキストと着順・着差から獲得賞金指数（クラス価値×着順・着差倍率）を算出"""
    s = str(race_text or "")
    if re.search(r"\b(?:GI|JpnI|G1|Jpn1)\b", s):
        base = 150.0
    elif re.search(r"\b(?:GII|JpnII|G2|Jpn2)\b", s):
        base = 70.0
    elif re.search(r"\b(?:GIII|JpnIII|G3|Jpn3|重賞)\b", s):
        base = 42.0
    elif re.search(r"(?:\bL\b|リステッド|\bOP\b|オープン)", s):
        base = 25.0
    elif "3勝" in s or "1600万" in s:
        base = 18.5
    elif "2勝" in s or "1000万" in s:
        base = 11.5
    elif "1勝" in s or "500万" in s:
        base = 8.0
    else:
        base = 6.0

    rk = int(round(float(rank_val))) if rank_val is not None else 8
    mult_map = {1: 1.00, 2: 0.42, 3: 0.26, 4: 0.16, 5: 0.11}
    mult = mult_map.get(rk, 0.04 if rk <= 8 else 0.0)
    m_mg = re.search(r"着差(\d+\.\d+)s", s)
    if m_mg:
        mg_v = float(m_mg.group(1))
        if mg_v <= 0.4:
            margin_mult = max(0.0, 0.48 - mg_v * 0.90)
            mult = max(mult, margin_mult)
    return round(base * mult + base * 0.05, 2)


def compute_pandas_horse_timeseries(
    df: pd.DataFrame,
    past_performances: dict = None,
    horse_profiles: dict = None,
) -> dict:
    """
    Pandasの groupby('horse_id').shift(1) および rolling(window=3).mean() を用いて
    各馬の時系列特徴量（前走着順・前走獲得賞金指数・過去3走平均獲得賞金・過去3走平均着順・レース内相対賞金比）を算出する。
    """
    past_performances = past_performances or {}
    horse_profiles = horse_profiles or {}
    ts_rows = []

    for idx, (_, row) in enumerate(df.iterrows()):
        h_name = str(row.get("馬名", f"horse_{idx}")).strip()
        p_info = past_performances.get(h_name, {})
        r_stats = p_info.get("real_stats", {}) or {}
        prof = horse_profiles.get(h_name, {})

        l0_str = str(p_info.get("last_race", ""))
        l1_str = str(p_info.get("second_last", ""))
        m_r0 = re.search(r"(\d+)着", l0_str)
        m_r1 = re.search(r"(\d+)着", l1_str)
        rk0 = float(m_r0.group(1)) if m_r0 else float(r_stats.get("last1_rank", prof.get("last_rank", 6.0)))
        rk_avg3 = float(r_stats.get("last3_avg_rank", rk0))
        rk1 = float(m_r1.group(1)) if m_r1 else rk_avg3
        rk2 = max(1.0, min(18.0, 3.0 * rk_avg3 - rk0 - rk1))

        pz0 = _estimate_race_prize_from_text(l0_str, rk0)
        pz1 = _estimate_race_prize_from_text(l1_str, rk1)
        pz2 = float(prof.get("rolling3_prize", (pz0 + pz1) * 0.5))

        # 時系列順（3走前 -> 2走前 -> 前走 -> 今回出走枠）で行を構築
        ts_rows.append({"horse_id": h_name, "seq": 1, "rank": rk2, "prize": pz2})
        ts_rows.append({"horse_id": h_name, "seq": 2, "rank": rk1, "prize": pz1})
        ts_rows.append({"horse_id": h_name, "seq": 3, "rank": rk0, "prize": pz0})
        ts_rows.append({"horse_id": h_name, "seq": 4, "rank": np.nan, "prize": np.nan})

    if not ts_rows:
        return {}

    ts_df = pd.DataFrame(ts_rows).sort_values(by=["horse_id", "seq"]).reset_index(drop=True)
    # 1. groupby('horse_id').shift(1): 1つ前の行（前走）の着順・賞金データを取得
    h_grp = ts_df.groupby("horse_id", sort=False)
    ts_df["shift1_rank"] = h_grp["rank"].shift(1)
    ts_df["shift1_prize"] = h_grp["prize"].shift(1)

    # 2. groupby('horse_id')['prize'].rolling(window=3).mean(): 過去3走の移動平均を算出
    ts_df["rolling3_prize"] = h_grp["shift1_prize"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())
    ts_df["rolling3_rank"] = h_grp["shift1_rank"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())

    current_rows = ts_df[ts_df["seq"] == 4].copy()
    mean_pz = max(1.0, float(current_rows["rolling3_prize"].mean()))
    current_rows["prize_rank"] = current_rows["rolling3_prize"].rank(ascending=False, method="min").astype(int)
    current_rows["prize_ratio"] = current_rows["rolling3_prize"] / mean_pz

    out_map = {}
    for _, r_cur in current_rows.iterrows():
        h_id = str(r_cur["horse_id"])
        out_map[h_id] = {
            "shift1_rank": round(float(r_cur["shift1_rank"]), 2) if pd.notna(r_cur["shift1_rank"]) else 6.0,
            "shift1_prize": round(float(r_cur["shift1_prize"]), 2) if pd.notna(r_cur["shift1_prize"]) else 3.0,
            "rolling3_prize": round(float(r_cur["rolling3_prize"]), 2) if pd.notna(r_cur["rolling3_prize"]) else 3.0,
            "rolling3_rank": round(float(r_cur["rolling3_rank"]), 2) if pd.notna(r_cur["rolling3_rank"]) else 6.0,
            "prize_rank": int(r_cur["prize_rank"]) if pd.notna(r_cur["prize_rank"]) else 8,
            "prize_ratio": round(float(r_cur["prize_ratio"]), 2) if pd.notna(r_cur["prize_ratio"]) else 1.0,
        }
    return out_map


def predict_race_with_bigdata_ai(
    df: pd.DataFrame,
    track_name: str,
    course_type: str,
    race_distance: int,
    track_condition: str,
    past_performances: dict = None,
    jockey_fallback_fn=None,
    prediction_mode: str = "standard",
) -> dict:
    """
    出馬表DataFrameに対して7モデル・ハイブリッドアンサンブル (LightGBM + CatBoost + LambdaMART Ranker) を実行し、
    各馬のアンサンブル勝率・純能力勝率・ランキング勝率・過小評価穴馬フラグを返す。
    prediction_mode:
      - "standard": 🛡️ 通常・王道モード (的中率最大化ハイブリッド)
      - "upset_mild": ⚡ 中波乱・妙味モード (オッズ依存半減・純能力＆妙味重視)
      - "upset_chaos": 🌪️ 大荒れ・オッズ封印モード (オッズ完全無視・純能力＆隠れ穴馬特化)
    """
    kb, ens = load_bigdata_resources()
    if df.empty or ens is None:
        return {}

    model_full = ens.get("model_full")
    model_full_top3 = ens.get("model_full_top3")
    model_pure = ens.get("model_pure")
    model_pure_top3 = ens.get("model_pure_top3")
    model_ranker = ens.get("model_ranker")
    model_cat_pure = ens.get("model_cat_pure")
    model_cat_full = ens.get("model_cat_full")
    cat_cols = ens.get("cat_cols", ["track_name_code", "course_type_code", "track_condition_code", "gender_code", "grade_code", "category_code"])
    full_features = ens.get("full_features", [])
    pure_features = ens.get("pure_features", [])

    if model_full is None or model_pure is None or model_ranker is None:
        return {}

    t_clean = extract_clean_track_name(track_name)
    is_nar = t_clean in ["盛岡", "水沢", "大井", "川崎", "船橋", "浦和", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "門別", "帯広ば"]
    cat_str = "NAR" if is_nar else "JRA"

    cond_clean = "良"
    if "不良" in str(track_condition):
        cond_clean = "不良"
    elif "重" in str(track_condition) and "稍" not in str(track_condition):
        cond_clean = "重"
    elif "稍" in str(track_condition):
        cond_clean = "稍重"

    cat_mappings = (kb or {}).get("cat_mappings", {})
    track_code = cat_mappings.get("track_name", {}).get(t_clean, 0)
    course_code = cat_mappings.get("course_type", {}).get(course_type, 0)
    cond_code = cat_mappings.get("track_condition", {}).get(cond_clean, 0)
    cat_code = cat_mappings.get("category", {}).get(cat_str, 0)
    grade_code = cat_mappings.get("grade", {}).get("一般", 0)
    gender_code = cat_mappings.get("gender", {}).get("牡", 0)

    horse_profiles = (kb or {}).get("horse_profile_map", {})
    past_performances = past_performances or {}

    n_horses = len(df)
    style_to_corner = {"逃げ": 0.10, "先行": 0.28, "差し": 0.62, "追込": 0.85}
    escape_or_lead_cnt = int(df["脚質"].isin(["逃げ", "先行"]).sum()) if "脚質" in df.columns else max(1, n_horses // 3)
    front_runners_cnt = int((df["脚質"] == "逃げ").sum()) if "脚質" in df.columns else 1

    rows_feat = []
    for idx, (_, row) in enumerate(df.iterrows()):
        h_name = str(row.get("馬名", "")).strip()
        h_num = int(row.get("馬番", idx + 1))
        bracket_num = max(1, min(8, (h_num - 1) * 8 // max(1, n_horses) + 1))
        pop = int(row.get("単勝人気", idx + 1))

        # 実オッズまたは推定オッズ
        odds_val = None
        for oc in ["想定単勝オッズ", "単勝オッズ"]:
            v = row.get(oc)
            if v is not None and not pd.isna(v):
                try:
                    vf = float(v)
                    if vf > 1.0:
                        odds_val = vf
                        break
                except Exception:
                    pass
        if odds_val is None:
            pop_map = {1: 2.2, 2: 3.8, 3: 5.6, 4: 8.2, 5: 12.0, 6: 18.0, 7: 26.0, 8: 38.0, 9: 55.0, 10: 78.0}
            odds_val = pop_map.get(pop, min(250.0, pop * 15.0))

        w_carried = 56.0
        try:
            if row.get("斤量") is not None and not pd.isna(row.get("斤量")):
                w_carried = float(row.get("斤量"))
        except Exception:
            pass

        j_name = str(row.get("騎手名", "")).strip()
        fb_j = jockey_fallback_fn(j_name) if jockey_fallback_fn else 9.0
        jc_pct, j_tot_pct = get_bigdata_jockey_rate(j_name, track_name, fallback_rate=fb_j)
        jc_rate = jc_pct / 100.0
        j_tot_rate = j_tot_pct / 100.0
        j_top3_rate = min(0.65, j_tot_rate * 2.8 + 0.05)

        # 競走馬プロファイル（ビッグデータ辞書 & netkeiba過去5走実測データの統合）
        prof = horse_profiles.get(h_name, {})
        p_info = past_performances.get(h_name, {})
        p_boost = float(p_info.get("ability_boost", 0.0))
        r_stats = p_info.get("real_stats", {}) or {}

        # ベースとなるスピード指数（過去走の実着差・勝率・複勝率と融合）
        pop_base_si = max(84.0, 104.5 - pop * 1.35) + p_boost * 1.8
        # オッズ・人気に引きずられない純粋実力ベース指数 (pure_si)
        neutral_base_si = 95.5 + p_boost * 2.4 + (jc_pct - 9.0) * 0.25
        if r_stats:
            last1_rank = float(r_stats.get("last1_rank", min(14, max(1, pop))))
            last3_avg_rk = float(r_stats.get("last3_avg_rank", last1_rank))
            last1_margin = float(r_stats.get("last1_margin_sec", max(0.0, (last1_rank - 1.0) * 0.20)))
            past_avg_margin = float(r_stats.get("past_avg_margin", max(0.15, last1_margin * 0.95)))
            past_wr = float(r_stats.get("past_win_rate", 0.15))
            past_t3 = float(r_stats.get("past_top3_rate", 0.35))
            course_apt = float(r_stats.get("course_aptitude_rate", 0.33))
            dist_apt = float(r_stats.get("dist_aptitude_rate", 0.33))
            dist_chg = float(r_stats.get("dist_change", 0.0))

            # ★ 障害帰りの平地戻し正規化: 平地戦(芝・ダート)で前走が障害戦(障3000m等)の場合、
            # 障害戦特有の大きな着差(5〜7秒差)で平地スピード指数が不当に破壊されるのを防ぎ、平地勝率・距離適性から平地相当着差を復元
            last_str_bd = str(p_info.get("last_race", ""))
            is_hurdle_return = (course_type in ["芝", "ダート"]) and bool(re.search(r"障\s*\d{3,4}", last_str_bd))
            if is_hurdle_return and past_wr >= 0.25 and dist_apt >= 0.55:
                flat_est_m = max(0.12, 0.65 - past_wr * 1.1 - max(0.0, dist_apt - 0.35) * 0.8)
                last1_margin = min(last1_margin, flat_est_m)
                past_avg_margin = min(past_avg_margin, max(0.15, flat_est_m * 0.95))
                last1_rank = min(last1_rank, 2.5 if past_wr >= 0.30 else 4.0)
                last3_avg_rk = min(last3_avg_rk, 2.8 if past_wr >= 0.30 else 4.2)

            # 着差と過去勝率から実力スピード指数を算出（人気順が低くても着差が小さければ高指数！）
            perf_si = 102.8 - min(2.5, past_avg_margin) * 5.2 + past_t3 * 5.8 + p_boost * 1.1
            if prof:
                si = 0.45 * float(prof.get("speed_index", perf_si)) + 0.40 * perf_si + 0.15 * pop_base_si
                pure_si = 0.50 * float(prof.get("speed_index", perf_si)) + 0.50 * perf_si
                best_si = max(pure_si + 2.0, float(prof.get("best_speed_index", pure_si + 3.5)), 104.5 - min(2.0, last1_margin) * 4.5)
                past_wr = 0.55 * past_wr + 0.45 * float(prof.get("win_rate", past_wr))
                past_t3 = 0.55 * past_t3 + 0.45 * float(prof.get("top3_rate", past_t3))
            else:
                si = 0.70 * perf_si + 0.30 * pop_base_si
                pure_si = perf_si
                best_si = max(pure_si + 2.2, 104.5 - min(2.0, min(last1_margin, past_avg_margin)) * 4.8 + past_wr * 6.0)
            last1_si = 103.5 - min(2.5, last1_margin) * 5.5 + (1.5 if last1_rank == 1 else 0.0)
            last3_si = 0.55 * last1_si + 0.45 * pure_si
            raw_agari = r_stats.get("last1_agari")
            if is_hurdle_return and raw_agari is None and r_stats.get("mean_agari") is not None and float(r_stats.get("mean_agari", 99.0)) < 50.0:
                raw_agari = float(r_stats.get("mean_agari"))
            agari_diff = -0.35 if p_boost >= 2.0 else (-0.15 if p_boost >= 1.0 else 0.0)
        elif prof:
            prof_si = float(prof.get("speed_index", neutral_base_si))
            si = 0.65 * prof_si + 0.35 * pop_base_si
            pure_si = 0.85 * prof_si + 0.15 * neutral_base_si
            best_si = max(pure_si + 2.0, float(prof.get("best_speed_index", pure_si + 3.5)))
            last1_si = float(prof.get("last_si", pure_si)) + p_boost * 1.2
            last3_si = 0.5 * last1_si + 0.5 * pure_si
            last1_rank = float(prof.get("last_rank", min(15, max(1, pop))))
            last3_avg_rk = max(1.0, 0.75 * last1_rank + 0.25 * min(8, pop))
            last1_margin = float(prof.get("last_margin", max(0.0, (last1_rank - 1) * 0.20)))
            past_avg_margin = max(0.1, last1_margin * 0.9 + 0.18)
            past_wr = float(prof.get("win_rate", max(0.04, 0.26 - (last1_rank - 1) * 0.02)))
            past_t3 = float(prof.get("top3_rate", max(0.12, 0.52 - (last1_rank - 1) * 0.035)))
            agari_diff = float(prof.get("agari_diff", -p_boost * 0.18))
            raw_agari = None
            suit_str = str(p_info.get("course_suitability", ""))
            course_apt = 0.48 if "◎" in suit_str else (0.35 if "○" in suit_str else 0.22)
            dist_apt = course_apt
            dist_chg = 0.0
        else:
            last_str = str(p_info.get("last_race", ""))
            m_rk = re.search(r"(\d+)着", last_str)
            last1_rank = float(m_rk.group(1)) if m_rk else float(min(12, max(1, pop)))
            last3_avg_rk = max(1.0, 0.70 * last1_rank + 0.30 * min(8, pop))
            last1_margin = max(0.0, (last1_rank - 1.0) * 0.18 - max(0.0, p_boost * 0.15))
            past_avg_margin = max(0.15, last1_margin * 0.92 + 0.12)
            si = pop_base_si
            # 過去走実測がない場合も、調子・コース適性・騎手・展開からオッズ非依存のpure_siを構築
            suit_str = str(p_info.get("course_suitability", ""))
            course_apt = 0.48 if "◎" in suit_str else (0.35 if "○" in suit_str else 0.22)
            dist_apt = course_apt
            pure_si = neutral_base_si + (3.5 - min(7.0, last1_rank)) * 0.9 + (course_apt - 0.33) * 12.0
            best_si = max(si, pure_si) + max(2.5, p_boost * 1.5 + 3.0)
            last1_si = pure_si + (3.5 - min(6.5, last1_rank)) * 1.1
            last3_si = 0.6 * last1_si + 0.4 * pure_si
            past_wr = max(0.03, min(0.50, 0.25 - (last1_rank - 1) * 0.022 + p_boost * 0.035))
            past_t3 = max(0.08, min(0.75, 0.52 - (last1_rank - 1) * 0.040 + p_boost * 0.055))
            agari_diff = -0.35 if p_boost >= 2.0 else (-0.15 if p_boost >= 1.0 else (0.25 if p_boost < 0 else 0.0))
            raw_agari = None
            dist_chg = 0.0

        style_str = str(row.get("脚質", "先行"))
        corner_pos = style_to_corner.get(style_str, float(prof.get("corner_pos", 0.45)))

        rows_feat.append({
            "horse_num": h_num,
            "horse_name": h_name,
            "popularity": pop,
            "odds": odds_val,
            "log_odds": np.log1p(max(1.0, odds_val)),
            "speed_index": si,
            "_pure_si": pure_si,
            "best_speed_index": best_si,
            "last1_speed_index": last1_si,
            "last3_speed_index": last3_si,
            "last1_margin_sec": last1_margin,
            "past_avg_margin": past_avg_margin,
            "last1_rank": last1_rank,
            "last3_avg_rank": last3_avg_rk,
            "past_win_rate": past_wr,
            "past_top3_rate": past_t3,
            "last1_agari_diff": agari_diff,
            "past_agari_diff_mean": agari_diff * 0.85,
            "_raw_agari": raw_agari,
            "past_corner_pos": corner_pos,
            "race_front_runners": front_runners_cnt + max(0, escape_or_lead_cnt - 1),
            "pace_pressure": (front_runners_cnt + max(0, escape_or_lead_cnt - 1)) / max(2, n_horses),
            "course_aptitude_rate": course_apt,
            "dist_aptitude_rate": dist_apt,
            "rest_days": 28.0,
            "dist_change": dist_chg,
            "jockey_course_win_rate": jc_rate,
            "jockey_total_win_rate": j_tot_rate,
            "jockey_top3_rate": j_top3_rate,
            "trainer_win_rate": max(0.06, j_tot_rate * 0.85),
            "trainer_top3_rate": max(0.20, j_top3_rate * 0.90),
            "jockey_horse_compat": max(0.22, 0.5 * j_top3_rate + 0.5 * past_t3),
            "weight_carried": w_carried,
            "horse_weight": 950.0 if t_clean == "帯広ば" else 472.0,
            "weight_change": 0.0,
            "bracket_num": bracket_num,
            "age": 4,
            "distance": int(race_distance),
            "field_size": n_horses,
            "track_name_code": track_code,
            "course_type_code": course_code,
            "track_condition_code": cond_code,
            "gender_code": gender_code,
            "grade_code": grade_code,
            "category_code": cat_code,
            "_style_str": style_str,
            "_p_boost": p_boost,
        })

    f_df = pd.DataFrame(rows_feat)

    # 実測上り3Fがある場合はレース内平均との差分(last1_agari_diff)を精密算出
    valid_ag_mask = f_df["_raw_agari"].notna()
    if valid_ag_mask.sum() >= 2:
        ag_mean = float(f_df.loc[valid_ag_mask, "_raw_agari"].mean())
        f_df.loc[valid_ag_mask, "last1_agari_diff"] = (f_df.loc[valid_ag_mask, "_raw_agari"].astype(float) - ag_mean).clip(-2.0, 2.0)
        f_df.loc[valid_ag_mask, "past_agari_diff_mean"] = f_df.loc[valid_ag_mask, "last1_agari_diff"] * 0.85

    def _populate_derived_features(df_target: pd.DataFrame) -> pd.DataFrame:
        df_target["si_momentum"] = (df_target["last1_speed_index"] - df_target["speed_index"]).clip(-25.0, 25.0)
        df_target["peak_gap"] = (df_target["best_speed_index"] - df_target["speed_index"]).clip(0.0, 30.0)
        df_target["rank_improvement"] = (df_target["last3_avg_rank"] - df_target["last1_rank"]).clip(-12.0, 12.0)

        si_m = df_target["speed_index"].mean()
        si_s = max(1.0, float(df_target["speed_index"].std()) if len(df_target) > 1 else 3.0)
        df_target["race_rel_speed_index"] = (df_target["speed_index"] - si_m) / si_s
        df_target["race_rel_best_si"] = df_target["best_speed_index"] - df_target["best_speed_index"].mean()

        l1_m = df_target["last1_speed_index"].mean()
        l1_s = max(1.0, float(df_target["last1_speed_index"].std()) if len(df_target) > 1 else 3.0)
        df_target["race_rel_last1_si"] = (df_target["last1_speed_index"] - l1_m) / l1_s

        l3_m = df_target["last3_speed_index"].mean()
        l3_s = max(1.0, float(df_target["last3_speed_index"].std()) if len(df_target) > 1 else 3.0)
        df_target["race_rel_last3_si"] = (df_target["last3_speed_index"] - l3_m) / l3_s

        df_target["race_rel_past_top3"] = df_target["past_top3_rate"] - df_target["past_top3_rate"].mean()
        df_target["race_rel_past_win"] = df_target["past_win_rate"] - df_target["past_win_rate"].mean()
        df_target["race_rel_last1_margin"] = (df_target["last1_margin_sec"] - df_target["last1_margin_sec"].min()).clip(0.0, 6.0)
        df_target["race_rel_avg_margin"] = (df_target["past_avg_margin"] - df_target["past_avg_margin"].min()).clip(0.0, 6.0)
        df_target["race_rel_jockey_rate"] = df_target["jockey_course_win_rate"] - df_target["jockey_course_win_rate"].mean()
        df_target["weight_diff_from_avg"] = df_target["weight_carried"] - df_target["weight_carried"].mean()
        df_target["ability_x_jockey"] = df_target["race_rel_speed_index"] * df_target["jockey_course_win_rate"]
        df_target["front_advantage"] = (1.0 - df_target["past_corner_pos"]) * (1.0 - df_target["pace_pressure"])

        inv_o = 1.0 / df_target["odds"].clip(lower=1.05)
        df_target["market_win_prob"] = (inv_o / max(0.1, inv_o.sum())).clip(0.001, 0.95)
        min_o = max(1.0, float(df_target["odds"].min()))
        df_target["odds_ratio_to_fav1"] = (df_target["odds"] / min_o).clip(1.0, 200.0)
        df_target["log_odds_diff_fav1"] = np.log1p(df_target["odds"]) - np.log1p(min_o)
        df_target["pop_pct"] = df_target["popularity"] / df_target["field_size"].clip(lower=2)
        return df_target

    ts_features_map = compute_pandas_horse_timeseries(df, past_performances, horse_profiles)
    f_df = _populate_derived_features(f_df)

    # 1. 純能力モデル推論 (オッズ・人気バイアスを完全除去した pure_si で推論！ LightGBM + CatBoost)
    f_pure_df = f_df.copy()
    f_pure_df["speed_index"] = f_pure_df["_pure_si"]
    f_pure_df = _populate_derived_features(f_pure_df)
    X_pure = f_pure_df[pure_features]
    pure_win_raw = model_pure.predict_proba(X_pure)[:, 1] ** 1.05
    pure_win_prob = pure_win_raw / max(1e-6, np.sum(pure_win_raw))
    if model_pure_top3 is not None:
        pure_t3_raw = model_pure_top3.predict_proba(X_pure)[:, 1] ** 1.75
        pure_t3_prob = pure_t3_raw / max(1e-6, np.sum(pure_t3_raw))
        pure_lgb_prob = 0.65 * pure_win_prob + 0.35 * pure_t3_prob
        pure_lgb_prob = pure_lgb_prob / max(1e-6, np.sum(pure_lgb_prob))
    else:
        pure_lgb_prob = pure_win_prob

    cat_pure_prob = pure_lgb_prob
    if model_cat_pure is not None:
        try:
            X_pure_cat = X_pure.copy()
            for c_col in cat_cols:
                if c_col in X_pure_cat.columns:
                    X_pure_cat[c_col] = X_pure_cat[c_col].astype(int)
            cp_raw = model_cat_pure.predict_proba(X_pure_cat)[:, 1] ** 1.05
            cat_pure_prob = cp_raw / max(1e-6, np.sum(cp_raw))
        except Exception:
            cat_pure_prob = pure_lgb_prob

    pure_prob = pure_lgb_prob

    # 2. フルモデル推論 (Win + Top3 マルチターゲット統合 LightGBM + CatBoost)
    X_full = f_df[full_features]
    full_win_raw = model_full.predict_proba(X_full)[:, 1] ** 1.05
    full_win_prob = full_win_raw / max(1e-6, np.sum(full_win_raw))
    if model_full_top3 is not None:
        full_t3_raw = model_full_top3.predict_proba(X_full)[:, 1] ** 1.85
        full_t3_prob = full_t3_raw / max(1e-6, np.sum(full_t3_raw))
        full_lgb_prob = 0.875 * full_win_prob + 0.125 * full_t3_prob
        full_lgb_prob = full_lgb_prob / max(1e-6, np.sum(full_lgb_prob))
    else:
        full_lgb_prob = full_win_prob

    cat_full_prob = full_lgb_prob
    if model_cat_full is not None:
        try:
            X_full_cat = X_full.copy()
            for c_col in cat_cols:
                if c_col in X_full_cat.columns:
                    X_full_cat[c_col] = X_full_cat[c_col].astype(int)
            cf_raw = model_cat_full.predict_proba(X_full_cat)[:, 1] ** 1.05
            cat_full_prob = cf_raw / max(1e-6, np.sum(cf_raw))
        except Exception:
            cat_full_prob = full_lgb_prob

    full_prob = full_lgb_prob
    cat_ens_prob = 0.65 * cat_full_prob + 0.35 * cat_pure_prob
    cat_ens_prob = cat_ens_prob / max(1e-6, np.sum(cat_ens_prob))

    # 3. LambdaMART ランキングモデル推論 (lambdarank)
    ranker_scores = model_ranker.predict(X_full)
    exp_r = np.exp(np.clip((ranker_scores - np.max(ranker_scores)) / 0.62, -25.0, 0.0))
    ranker_prob = exp_r / max(1e-6, np.sum(exp_r))

    # 🛡️ 通常・王道モード確率 (LightGBM + CatBoost + LambdaMART Plackett-Luce最適ブレンド)
    std_raw = 0.60 * full_prob + 0.02 * cat_ens_prob + 0.16 * ranker_prob + 0.14 * pure_prob + 0.08 * f_df["market_win_prob"].values
    std_prob = std_raw / max(1e-6, np.sum(std_raw))

    # 純能力ランキング & 通常ランキング & CatBoostランキング
    pure_ranks = pd.Series(pure_prob).rank(ascending=False, method="first").astype(int).values
    std_ranks = pd.Series(std_prob).rank(ascending=False, method="first").astype(int).values
    cat_ranks = pd.Series(cat_full_prob).rank(ascending=False, method="first").astype(int).values
    cat_pure_ranks = pd.Series(cat_pure_prob).rank(ascending=False, method="first").astype(int).values

    # 🌪️ 穴馬激走ポテンシャル (オッズ抜き実力 × 着差僅差 × 末脚 × 展開恩恵)
    upset_potentials = np.zeros(n_horses, dtype=float)
    upset_reasons = []
    dangerous_favs = []

    for i, r in f_df.iterrows():
        pop = int(r["popularity"])
        p_rk = int(pure_ranks[i])
        pp = float(pure_prob[i])
        mp = float(r["market_win_prob"])
        l1_m = float(r["last1_margin_sec"])
        avg_m = float(r["past_avg_margin"])
        ag_d = float(r["last1_agari_diff"])
        c_apt = float(r["course_aptitude_rate"])
        st_str = str(r["_style_str"])
        p_bst = float(r.get("_p_boost", 0.0))
        odds_v = float(r.get("odds", 10.0))

        reasons_i = []
        u_mult = 1.0
        # 穴馬ブースト対象は「純能力上位8位以内」または「過去走高評価・着差0.8秒以内」の実力根拠がある馬に限定（無根拠な超人気薄の暴走を防止）
        is_viable_dark = bool(p_rk <= 8 or p_bst >= 0.8 or l1_m <= 0.75 or odds_v <= 60.0)

        # ① 人気より純能力順位が高い（過小評価）
        if pop >= 4 and p_rk < pop and is_viable_dark:
            rank_gap = min(6, pop - p_rk)
            u_mult += min(0.55, rank_gap * 0.10)
            if p_rk <= 4:
                reasons_i.append(f"オッズ封印純能力{p_rk}位(人気{pop}位)")
        # ② 着順は悪くても実着差が僅差（0.6秒以内）
        if pop >= 4 and is_viable_dark and (l1_m <= 0.65 or avg_m <= 0.65):
            u_mult += 0.25
            reasons_i.append(f"近走着差僅差({min(l1_m, avg_m):.1f}秒差)")
        # ③ 上がり3Fが優秀
        if pop >= 4 and is_viable_dark and ag_d <= -0.15:
            u_mult += 0.20
            reasons_i.append("末脚キレ味上位")
        # ④ コース・距離適性◎
        if pop >= 4 and is_viable_dark and c_apt >= 0.42:
            u_mult += 0.18
            reasons_i.append("コース適性◎")
        # ⑤ ペース恩恵（実力裏付けがある馬のみ）
        if pop >= 4 and is_viable_dark and p_rk <= 7:
            if front_runners_cnt >= 2 and st_str in ["差し", "追込"]:
                u_mult += 0.20
                reasons_i.append("ハイペース前崩れ恩恵")
            elif front_runners_cnt == 1 and st_str == "逃げ":
                u_mult += 0.25
                reasons_i.append("単騎逃げ残り警戒")

        # ⚠️ 危険な人気馬判定（過去走もコース適性も良い人気馬は絶対に危険馬にしない！）
        has_strong_profile = bool(p_bst >= 1.0 or l1_m <= 0.35 or c_apt >= 0.42 or p_rk <= 4)
        is_dang = False
        if pop <= 2 and not has_strong_profile and (p_rk >= 6 and pp < mp * 0.55):
            is_dang = True
            u_mult *= 0.72
        elif pop == 3 and not has_strong_profile and p_rk >= 8:
            is_dang = True
            u_mult *= 0.80

        upset_potentials[i] = pp * u_mult
        upset_reasons.append("・".join(reasons_i) if reasons_i else "")
        dangerous_favs.append(is_dang)

    chaos_raw = 0.25 * std_prob + 0.35 * pure_prob + 0.40 * (upset_potentials / max(1e-6, np.sum(upset_potentials)))
    chaos_prob = chaos_raw / max(1e-6, np.sum(chaos_raw))
    mild_raw = 0.55 * std_prob + 0.25 * pure_prob + 0.20 * chaos_prob
    mild_prob = mild_raw / max(1e-6, np.sum(mild_raw))

    if prediction_mode == "upset_chaos":
        active_prob = chaos_prob
    elif prediction_mode == "upset_mild":
        active_prob = mild_prob
    else:
        active_prob = std_prob

    ens_ranks = pd.Series(active_prob).rank(ascending=False, method="first").astype(int).values
    chaos_ranks = pd.Series(chaos_prob).rank(ascending=False, method="first").astype(int).values

    # スコア補正値 (app.py の総合スコアに加算する高精度AIブースト)
    avg_p = 1.0 / max(1, n_horses)
    result_map = {}
    for i, r in f_df.iterrows():
        h_num = int(r["horse_num"])
        h_nm_r = str(r["horse_name"])
        ts_info = ts_features_map.get(h_nm_r, {})
        ep = float(active_prob[i])
        sp = float(std_prob[i])
        pp = float(pure_prob[i])
        cp = float(chaos_prob[i])
        rp = float(ranker_prob[i])
        cat_fp = float(cat_full_prob[i])
        cat_pp = float(cat_pure_prob[i])
        mp = float(r["market_win_prob"])
        pop = int(r["popularity"])

        # 選択モードに応じたスコアブースト
        ratio = max(0.15, ep / max(0.01, avg_p))
        scale = 9.5 if prediction_mode == "upset_chaos" else (9.0 if prediction_mode == "upset_mild" else 8.5)
        lgbm_boost = float(np.clip(scale * np.log(ratio), -8.0, 15.0))

        # 純能力モデルが市場オッズより高く評価している「本物の実力穴馬」への特別加算
        is_pure_value = bool(
            (pop >= 4)
            and (
                pure_ranks[i] <= 3
                or (pp > mp * 1.25 and (pure_ranks[i] <= 5 or ens_ranks[i] <= 4))
            )
        )
        if is_pure_value:
            if prediction_mode == "upset_chaos":
                lgbm_boost += 5.0
            elif prediction_mode == "upset_mild":
                lgbm_boost += 3.5
            else:
                lgbm_boost += 2.5

        if dangerous_favs[i]:
            if prediction_mode == "upset_chaos":
                lgbm_boost -= 3.5
            elif prediction_mode == "upset_mild":
                lgbm_boost -= 2.0

        result_map[h_num] = {
            "ensemble_prob": round(ep * 100.0, 1),
            "standard_prob": round(sp * 100.0, 1),
            "pure_ability_prob": round(pp * 100.0, 1),
            "upset_chaos_prob": round(cp * 100.0, 1),
            "ranker_prob": round(rp * 100.0, 1),
            "catboost_prob": round(cat_fp * 100.0, 1),
            "catboost_pure_prob": round(cat_pp * 100.0, 1),
            "cat_rank": int(cat_ranks[i]),
            "cat_pure_rank": int(cat_pure_ranks[i]),
            "shift1_rank": ts_info.get("shift1_rank", 6.0),
            "shift1_prize": ts_info.get("shift1_prize", 3.0),
            "rolling3_prize": ts_info.get("rolling3_prize", 3.0),
            "rolling3_rank": ts_info.get("rolling3_rank", 6.0),
            "prize_rank": int(ts_info.get("prize_rank", 8)),
            "prize_ratio": float(ts_info.get("prize_ratio", 1.0)),
            "pure_rank": int(pure_ranks[i]),
            "ens_rank": int(ens_ranks[i]),
            "standard_rank": int(std_ranks[i]),
            "chaos_rank": int(chaos_ranks[i]),
            "lgbm_boost": round(lgbm_boost, 2),
            "is_pure_value": is_pure_value,
            "is_dangerous_fav": bool(dangerous_favs[i]),
            "upset_reason": upset_reasons[i],
            "speed_index": round(float(r["_pure_si"] if prediction_mode != "standard" else r["speed_index"]), 1),
            "best_speed_index": round(float(r["best_speed_index"]), 1),
        }

    return result_map


def estimate_race_upset_risk(
    df: pd.DataFrame,
    track_condition: str = "良馬場",
    race_distance: int = 1600,
    course_type: str = "芝",
) -> dict:
    """
    出馬表のオッズ分布・脚質構成・馬場・斤量差から「このレースの荒れそう度(0〜98%)」を事前診断し、
    最適なAI予想モード ('standard' / 'upset_mild' / 'upset_chaos') を提案する。
    """
    if df is None or df.empty:
        return {
            "risk_pct": 25,
            "recommended_mode": "standard",
            "badge": "🟢 順当〜標準傾向",
            "color": "#198754",
            "reasons": ["出走馬データ標準"],
        }

    n_horses = len(df)
    risk = 28.0
    reasons = []

    # 1. 1番人気オッズと上位オッズの拮抗度
    odds_list = []
    for _, r in df.iterrows():
        v = r.get("想定単勝オッズ")
        if v is None or pd.isna(v) or float(v) <= 1.0:
            v = r.get("単勝オッズ")
        if v is not None and not pd.isna(v):
            try:
                vf = float(v)
                if vf > 1.0:
                    odds_list.append((int(r.get("単勝人気", 99)), vf, str(r.get("馬名", "")), str(r.get("脚質", "先行"))))
            except Exception:
                pass

    odds_list.sort(key=lambda x: (x[0], x[1]))
    if odds_list:
        fav1_odds = odds_list[0][1]
        fav2_odds = odds_list[1][1] if len(odds_list) >= 2 else 5.0
        if fav1_odds <= 1.8 and (fav2_odds - fav1_odds) >= 2.0:
            risk -= 26.0
            reasons.append(f"1番人気が断然({fav1_odds:.1f}倍・2番人気と{fav2_odds - fav1_odds:.1f}倍差)で鉄板軸")
        elif fav1_odds < 1.8:
            risk -= 16.0
            reasons.append(f"1番人気が断然({fav1_odds:.1f}倍)で軸堅め")
        elif fav1_odds < 2.3:
            risk -= 8.0
            reasons.append(f"1番人気({fav1_odds:.1f}倍)が上位安定")
        elif fav1_odds >= 3.5:
            risk += 26.0
            reasons.append(f"1番人気が{fav1_odds:.1f}倍の主役不在・大混戦")
        elif fav1_odds >= 2.8:
            risk += 14.0
            reasons.append(f"1番人気が{fav1_odds:.1f}倍でオッズ割れ")

        if len(odds_list) >= 3:
            fav3_odds = odds_list[2][1]
            if fav3_odds - fav1_odds <= 2.5 and fav1_odds >= 2.5:
                risk += 10.0
                reasons.append("上位人気のオッズが密集(三つ巴以上)")

    # 2. 逃げ・先行馬の頭数（ハイペース前崩れ or 人気薄単騎逃げ）
    if "脚質" in df.columns:
        esc_df = df[df["脚質"] == "逃げ"]
        lead_df = df[df["脚質"] == "先行"]
        esc_cnt = len(esc_df)
        lead_cnt = len(lead_df)
        if esc_cnt >= 4:
            risk += 26.0
            reasons.append(f"逃げ{esc_cnt}頭・先行{lead_cnt}頭が激突する超ハイペース波乱警報")
        elif esc_cnt >= 3 or (esc_cnt >= 2 and lead_cnt >= 4):
            risk += 18.0
            reasons.append(f"逃げ{esc_cnt}頭・先行{lead_cnt}頭でハイペース前崩れ警報")
        elif esc_cnt == 1:
            try:
                esc_pop = int(esc_df.iloc[0].get("単勝人気", 1))
                if esc_pop >= 5:
                    risk += 15.0
                    reasons.append(f"{esc_pop}番人気の単騎逃げ残り一発警戒")
            except Exception:
                pass

    # 3. 馬場状態・多頭数・斤量差(ハンデ戦)
    cond_s = str(track_condition)
    if "不良" in cond_s or ("重" in cond_s and "稍" not in cond_s):
        risk += 16.0
        reasons.append("重・不良の道悪馬場で紛れ大")
    elif "稍" in cond_s:
        risk += 7.0
        reasons.append("稍重馬場で消耗度アップ")

    if n_horses >= 15:
        risk += 9.0
        reasons.append(f"{n_horses}頭の多頭数で進路・枠順バイアス大")

    if int(race_distance) <= 1200:
        risk += 7.0
        reasons.append("電撃短距離戦でスタート・展開紛れやすい")

    if "斤量" in df.columns:
        try:
            w_vals = pd.to_numeric(df["斤量"], errors="coerce").dropna()
            if len(w_vals) >= 4 and (w_vals.max() - w_vals.min()) >= 3.5:
                risk += 12.0
                reasons.append(f"斤量差{w_vals.max() - w_vals.min():.1f}kg(ハンデ戦傾向)で軽斤量穴馬に注意")
        except Exception:
            pass

    risk_pct = int(round(np.clip(risk, 8.0, 96.0)))
    if risk_pct >= 65:
        rec_mode = "upset_chaos"
        badge = f"🔴 【大荒れ警報：波乱リスク {risk_pct}%】"
        color = "#dc3545"
    elif risk_pct >= 45:
        rec_mode = "upset_mild"
        badge = f"🟡 【混戦・中波乱注意：波乱リスク {risk_pct}%】"
        color = "#fd7e14"
    else:
        rec_mode = "standard"
        badge = f"🟢 【順当・堅め傾向：波乱リスク {risk_pct}%】"
        color = "#198754"

    if not reasons:
        reasons.append("極端な波乱要因は少なく能力通り決まりやすい構成")

    return {
        "risk_pct": risk_pct,
        "recommended_mode": rec_mode,
        "badge": badge,
        "color": color,
        "reasons": reasons,
    }
