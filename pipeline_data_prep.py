"""
pipeline_data_prep.py
中央競馬（JRA）および地方競馬（NAR）の過去データを共通フォーマットに統合し、
未活用データ（着差・上がり3F・通過順/脚質・馬体重・ローテーション・調教師・血統・コース適性・日次馬場差）を
フル活用した高度な特徴量エンジニアリングを実行して前処理済みデータおよび知識辞書を生成する。
"""

import os
import glob
import re
import sys
import pickle
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def parse_jra_time(t_val):
    if pd.isna(t_val):
        return np.nan
    t_str = str(t_val).strip()
    m = re.match(r"(\d+):(\d+\.?\d*)", t_str)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    try:
        return float(t_str)
    except Exception:
        return np.nan


def parse_nar_time(t_val):
    if pd.isna(t_val):
        return np.nan
    try:
        val = float(t_val)
        if val <= 0:
            return np.nan
        mins = int(val // 1000)
        secs = (val % 1000) / 10.0
        return mins * 60.0 + secs
    except Exception:
        return np.nan


def clean_rank(r_val):
    if pd.isna(r_val):
        return 99
    r_str = str(r_val).strip()
    digits = re.findall(r"\d+", r_str)
    if digits:
        return int(digits[0])
    return 99


def determine_grade(race_name, listed_val):
    rn = str(race_name) if not pd.isna(race_name) else ""
    lv = str(listed_val) if not pd.isna(listed_val) else ""

    if (
        "G1" in lv
        or "J.G1" in lv
        or "Jpn1" in lv
        or any(
            k in rn
            for k in [
                "有馬記念",
                "日本ダービー",
                "東京優駿",
                "ジャパンカップ",
                "菊花賞",
                "天皇賞",
                "宝塚記念",
                "安田記念",
                "桜花賞",
                "皐月賞",
                "優駿牝馬",
                "オークス",
                "秋華賞",
                "スプリンターズS",
                "マイルCS",
                "エリザベス女王杯",
                "チャンピオンズC",
                "フェブラリーS",
                "高松宮記念",
                "大阪杯",
                "ヴィクトリアマイル",
                "NHKマイルC",
                "ホープフルS",
                "東京大賞典",
            ]
        )
    ):
        return "G1"
    if "G2" in lv or "J.G2" in lv or "Jpn2" in lv:
        return "G2"
    if "G3" in lv or "J.G3" in lv or "Jpn3" in lv:
        return "G3"
    if "L" in lv or "リステッド" in rn or "特別" in rn:
        return "L"
    if "重賞" in rn or "記念" in rn or "盃" in rn or "賞" in rn:
        return "重賞"
    if "オープン" in rn or "OP" in rn:
        return "OP"
    return "一般"


def parse_record_series(series: pd.Series, prior_win=0.25, prior_top3=0.75, prior_n=3.0):
    """'15-9-4-71' のような着度数文字列から (win_rate, top3_rate, total_starts) を高速ベクトル算出"""
    extracted = series.fillna("0-0-0-0").astype(str).str.extract(r"(\d+)-(\d+)-(\d+)-(\d+)")
    w = pd.to_numeric(extracted[0], errors="coerce").fillna(0.0).values
    p = pd.to_numeric(extracted[1], errors="coerce").fillna(0.0).values
    s = pd.to_numeric(extracted[2], errors="coerce").fillna(0.0).values
    l = pd.to_numeric(extracted[3], errors="coerce").fillna(0.0).values
    tot = w + p + s + l
    win_rate = (w + prior_win) / (tot + prior_n)
    top3_rate = (w + p + s + prior_top3) / (tot + prior_n)
    return win_rate, top3_rate, tot


def load_and_preprocess(data_dir="data", jra_years_start="2017", force_rebuild=False):
    print("=" * 65, flush=True)
    print("🚀 [Step 1] データ統合 & 高精度特徴量エンジニアリング (JRA + NAR)", flush=True)
    print("=" * 65, flush=True)

    out_path = os.path.join(data_dir, "integrated_dataset.parquet")
    required_new_cols = [
        "best_speed_index", "last1_speed_index", "last1_margin_sec",
        "last1_agari_diff", "trainer_win_rate", "course_aptitude_rate", "market_win_prob"
    ]
    if os.path.exists(out_path) and not force_rebuild:
        print(f"⚡ 既存統合データセットを読み込み中: {out_path}", flush=True)
        df_all = pd.read_parquet(out_path)
        if all(c in df_all.columns for c in required_new_cols) and os.path.exists("models/bigdata_knowledge.pkl"):
            print(f"✅ 最新版特徴量データセット確認完了 ({len(df_all):,} 行 / {len(df_all.columns)} カラム)", flush=True)
            return df_all

        print(f"🛠️ 統合ベースデータ ({len(df_all):,} 行) にNAR詳細戦績と拡張特徴量を追加します...", flush=True)
        df_all["horse_name"] = df_all["horse_name"].fillna("").astype(str).str.strip()
        df_all["jockey_name"] = df_all["jockey_name"].fillna("未定").astype(str).str.strip()
        df_all["is_top2"] = (df_all["rank"] <= 2).astype(int)
        df_all["is_top3"] = (df_all["rank"] <= 3).astype(int)

        # NARのhorselistから事前確定戦績・調教師・上がり3F・馬体重を高速マージ
        nar_base = os.path.join(data_dir, "nar")
        if not os.path.exists(nar_base):
            nar_base = r"c:\Users\Yua11\OneDrive - izumi\izumi-y\antigravity\競馬データ　地方"
        horse_files = sorted(glob.glob(os.path.join(nar_base, "*_race", "*_horselist.csv")))
        if horse_files and "nar_pre_win_rate" not in df_all.columns:
            nar_h_cols = ["競馬場", "競走年月日", "レース番号", "馬番", "調教師", "父馬名", "馬体重", "馬体重増減", "全成績", "当競馬場成績", "うち当距離成績", "騎手成績", "上がり3F"]
            df_nh = pd.concat([pd.read_csv(hf, usecols=lambda c: c in nar_h_cols, encoding="utf-8", low_memory=False) for hf in horse_files], ignore_index=True)
            df_nh["race_id"] = "NAR_" + df_nh["競馬場"].astype(str) + "_" + df_nh["競走年月日"].astype(str) + "_" + df_nh["レース番号"].astype(str)
            df_nh["horse_num"] = pd.to_numeric(df_nh["馬番"], errors="coerce").fillna(1).astype(int)
            df_nh = df_nh.drop_duplicates(subset=["race_id", "horse_num"])

            nar_wr, nar_t3, _ = parse_record_series(df_nh.get("全成績", pd.Series(dtype=str)))
            _, nar_c_t3, _ = parse_record_series(df_nh.get("当競馬場成績", pd.Series(dtype=str)))
            _, nar_d_t3, _ = parse_record_series(df_nh.get("うち当距離成績", pd.Series(dtype=str)))
            _, nar_jh_t3, _ = parse_record_series(df_nh.get("騎手成績", pd.Series(dtype=str)))

            df_nh_sub = pd.DataFrame({
                "race_id": df_nh["race_id"],
                "horse_num": df_nh["horse_num"],
                "trainer_name_nar": df_nh["調教師"].fillna("未定").astype(str).str.strip(),
                "sire_name_nar": df_nh["父馬名"].fillna("UNKNOWN").astype(str).str.strip(),
                "agari_3f_nar": pd.to_numeric(df_nh["上がり3F"], errors="coerce"),
                "horse_weight_nar": pd.to_numeric(df_nh["馬体重"], errors="coerce"),
                "weight_change_nar": pd.to_numeric(df_nh["馬体重増減"], errors="coerce"),
                "nar_pre_win_rate": nar_wr,
                "nar_pre_top3_rate": nar_t3,
                "nar_course_top3_rate": nar_c_t3,
                "nar_dist_top3_rate": nar_d_t3,
                "nar_jockey_horse_top3_rate": nar_jh_t3,
            })
            df_all = pd.merge(df_all, df_nh_sub, on=["race_id", "horse_num"], how="left")
            df_all["trainer_name"] = df_all["trainer_name_nar"].fillna(df_all["jockey_name"])
            df_all["sire_name"] = df_all["sire_name_nar"].fillna("UNKNOWN")
            df_all["agari_3f"] = df_all["agari_3f_nar"]
            df_all["horse_weight"] = df_all["horse_weight_nar"].fillna(470.0)
            df_all["weight_change"] = df_all["weight_change_nar"].fillna(0.0)
            df_all.drop(columns=["trainer_name_nar", "sire_name_nar", "agari_3f_nar", "horse_weight_nar", "weight_change_nar"], inplace=True)

        for col, default_v in [
            ("trainer_name", df_all["jockey_name"]),
            ("sire_name", "UNKNOWN"),
            ("agari_3f", np.nan),
            ("horse_weight", 470.0),
            ("weight_change", 0.0),
            ("corner_pos_raw", np.nan),
            ("nar_pre_win_rate", np.nan),
            ("nar_pre_top3_rate", np.nan),
            ("nar_course_top3_rate", np.nan),
            ("nar_dist_top3_rate", np.nan),
            ("nar_jockey_horse_top3_rate", np.nan),
        ]:
            if col not in df_all.columns:
                df_all[col] = default_v
        df_all = df_all.sort_values(by=["date", "race_id", "horse_num"]).reset_index(drop=True)

    if not os.path.exists(out_path) or force_rebuild:
        df_all = _build_raw_integrated_df(data_dir, jra_years_start)

    # 同一馬名でも中央・地方・ばんえいで別馬が混ざらないようユニーク馬キーを作成
    is_banei_all = df_all["track_name"].str.contains("帯広")
    df_all["horse_key"] = np.where(is_banei_all, "BANEI_", df_all["category"] + "_") + df_all["horse_name"]

    # レース頭数
    df_all["field_size"] = df_all.groupby("race_id")["horse_num"].transform("count").clip(lower=2)

    # ==========================================================
    # 4. 高度な特徴量エンジニアリング (リーク完全防止)
    # ==========================================================
    print("🛠️ [特徴量 1/6] 日次馬場差を補正した『真のスピード指数』とピーク能力値・近走指数を算出中...", flush=True)
    course_key = df_all["track_name"] + "_" + df_all["course_type"] + "_" + df_all["distance"].astype(str)
    df_all["course_key"] = course_key

    for c_drop in ["std_time_mean", "std_time_sd"]:
        if c_drop in df_all.columns:
            df_all.drop(columns=[c_drop], inplace=True)

    valid_times = df_all[df_all["time_sec"].notna() & (df_all["time_sec"] > 10.0)]
    course_stats = valid_times.groupby("course_key")["time_sec"].agg(["mean", "std"]).reset_index()
    course_stats.rename(columns={"mean": "std_time_mean", "std": "std_time_sd"}, inplace=True)

    df_all = pd.merge(df_all, course_stats, on="course_key", how="left")
    df_all["std_time_sd"] = df_all["std_time_sd"].fillna(2.0).clip(lower=0.5)
    raw_speed_idx = 100.0 - ((df_all["time_sec"] - df_all["std_time_mean"]) / df_all["std_time_sd"]) * 10.0
    df_all["raw_speed_index"] = raw_speed_idx.fillna(95.0).clip(50.0, 140.0)

    # 日次馬場差 (同日・同競馬場・同コース種別の平均スピード指数偏差)
    day_surface_key = df_all["date"].astype(str) + "_" + df_all["track_name"] + "_" + df_all["course_type"]
    daily_mean_si = df_all.groupby(day_surface_key)["raw_speed_index"].transform("mean")
    daily_track_bias = (daily_mean_si - 100.0).clip(-12.0, 12.0)
    df_all["true_raw_si"] = (df_all["raw_speed_index"] - daily_track_bias * 0.6).clip(50.0, 145.0)

    # 過去走スピード指数 (通算平均・自己ベスト・前走・直近3走平均) - リーク防止
    h_grp = df_all.groupby("horse_key", sort=False)
    cum_cnt = h_grp.cumcount()
    df_all["past_starts_count"] = cum_cnt

    cum_sum_si = h_grp["true_raw_si"].cumsum() - df_all["true_raw_si"]
    df_all["speed_index"] = np.where(cum_cnt > 0, cum_sum_si / cum_cnt, 95.0)

    s1_si = h_grp["true_raw_si"].shift(1)
    s2_si = h_grp["true_raw_si"].shift(2)
    s3_si = h_grp["true_raw_si"].shift(3)

    df_all["last1_speed_index"] = s1_si.fillna(df_all["speed_index"])
    df_all["last3_speed_index"] = pd.concat([s1_si, s2_si, s3_si], axis=1).mean(axis=1).fillna(df_all["speed_index"])
    df_all["best_speed_index"] = s1_si.groupby(df_all["horse_key"], sort=False).cummax().fillna(df_all["speed_index"])

    print("🛠️ [特徴量 2/6] 前走・過去走の『着差(秒差)』『上がり3F末脚指数』『過去勝率・複勝率』を算出中...")
    # 各レースの1着タイムとの秒差 (着差秒)
    race_min_time = df_all.groupby("race_id")["time_sec"].transform("min")
    raw_margin_sec = (df_all["time_sec"] - race_min_time).clip(0.0, 8.0)
    # タイム欠損時は着順から近似秒差を補完
    df_all["raw_margin_sec"] = raw_margin_sec.fillna(((df_all["rank"] - 1) * 0.25).clip(0.0, 6.0))

    df_all["last1_margin_sec"] = h_grp["raw_margin_sec"].shift(1).fillna(1.2)
    cum_margin = h_grp["raw_margin_sec"].cumsum() - df_all["raw_margin_sec"]
    df_all["past_avg_margin"] = np.where(cum_cnt > 0, cum_margin / cum_cnt, 1.2)

    df_all["last1_rank"] = h_grp["rank"].shift(1).fillna(6.0).clip(1.0, 18.0)
    r1 = h_grp["rank"].shift(1)
    r2 = h_grp["rank"].shift(2)
    r3 = h_grp["rank"].shift(3)
    df_all["last3_avg_rank"] = pd.concat([r1, r2, r3], axis=1).mean(axis=1).fillna(6.0).clip(1.0, 18.0)

    # 過去勝率・複勝率 (NARは事前全成績と統合して初回出走から高精度化)
    cum_wins = h_grp["is_win"].cumsum() - df_all["is_win"]
    cum_top3 = h_grp["is_top3"].cumsum() - df_all["is_top3"]
    calc_win_rate = (cum_wins + 0.25) / (cum_cnt + 3.0)
    calc_top3_rate = (cum_top3 + 0.75) / (cum_cnt + 3.0)

    df_all["past_win_rate"] = np.where(
        df_all["nar_pre_win_rate"].notna(),
        0.65 * df_all["nar_pre_win_rate"] + 0.35 * calc_win_rate,
        calc_win_rate,
    )
    df_all["past_top3_rate"] = np.where(
        df_all["nar_pre_top3_rate"].notna(),
        0.65 * df_all["nar_pre_top3_rate"] + 0.35 * calc_top3_rate,
        calc_top3_rate,
    )

    # 上がり3Fのレース内相対差 (負の値ほどメンバー中で速い極上の末脚)
    race_mean_agari = df_all.groupby("race_id")["agari_3f"].transform("mean")
    raw_agari_diff = (df_all["agari_3f"] - race_mean_agari).clip(-3.5, 4.5).fillna(0.0)
    df_all["raw_agari_diff"] = raw_agari_diff
    df_all["last1_agari_diff"] = h_grp["raw_agari_diff"].shift(1).fillna(0.0)
    cum_agari = h_grp["raw_agari_diff"].cumsum() - df_all["raw_agari_diff"]
    df_all["past_agari_diff_mean"] = np.where(cum_cnt > 0, cum_agari / cum_cnt, 0.0)

    print("🛠️ [特徴量 3/6] 脚質・コーナー通過位置取り & レース内展開(先行馬数・ペース圧力)を算出中...")
    # コーナー通過順の正規化 (0.0=先頭逃げ 〜 1.0=最後方追込)
    norm_corner = ((df_all["corner_pos_raw"] - 1.0) / (df_all["field_size"] - 1.0)).clip(0.0, 1.0)
    # コーナー通過順が欠損している場合は、上がり差と着順から位置取りを推計
    est_corner = np.where(df_all["raw_agari_diff"] < -0.5, 0.65, np.where(df_all["rank"] <= 3, 0.28, 0.50))
    df_all["norm_corner_pos"] = norm_corner.fillna(pd.Series(est_corner, index=df_all.index))

    cum_corner = h_grp["norm_corner_pos"].cumsum() - df_all["norm_corner_pos"]
    df_all["past_corner_pos"] = np.where(cum_cnt > 0, cum_corner / cum_cnt, 0.45)

    # レース内の逃げ・先行馬頭数 (past_corner_pos < 0.30 の馬の頭数)
    df_all["_is_front_type"] = ((cum_cnt > 0) & (df_all["past_corner_pos"] < 0.30)).astype(int)
    df_all["race_front_runners"] = df_all.groupby("race_id")["_is_front_type"].transform("sum")
    df_all["pace_pressure"] = df_all["race_front_runners"] / df_all["field_size"]
    df_all.drop(columns=["_is_front_type"], inplace=True)

    print("🛠️ [特徴量 4/6] コース適性・距離適性・ローテーション(間隔・距離変更・馬体重)を算出中...")
    # 当該競馬場での過去複勝率
    hc_key = df_all["horse_key"] + "_" + df_all["track_name"]
    hc_grp = df_all.groupby(hc_key, sort=False)
    hc_cnt = hc_grp.cumcount()
    hc_top3 = hc_grp["is_top3"].cumsum() - df_all["is_top3"]
    calc_course_apt = (hc_top3 + 0.5) / (hc_cnt + 2.0)
    df_all["course_aptitude_rate"] = np.where(
        df_all["nar_course_top3_rate"].notna(),
        0.7 * df_all["nar_course_top3_rate"] + 0.3 * calc_course_apt,
        calc_course_apt,
    )

    # 当該距離帯での過去複勝率
    dist_band = (df_all["distance"] // 200).astype(str)
    hd_key = df_all["horse_key"] + "_" + dist_band
    hd_grp = df_all.groupby(hd_key, sort=False)
    hd_cnt = hd_grp.cumcount()
    hd_top3 = hd_grp["is_top3"].cumsum() - df_all["is_top3"]
    calc_dist_apt = (hd_top3 + 0.5) / (hd_cnt + 2.0)
    df_all["dist_aptitude_rate"] = np.where(
        df_all["nar_dist_top3_rate"].notna(),
        0.7 * df_all["nar_dist_top3_rate"] + 0.3 * calc_dist_apt,
        calc_dist_apt,
    )

    # ローテーション（前走からの間隔日数 & 距離増減）
    prev_date = h_grp["date"].shift(1)
    df_all["rest_days"] = (df_all["date"] - prev_date).dt.days.fillna(28.0).clip(5.0, 365.0)
    prev_dist = h_grp["distance"].shift(1)
    df_all["dist_change"] = (df_all["distance"] - prev_dist).fillna(0.0).clip(-1200.0, 1200.0)

    print("🛠️ [特徴量 5/6] 騎手・調教師(厩舎)・騎手×厩舎勝負ライン・血統のリーク防止勝率を算出中...")
    # (1) 騎手総合勝率 & 複勝率 (時系列累積ベイズ平滑化)
    j_grp = df_all.groupby("jockey_name", sort=False)
    j_cnt = j_grp.cumcount()
    j_wins = j_grp["is_win"].cumsum() - df_all["is_win"]
    j_top3 = j_grp["is_top3"].cumsum() - df_all["is_top3"]
    df_all["jockey_total_win_rate"] = (j_wins + 0.8) / (j_cnt + 10.0)
    df_all["jockey_top3_rate"] = (j_top3 + 2.4) / (j_cnt + 10.0)

    # (2) 騎手×競馬場コース別勝率
    jc_key = df_all["jockey_name"] + "_" + df_all["track_name"]
    df_all["jockey_course_key"] = jc_key
    jc_grp = df_all.groupby("jockey_course_key", sort=False)
    jc_cnt = jc_grp.cumcount()
    jc_wins = jc_grp["is_win"].cumsum() - df_all["is_win"]
    df_all["jockey_course_win_rate"] = (jc_wins + df_all["jockey_total_win_rate"] * 5.0) / (jc_cnt + 5.0)

    # (3) 調教師(厩舎)勝率 & 複勝率
    t_grp = df_all.groupby("trainer_name", sort=False)
    t_cnt = t_grp.cumcount()
    t_wins = t_grp["is_win"].cumsum() - df_all["is_win"]
    t_top3 = t_grp["is_top3"].cumsum() - df_all["is_top3"]
    df_all["trainer_win_rate"] = (t_wins + 0.8) / (t_cnt + 10.0)
    df_all["trainer_top3_rate"] = (t_top3 + 2.4) / (t_cnt + 10.0)

    # (4) 騎手×調教師の勝負ライン複勝率
    jt_key = df_all["jockey_name"] + "_" + df_all["trainer_name"]
    jt_grp = df_all.groupby(jt_key, sort=False)
    jt_cnt = jt_grp.cumcount()
    jt_top3 = jt_grp["is_top3"].cumsum() - df_all["is_top3"]
    df_all["jockey_trainer_top3_rate"] = (jt_top3 + df_all["jockey_top3_rate"] * 4.0) / (jt_cnt + 4.0)

    # NARの騎手×当該馬コンビ実績があればブレンド
    df_all["jockey_horse_compat"] = np.where(
        df_all["nar_jockey_horse_top3_rate"].notna(),
        0.6 * df_all["nar_jockey_horse_top3_rate"] + 0.4 * df_all["jockey_trainer_top3_rate"],
        df_all["jockey_trainer_top3_rate"],
    )

    print("🛠️ [特徴量 6/6] レース内相対偏差値 (能力Zスコア・騎手Zスコア・相対斤量・市場支持率) を算出中...")
    # レース内相対特徴量（同じレースに出走するメンバーの中での相対的な立ち位置！）
    r_grp = df_all.groupby("race_id", sort=False)

    race_si_mean = r_grp["speed_index"].transform("mean")
    race_si_std = r_grp["speed_index"].transform("std").fillna(3.0).clip(lower=1.0)
    df_all["race_rel_speed_index"] = (df_all["speed_index"] - race_si_mean) / race_si_std

    race_best_si_mean = r_grp["best_speed_index"].transform("mean")
    df_all["race_rel_best_si"] = df_all["best_speed_index"] - race_best_si_mean

    race_j_mean = r_grp["jockey_course_win_rate"].transform("mean")
    df_all["race_rel_jockey_rate"] = df_all["jockey_course_win_rate"] - race_j_mean

    race_w_mean = r_grp["weight_carried"].transform("mean")
    df_all["weight_diff_from_avg"] = (df_all["weight_carried"] - race_w_mean).fillna(0.0)

    # 対数単勝オッズ & レース内インプライド市場勝率 (完全ベクトル化)
    df_all["log_odds"] = np.log1p(df_all["odds"].clip(lower=1.0))
    df_all["_inv_odds"] = 1.0 / df_all["odds"].clip(lower=1.05)
    sum_inv_odds = df_all.groupby("race_id", sort=False)["_inv_odds"].transform("sum")
    df_all["market_win_prob"] = (df_all["_inv_odds"] / sum_inv_odds.clip(lower=0.1)).clip(0.001, 0.95)
    df_all.drop(columns=["_inv_odds"], inplace=True)

    # カテゴリ変数のエンコーディング
    cat_cols = ["track_name", "course_type", "track_condition", "gender", "grade", "category"]
    cat_mappings = {}
    for c in cat_cols:
        cat_s = df_all[c].astype("category")
        df_all[f"{c}_code"] = cat_s.cat.codes
        cat_mappings[c] = {val: code for code, val in enumerate(cat_s.cat.categories)}

    # 数値欠損の最終クリーンアップ
    fill_zero_cols = [
        "race_rel_speed_index", "race_rel_best_si", "race_rel_jockey_rate",
        "weight_diff_from_avg", "last1_agari_diff", "past_agari_diff_mean", "dist_change"
    ]
    for c in fill_zero_cols:
        df_all[c] = df_all[c].fillna(0.0)

    # 保存
    os.makedirs(data_dir, exist_ok=True)
    df_all.to_parquet(out_path, index=False)
    print(f"✅ 前処理済み統合データセットを保存しました: {out_path} ({len(df_all):,} 行 / {len(df_all.columns)} カラム)")

    # ==========================================================
    # 5. アプリ本体 (app.py / win5_simulator.py) 連携用の統計知識辞書を保存
    # ==========================================================
    os.makedirs("models", exist_ok=True)
    # 騎手総合勝率辞書 (%)
    j_final = df_all.groupby("jockey_name")["is_win"].agg(["count", "mean"]).reset_index()
    j_final["win_pct"] = ((j_final["mean"] * j_final["count"] + 0.8) / (j_final["count"] + 10.0)) * 100.0
    jockey_win_pct_map = dict(zip(j_final["jockey_name"], np.round(j_final["win_pct"], 2)))

    # 騎手×競馬場勝率辞書 (%)
    jc_final = df_all.groupby(["jockey_name", "track_name"])["is_win"].agg(["count", "mean"]).reset_index()
    jc_final = jc_final[jc_final["count"] >= 3].copy()
    jc_final["j_base"] = jc_final["jockey_name"].map(lambda x: jockey_win_pct_map.get(x, 8.0) / 100.0)
    jc_final["win_pct"] = np.round(((jc_final["mean"] * jc_final["count"] + jc_final["j_base"] * 5.0) / (jc_final["count"] + 5.0)) * 100.0, 2)
    jc_keys = jc_final["jockey_name"] + "_" + jc_final["track_name"]
    jockey_course_pct_map = dict(zip(jc_keys, jc_final["win_pct"]))

    # 調教師勝率辞書 (%)
    t_final = df_all.groupby("trainer_name")["is_win"].agg(["count", "mean"]).reset_index()
    t_final["win_pct"] = np.round(((t_final["mean"] * t_final["count"] + 0.8) / (t_final["count"] + 10.0)) * 100.0, 2)
    trainer_win_pct_map = dict(zip(t_final["trainer_name"], t_final["win_pct"]))

    # 競走馬の最新能力プロファイル辞書 (高速ベクトル構築)
    latest_horses = df_all[df_all["horse_name"] != ""].drop_duplicates(subset=["horse_name"], keep="last").copy()
    new_cnt_arr = latest_horses["past_starts_count"].values + 1
    new_si_arr = np.round((latest_horses["speed_index"].values * latest_horses["past_starts_count"].values + latest_horses["true_raw_si"].values) / new_cnt_arr, 1)
    new_best_si_arr = np.round(np.maximum(latest_horses["best_speed_index"].values, latest_horses["true_raw_si"].values), 1)

    horse_profile_map = {}
    for h_name, si, b_si, l_si, l_rk, l_mg, ag_d, wr, t3r, c_pos, st_c in zip(
        latest_horses["horse_name"].values,
        new_si_arr,
        new_best_si_arr,
        np.round(latest_horses["true_raw_si"].values, 1),
        latest_horses["rank"].values.astype(int),
        np.round(latest_horses["raw_margin_sec"].values, 2),
        np.round(latest_horses["raw_agari_diff"].values, 2),
        np.round(latest_horses["past_win_rate"].values, 3),
        np.round(latest_horses["past_top3_rate"].values, 3),
        np.round(latest_horses["past_corner_pos"].values, 2),
        new_cnt_arr.astype(int),
    ):
        horse_profile_map[str(h_name)] = {
            "speed_index": float(si),
            "best_speed_index": float(b_si),
            "last_si": float(l_si),
            "last_rank": int(l_rk),
            "last_margin": float(l_mg),
            "agari_diff": float(ag_d),
            "win_rate": float(wr),
            "top3_rate": float(t3r),
            "corner_pos": float(c_pos),
            "starts": int(st_c),
        }

    knowledge_pack = {
        "jockey_win_pct_map": jockey_win_pct_map,
        "jockey_course_pct_map": jockey_course_pct_map,
        "trainer_win_pct_map": trainer_win_pct_map,
        "horse_profile_map": horse_profile_map,
        "course_stats": dict(zip(course_stats["course_key"], zip(course_stats["std_time_mean"], course_stats["std_time_sd"]))),
        "cat_mappings": cat_mappings,
        "total_records": len(df_all),
    }
    kb_path = "models/bigdata_knowledge.pkl"
    with open(kb_path, "wb") as f:
        pickle.dump(knowledge_pack, f)
    print(f"🧠 ビッグデータ知識辞書を保存しました: {kb_path} (登録馬: {len(horse_profile_map):,}頭 / 騎手: {len(jockey_win_pct_map):,}名)")
    print("=" * 65)
    return df_all


if __name__ == "__main__":
    load_and_preprocess(force_rebuild=False)
