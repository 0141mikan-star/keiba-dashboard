"""
pipeline_g1_adjust.py
レースの「グレード」カラムを判別し、G1などの上位クラスレース（有馬記念等）に対して
過去の重賞実績・最高スピード指数・上がり3F末脚・トップ騎手×名門厩舎ラインを重視するよう
予測ロジックを分岐・補正する処理。
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def apply_g1_adjustment(df_input, g1_boost_weight=1.35, g1_temp=0.82):
    """
    G1レース判別および専用の力関係・重賞実績補正 (高速ベクトル＋G1精密分岐)
    """
    df = df_input.copy().reset_index(drop=True)

    # 1. 全レース共通のベース正規化 (レース内勝率合計 = 1.0)
    r_sum = df.groupby("race_id", sort=False)["pred_win_prob"].transform("sum").values
    norm_base = df["pred_win_prob"].values / np.clip(r_sum, 1e-6, None)
    adjusted_arr = norm_base.copy()

    # 2. G1レースの特定
    g1_keywords = ["有馬記念", "ジャパンカップ", "日本ダービー", "東京優駿", "菊花賞", "天皇賞", "宝塚記念", "安田記念", "桜花賞", "皐月賞", "オークス", "秋華賞", "マイルCS", "スプリンターズS", "東京大賞典"]
    is_g1_mask = (df["grade"] == "G1") | df["race_name"].fillna("").apply(lambda rn: any(k in str(rn) for k in g1_keywords))
    g1_df = df[is_g1_mask]

    if len(g1_df) > 0:
        for race_id, r_df in g1_df.groupby("race_id", sort=False):
            idx_list = r_df.index.values
            base_p = norm_base[idx_list]

            # 最高スピード指数ランク & 平均スピード指数ランク
            best_si_col = "best_speed_index" if "best_speed_index" in r_df.columns else "speed_index"
            best_si_ranks = r_df[best_si_col].rank(ascending=False).values
            speed_ranks = r_df["speed_index"].rank(ascending=False).values

            # 上がり3F末脚ランク (小さい＝速い順)
            agari_ranks = r_df["last1_agari_diff"].rank(ascending=True).values if "last1_agari_diff" in r_df.columns else np.full(len(r_df), 5.0)

            jockey_rates = r_df["jockey_course_win_rate"].values if "jockey_course_win_rate" in r_df.columns else r_df["jockey_total_win_rate"].values
            trainer_rates = r_df["trainer_win_rate"].values if "trainer_win_rate" in r_df.columns else np.full(len(r_df), 0.08)
            pop_ranks = r_df["popularity"].values

            eps = 1e-6
            base_p_clipped = np.clip(base_p, eps, 1.0 - eps)
            logits = np.log(base_p_clipped / (1.0 - base_p_clipped))

            # G1専用シナジーボーナス:
            # - 最高スピード指数・平均指数上位3頭にプラス
            # - 前走上がり3F最速級 (上位3位以内) の極上末脚馬にプラス
            # - トップ騎手 & 名門厩舎にプラス
            # - G1での上位人気支持にボーナス、二桁人気の激走ノイズを抑制
            g1_bonus = np.zeros_like(logits)
            g1_bonus += np.where((speed_ranks <= 3) | (best_si_ranks <= 3), 0.25 * g1_boost_weight, 0.0)
            g1_bonus += np.where(agari_ranks <= 3, 0.18 * g1_boost_weight, 0.0)
            g1_bonus += np.where(jockey_rates >= 0.13, 0.22 * g1_boost_weight, 0.0)
            g1_bonus += np.where(trainer_rates >= 0.12, 0.15 * g1_boost_weight, 0.0)
            g1_bonus += np.where(pop_ranks <= 3, 0.28 * g1_boost_weight, 0.0)
            g1_bonus -= np.where(pop_ranks >= 10, 0.45, 0.0)

            scaled_logits = (logits + g1_bonus) / g1_temp
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
            g1_norm_p = exp_logits / np.sum(exp_logits)

            adjusted_arr[idx_list] = g1_norm_p

    df["adjusted_win_prob"] = adjusted_arr
    return df


def demonstrate_arima_kinen(test_data_path="data/test_with_preds.parquet"):
    print("=" * 65)
    print("🏇 [Step 3] 特殊条件 (有馬記念・G1) の予測補正 & 実証検証")
    print("=" * 65)

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"テストデータが見つかりません: {test_data_path}")

    df_test = pd.read_parquet(test_data_path)
    df_adjusted = apply_g1_adjustment(df_test)

    arima_races = df_adjusted[df_adjusted["race_name"].str.contains("有馬記念", na=False)]
    if len(arima_races) > 0:
        target_race_id = arima_races["race_id"].iloc[-1]
        r_date = arima_races[arima_races["race_id"] == target_race_id]["date"].iloc[0]
        sample_title = f"有馬記念 ({r_date.strftime('%Y-%m-%d')})"
    else:
        g1_races = df_adjusted[df_adjusted["grade"] == "G1"]
        target_race_id = g1_races["race_id"].iloc[0]
        sample_title = f"{g1_races['race_name'].iloc[0]} ({g1_races['date'].iloc[0].strftime('%Y-%m-%d')})"

    target_df = df_adjusted[df_adjusted["race_id"] == target_race_id].copy()
    target_df = target_df.sort_values(by="adjusted_win_prob", ascending=False).reset_index(drop=True)

    print(f"🎯 【{sample_title}】 予測補正対比検証 (出走頭数: {len(target_df)}頭)")
    print("-" * 80)
    print(f"{'予想順':<5} {'馬番':<4} {'馬名':<16} {'騎手':<8} {'単勝オッズ':<8} {'純能力率':<8} {'補正後勝率':<10} {'確定着順':<6}")
    print("-" * 80)

    for idx, row in target_df.iterrows():
        pure_p = row.get("pure_ability_prob", row["pred_win_prob"]) * 100
        a_p = row["adjusted_win_prob"] * 100
        mark = "🥇" if idx == 0 else ("🥈" if idx == 1 else ("🥉" if idx == 2 else "  "))
        act_rank = f"{int(row['rank'])}着" if row["rank"] < 90 else "外"
        print(
            f"{mark}{idx+1:2d}位  {int(row['horse_num']):2d}番  {row['horse_name']:<14}  {row['jockey_name']:<6}  "
            f"{row['odds']:5.1f}倍   {pure_p:5.1f}%   -->  {a_p:5.1f}%     {act_rank}"
        )

    print("-" * 80)
    out_path = "data/test_adjusted.parquet"
    df_adjusted.to_parquet(out_path, index=False)
    print(f"💾 G1補正済みテストデータを保存しました: {out_path}\n")
    return df_adjusted


if __name__ == "__main__":
    demonstrate_arima_kinen()
