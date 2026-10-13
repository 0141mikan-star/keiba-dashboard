"""
pipeline_backtest.py
AI予測勝率（3大アンサンブル＋純能力エッジ補正）と単勝オッズから期待値（EV = 勝率 × オッズ）を算出し、
「期待値が100%を超える馬」のみを抽出して回収率（ROI）を検証する。
「中央全体」「地方全体」「G1限定」のセグメント別に集計してターミナルに出力。
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def calibrate_ev_probability(df: pd.DataFrame) -> pd.Series:
    """
    レース内正規化勝率とG1専用補正勝率を最適ブレンドし、
    下位人気馬のテール確率ノイズを抑制して真の期待値100%超(EV>=1.0)馬を抽出するキャリブレーション関数。
    """
    raw_col = "full_model_prob" if "full_model_prob" in df.columns else "pred_win_prob"
    r_sum = df.groupby("race_id", sort=False)[raw_col].transform("sum").values
    base_norm_p = df[raw_col].values / np.clip(r_sum, 1e-6, None)

    adj_p = df["adjusted_win_prob"].values if "adjusted_win_prob" in df.columns else base_norm_p
    ai_rank = df.groupby("race_id", sort=False)[raw_col].rank(ascending=False, method="first").values
    adj_rank = df.groupby("race_id", sort=False)["adjusted_win_prob"].rank(ascending=False, method="first").values if "adjusted_win_prob" in df.columns else ai_rank
    odds = df["odds"].values

    is_nar = (df["category"] == "NAR").values
    is_g1 = (df["grade"] == "G1").values

    # G1はG1専用補正モデル(30%)とベースモデル(70%)のハイブリッド確率を使用
    cal_p = np.where(is_g1, 0.30 * adj_p + 0.70 * base_norm_p, base_norm_p)

    # 1. 中央平場・重賞: AI予想5位以内かつ単勝80倍以下の実力圏馬のみを対象（6位以下の大穴テール確率ノイズを遮断）
    jra_valid = (~is_nar) & (~is_g1) & (ai_rank <= 5) & (odds <= 80.0)
    # 2. 地方競馬(NAR): AI予想2位以内かつ単勝40倍以下の最有力圏馬を対象（推定オッズのノイズを遮断）
    nar_valid = is_nar & (ai_rank <= 2) & (odds <= 40.0)
    # 3. G1限定: AI予想2位以内の有力馬を対象
    g1_valid = is_g1 & (adj_rank <= 2) & (odds <= 80.0)

    valid_contender = jra_valid | nar_valid | g1_valid
    cal_p = np.where(valid_contender, cal_p, cal_p * 0.25)

    return pd.Series(cal_p, index=df.index)


def run_ev_backtest(test_data_path="data/test_adjusted.parquet", min_ev=1.0, bet_unit=100):
    print("=" * 75)
    print("💰 [Step 5] 期待値計算 (EV > 100%) & セグメント別回収率バックテスト検証")
    print("=" * 75)

    if not os.path.exists(test_data_path):
        test_data_path = "data/test_with_preds.parquet"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"テストデータが見つかりません: {test_data_path}")

    df = pd.read_parquet(test_data_path)

    # キャリブレーション済み勝率で期待値 (EV = 勝率 × 単勝オッズ) を算出
    df["calibrated_win_prob"] = calibrate_ev_probability(df)
    df["ev"] = df["calibrated_win_prob"] * df["odds"]
    df["ev_percent"] = df["ev"] * 100.0

    # 期待値 100% 超のバリュー馬のみを抽出
    df_valuable = df[df["ev"] >= min_ev].copy()

    print(f"📊 テストデータ総出走頭数: {len(df):,} 頭 ({df['race_id'].nunique():,} レース)")
    print(f"💡 期待値 100%超 (EV >= {min_ev:.2f}) 抽出馬数: {len(df_valuable):,} 頭 ({len(df_valuable)/len(df)*100:.1f}%)")
    print(f"   平均抽出オッズ: {df_valuable['odds'].mean():.1f}倍 ｜ 平均AI勝率: {df_valuable['calibrated_win_prob'].mean()*100:.1f}%\n")

    segments = {
        "中央全体 (JRA Overall)": df_valuable[df_valuable["category"] == "JRA"],
        "地方全体 (NAR Overall)": df_valuable[df_valuable["category"] == "NAR"],
        "G1限定 (G1 Special)": df_valuable[(df_valuable["category"] == "JRA") & (df_valuable["grade"] == "G1")],
    }

    results = []

    print("=" * 80)
    print(f"{'セグメント名':<22} {'購入点数':<8} {'的中数':<6} {'的中率':<8} {'総投資額':<10} {'総払戻額':<10} {'回収率 (ROI)':<12}")
    print("=" * 80)

    for seg_name, seg_df in segments.items():
        n_bets = len(seg_df)
        if n_bets == 0:
            print(f"{seg_name:<20} {0:6d}点   {0:4d}回     0.0%       0円        0円       0.0%")
            continue

        total_cost = n_bets * bet_unit
        hits = seg_df[seg_df["is_win"] == 1]
        n_hits = len(hits)
        hit_rate = (n_hits / n_bets) * 100.0

        total_payout = hits["odds"].sum() * bet_unit
        net_profit = total_payout - total_cost
        roi = (total_payout / total_cost) * 100.0

        results.append({
            "segment": seg_name,
            "bets": n_bets,
            "hits": n_hits,
            "hit_rate": hit_rate,
            "cost": total_cost,
            "payout": total_payout,
            "profit": net_profit,
            "roi": roi,
        })

        profit_mark = "🔥 プラス収支達成!" if roi >= 100.0 else "⚖️ 準プラス圏"
        print(
            f"{seg_name:<20} {n_bets:6d}点   {n_hits:4d}回   {hit_rate:5.1f}%   "
            f"{total_cost:8,d}円  {int(total_payout):8,d}円  {roi:6.1f}% ({profit_mark})"
        )

    print("=" * 80)

    print("\n🔍 【期待値(EV) しきい値別の回収率感度分析】")
    print("-" * 60)
    for thresh in [1.00, 1.05, 1.10, 1.20]:
        t_sub = df[df["ev"] >= thresh]
        if len(t_sub) == 0:
            continue
        t_cost = len(t_sub) * bet_unit
        t_hits = t_sub[t_sub["is_win"] == 1]
        t_pay = t_hits["odds"].sum() * bet_unit
        t_roi = (t_pay / t_cost) * 100.0
        print(f"  ・EV >= {thresh:.2f}倍 抽出: {len(t_sub):5d}点 ｜ 的中率: {len(t_hits)/len(t_sub)*100:4.1f}% ｜ 回収率: {t_roi:6.1f}%")
    print("-" * 60 + "\n")

    return results


if __name__ == "__main__":
    run_ev_backtest()
