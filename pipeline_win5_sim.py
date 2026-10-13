"""
pipeline_win5_sim.py
任意の5レースを対象として、各馬の予測勝率を元にモンテカルロ法で1万回シミュレーションを行い、
発生確率が高い買い目の組み合わせTop 10を出力するWIN5予測シミュレーター。
"""

import os
import sys
from collections import Counter
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def simulate_win5(races_data, n_simulations=10000, random_seed=42):
    """
    モンテカルロ法によるWIN5シミュレーション
    :param races_data: 5レース分の出走馬リスト [{'race_name': str, 'horses': [{'horse_num': int, 'horse_name': str, 'prob': float, 'odds': float}]}]
    :param n_simulations: シミュレーション試行回数 (デフォルト: 10,000回)
    """
    np.random.seed(random_seed)
    assert len(races_data) == 5, "WIN5はちょうど5レースである必要があります。"

    print("=" * 65)
    print(f"🎲 [Step 4] WIN5 モンテカルロ・シミュレーター ({n_simulations:,}回試行)")
    print("=" * 65)

    # 1. 各レースの概要表示
    for i, r in enumerate(races_data):
        horses = r["horses"]
        top_h = sorted(horses, key=lambda x: x["prob"], reverse=True)[0]
        print(f"🏁 WIN{i+1}: {r['race_name']:<22} ({len(horses):2d}頭) ｜ 最有力: {top_h['horse_num']}番 {top_h['horse_name']} (勝率 {top_h['prob']*100:.1f}%)")

    # 2. モンテカルロ・サンプリング
    sim_winners = []
    for r in races_data:
        horses = r["horses"]
        h_nums = [h["horse_num"] for h in horses]
        probs = np.array([h["prob"] for h in horses], dtype=float)
        probs = probs / np.sum(probs)  # 念のため正規化

        # 1万回の勝馬を同時サンプリング
        samples = np.random.choice(h_nums, size=n_simulations, p=probs)
        sim_winners.append(samples)

    sim_winners = np.array(sim_winners)  # shape: (5, n_simulations)

    # 3. 買い目組み合わせの集計
    combos = [tuple(sim_winners[:, sim_i]) for sim_i in range(n_simulations)]
    combo_counts = Counter(combos)

    # 4. 全組み合わせの理論確率計算 (Cartesian Product から Top 10 を探索)
    # 理論確率 = p1 * p2 * p3 * p4 * p5
    all_theoretical = []
    
    # 探索空間を絞るため各レースの上位有力馬（上位5頭）の積を全網羅
    r_top_horses = [sorted(r["horses"], key=lambda x: x["prob"], reverse=True)[:5] for r in races_data]
    
    import itertools
    for c in itertools.product(*r_top_horses):
        ticket_nums = tuple(h["horse_num"] for h in c)
        ticket_names = tuple(h["horse_name"] for h in c)
        prob = np.prod([h["prob"] for h in c])
        est_payout = (1.0 / prob * 0.70) * 100 if prob > 0 else 0  # 控除率30%想定配当
        
        sim_cnt = combo_counts.get(ticket_nums, 0)
        sim_pct = (sim_cnt / n_simulations) * 100.0

        all_theoretical.append({
            "ticket_nums": ticket_nums,
            "ticket_names": ticket_names,
            "theory_prob": prob,
            "theory_pct": prob * 100.0,
            "sim_count": sim_cnt,
            "sim_pct": sim_pct,
            "est_payout": est_payout
        })

    df_combos = pd.DataFrame(all_theoretical).sort_values(by="theory_prob", ascending=False).reset_index(drop=True)
    top10 = df_combos.head(10)

    print("\n" + "=" * 65)
    print("🏆 【WIN5 発生確率が最も高い買い目 Top 10】")
    print("=" * 65)
    print(f"{'順位':<4} {'買い目 (WIN1〜5 馬番)':<22} {'理論的中確率':<12} {'1万回出現数':<12} {'推定配当金':<14}")
    print("-" * 65)

    for idx, row in top10.iterrows():
        nums_str = "-".join(f"{n:02d}" for n in row["ticket_nums"])
        print(f" {idx+1:2d}位  [{nums_str}]      {row['theory_pct']:6.3f}%        {row['sim_count']:4d}回 ({row['sim_pct']:4.2f}%)   約 {row['est_payout']:,.0f} 円")

    print("-" * 65)
    best_ticket = top10.iloc[0]
    print(f"🥇 AI推奨 本命1点買い目: {best_ticket['ticket_nums']}")
    for i in range(5):
        print(f"   WIN{i+1}: {best_ticket['ticket_nums'][i]}番 {best_ticket['ticket_names'][i]}")
    print(f"   ➡ 5連勝的中期待確率: {best_ticket['theory_pct']:.3f}% ｜ シミュレーション出現: {best_ticket['sim_count']}回/{n_simulations:,}回")
    print("=" * 65 + "\n")

    return top10

def run_win5_demo(test_data_path="data/test_adjusted.parquet"):
    if not os.path.exists(test_data_path):
        test_data_path = "data/test_with_preds.parquet"
    if not os.path.exists(test_data_path):
        raise FileNotFoundError("テストデータが見つかりません。")

    df = pd.read_parquet(test_data_path)
    prob_col = "adjusted_win_prob" if "adjusted_win_prob" in df.columns else "pred_win_prob"

    # テストセットからJRAのWIN5対象帯（10R〜11R、頭数8頭以上）の5レースを選択
    jra_main = df[(df["category"] == "JRA") & (df["race_num"].isin([10, 11]))]
    race_counts = jra_main.groupby("race_id").size()
    valid_rids = race_counts[race_counts >= 8].index.tolist()
    if len(valid_rids) >= 5:
        sample_races = valid_rids[:5]
    else:
        sample_races = df["race_id"].unique()[:5]

    races_data = []
    for rid in sample_races:
        r_df = df[df["race_id"] == rid].copy()
        # 勝率正規化
        p_vals = r_df[prob_col].values
        p_norm = p_vals / np.sum(p_vals) if np.sum(p_vals) > 0 else np.ones_like(p_vals)/len(p_vals)
        r_df["norm_p"] = p_norm

        horses_list = []
        for _, h in r_df.iterrows():
            horses_list.append({
                "horse_num": int(h["horse_num"]),
                "horse_name": str(h["horse_name"]),
                "prob": float(h["norm_p"]),
                "odds": float(h["odds"])
            })

        races_data.append({
            "race_name": str(r_df["race_name"].iloc[0]),
            "horses": horses_list
        })

    res_top10 = simulate_win5(races_data)
    evaluate_win5_historical_backtest(df)
    return res_top10


def evaluate_win5_historical_backtest(df: pd.DataFrame):
    """テストセット内の全JRA WIN5対象開催日(86日・430レース)における的中率・回収率バックテスト"""
    import itertools

    jra = df[(df["category"] == "JRA") & (df["race_num"].isin([10, 11]))].copy()
    if jra.empty:
        return

    jra["ai_rank"] = jra.groupby("race_id")["adjusted_win_prob"].rank(ascending=False, method="first")
    jra["pop_rank"] = jra.groupby("race_id")["popularity"].rank(ascending=True, method="first")

    win5_days = []
    for dt, grp in jra.groupby("date"):
        rids = sorted(grp["race_id"].unique())
        if len(rids) >= 5:
            win5_days.append((dt, rids[-5:]))

    if not win5_days:
        return

    print("=" * 78)
    print(f"📊 【WIN5 過去全{len(win5_days)}開催日 ({len(win5_days)*5}レース) 実戦バックテスト検証】")
    print("=" * 78)

    leg_ai_ranks, leg_pop_ranks = [], []
    box_max_ai, box_max_pop = [], []

    for dt, rids in win5_days:
        r_ai, r_pop = [], []
        for rid in rids:
            r_df = jra[jra["race_id"] == rid]
            w = r_df[r_df["is_win"] == 1]
            if len(w) == 0:
                break
            r_ai.append(int(w.iloc[0]["ai_rank"]))
            r_pop.append(int(w.iloc[0]["pop_rank"]))
        if len(r_ai) == 5:
            leg_ai_ranks.extend(r_ai)
            leg_pop_ranks.extend(r_pop)
            box_max_ai.append(max(r_ai))
            box_max_pop.append(max(r_pop))

    leg_ai_arr = np.array(leg_ai_ranks)
    leg_pop_arr = np.array(leg_pop_ranks)
    print("① 単一レース (WIN5対象10R・11R) 1着馬捕捉率:")
    for k in [1, 2, 3, 5]:
        print(f"   ・上位{k}頭以内捕捉: 3大アンサンブルAI {np.mean(leg_ai_arr<=k)*100:5.1f}%  vs  単勝人気順 {np.mean(leg_pop_arr<=k)*100:5.1f}%")

    print("\n② 均等ボックス買い vs AI確率積上位抽出 (3大アンサンブル + 純能力AIブレンド):")
    print("-" * 78)
    print(f"{'買い方・予算点数':<26} {'AI的中回数 (的中率)':<22} {'AI回収率(ROI)':<15} {'人気順ROI':<12}")
    print("-" * 78)

    # 均等3頭ボックス (243点) の比較
    box3_ai_hits = int(np.sum(np.array(box_max_ai) <= 3))
    box3_pop_hits = int(np.sum(np.array(box_max_pop) <= 3))
    print(f"{'均等3頭BOX (3^5 = 243点)':<26} {box3_ai_hits:2d}/{len(win5_days)}回 ({box3_ai_hits/len(win5_days)*100:4.1f}%)        {'(均等配分)':<15} {box3_pop_hits:2d}/{len(win5_days)}回 ({box3_pop_hits/len(win5_days)*100:4.1f}%)")

    # 確率積の上位N点抽出
    for budget in [36, 72, 144, 243, 500]:
        hits_ai, hits_pop = 0, 0
        pay_ai, pay_pop = 0.0, 0.0
        for dt, rids in win5_days:
            legs_ai, legs_pop, winners, win_odds = [], [], [], []
            valid = True
            for rid in rids:
                r_df = jra[jra["race_id"] == rid].copy()
                w = r_df[r_df["is_win"] == 1]
                if len(w) == 0:
                    valid = False
                    break
                winners.append(int(w.iloc[0]["horse_num"]))
                win_odds.append(float(w.iloc[0]["odds"]))

                p_adj = r_df["adjusted_win_prob"].values
                p_adj = p_adj / np.sum(p_adj)
                p_pure = r_df["pure_ability_prob"].values if "pure_ability_prob" in r_df.columns else p_adj
                p_pure = p_pure / np.sum(p_pure)
                r_df["win5_p"] = 0.85 * p_adj + 0.15 * p_pure
                legs_ai.append(r_df.sort_values("win5_p", ascending=False).head(7)[["horse_num", "win5_p"]].to_dict("records"))

                inv_o = 1.0 / r_df["odds"].clip(lower=1.1)
                r_df["pop_p"] = inv_o / inv_o.sum()
                legs_pop.append(r_df.sort_values(["popularity", "odds"], ascending=[True, True]).head(7)[["horse_num", "pop_p"]].to_dict("records"))

            if not valid:
                continue
            win_tuple = tuple(winners)
            est_pay = min(50000000.0, np.prod([o / 0.8 for o in win_odds]) * 0.7 * 100)

            ai_combos = sorted(
                [(c[0]["win5_p"] * c[1]["win5_p"] * c[2]["win5_p"] * c[3]["win5_p"] * c[4]["win5_p"],
                  (c[0]["horse_num"], c[1]["horse_num"], c[2]["horse_num"], c[3]["horse_num"], c[4]["horse_num"]))
                 for c in itertools.product(*legs_ai)],
                key=lambda x: x[0], reverse=True
            )
            if win_tuple in {x[1] for x in ai_combos[:budget]}:
                hits_ai += 1
                pay_ai += est_pay

            pop_combos = sorted(
                [(c[0]["pop_p"] * c[1]["pop_p"] * c[2]["pop_p"] * c[3]["pop_p"] * c[4]["pop_p"],
                  (c[0]["horse_num"], c[1]["horse_num"], c[2]["horse_num"], c[3]["horse_num"], c[4]["horse_num"]))
                 for c in itertools.product(*legs_pop)],
                key=lambda x: x[0], reverse=True
            )
            if win_tuple in {x[1] for x in pop_combos[:budget]}:
                hits_pop += 1
                pay_pop += est_pay

        cost = len(win5_days) * budget * 100
        roi_ai = (pay_ai / cost) * 100.0
        roi_pop = (pay_pop / cost) * 100.0
        mark = "🔥" if roi_ai >= 100.0 else "  "
        print(f"{f'AI確率積 Top-{budget:3d}点買い':<26} {hits_ai:2d}/{len(win5_days)}回 ({hits_ai/len(win5_days)*100:4.1f}%)       {roi_ai:6.1f}% {mark}      {roi_pop:6.1f}%")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    run_win5_demo()

