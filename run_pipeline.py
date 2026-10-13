"""
run_pipeline.py
統合予測AI開発用マスターパイプライン
中央（JRA）および地方（NAR）の過去データを統合し、
1. データ統合・前処理・特徴量エンジニアリング
2. LightGBM汎用予測モデル構築・AUC評価・Feature Importance出力
3. G1・有馬記念等の特殊条件予測補正
4. WIN5モンテカルロ法予測シミュレーター（1万回試行・Top10買い目）
5. 期待値(EV>100%)計算＆回収率バックテスト（中央/地方/G1別）
を順次実行してターミナルに出力する。
"""

import sys
import os
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    start_time = time.time()
    print("\n" + "#" * 70)
    print("🏇 【中央・地方 統合競馬予測AI パイプライン マスター実行】 🏇")
    print("#" * 70 + "\n")

    # Step 1: データ統合 & 前処理
    import pipeline_data_prep
    t0 = time.time()
    pipeline_data_prep.load_and_preprocess()
    print(f"⏱️ Step 1 完了: {time.time() - t0:.1f} 秒\n")

    # Step 2: 汎用予測モデルの構築 & AUC評価
    import pipeline_train
    t0 = time.time()
    pipeline_train.train_base_model()
    print(f"⏱️ Step 2 完了: {time.time() - t0:.1f} 秒\n")

    # Step 3: 特殊条件 (有馬記念・G1等) の予測補正
    import pipeline_g1_adjust
    t0 = time.time()
    pipeline_g1_adjust.demonstrate_arima_kinen()
    print(f"⏱️ Step 3 完了: {time.time() - t0:.1f} 秒\n")

    # Step 4: WIN5予測シミュレーターの実装
    import pipeline_win5_sim
    t0 = time.time()
    pipeline_win5_sim.run_win5_demo()
    print(f"⏱️ Step 4 完了: {time.time() - t0:.1f} 秒\n")

    # Step 5: 期待値計算とバックテスト（回収率の検証）
    import pipeline_backtest
    t0 = time.time()
    pipeline_backtest.run_ev_backtest()
    print(f"⏱️ Step 5 完了: {time.time() - t0:.1f} 秒\n")

    total_sec = time.time() - start_time
    print("#" * 70)
    print(f"🎉 【全パイプライン正常完了！】 総所要時間: {total_sec:.1f} 秒")
    print("#" * 70 + "\n")

if __name__ == "__main__":
    main()
