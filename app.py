import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
import socket
from PIL import Image
from scipy.optimize import minimize

# ==========================================
# 1. ページ基本設定 (layout="wide") & モバイル最適化
# ==========================================
st.set_page_config(
    page_title="競馬着順予測AIシミュレータ",
    page_icon="🏇",
    layout="wide"
)

# スマホ画面向けレスポンシブCSS
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        h1 {
            font-size: 1.6rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.15rem !important;
        }
        .stButton button {
            width: 100% !important;
            padding: 0.5rem 1rem !important;
        }
        /* モバイルでのメトリックカードの余白調整 */
        [data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 8px 12px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_local_ip() -> str:
    """PCのローカルネットワークIPアドレスを取得（スマホ連携用）"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ==========================================
# 永続化ストレージ (ファイル保存) 設定
# ==========================================
MODEL_FILE = os.path.join(os.path.dirname(__file__), "ai_model.json")

# デフォルト重み (線形3要素 + 非線形ゲーム理論2要素)
DEFAULT_WEIGHTS = {
    "ability": 1.5,
    "pace": 1.2,
    "jockey": 0.9,
    "escape_dynamic": 1.2,  # 単騎逃げの利 / 逃げ競り合い共倒れの力学強度
    "synergy": 1.1          # 距離×騎手、距離×脚質の相乗効果強度
}

def load_ai_model():
    """ローカルファイルから学習済み重みと履歴を復元"""
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                weights = DEFAULT_WEIGHTS.copy()
                weights.update(data.get("learned_weights", {}))
                history = data.get("training_history", [])
                return weights, history
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy(), []

def save_ai_model(weights, history):
    """学習済み重みと全履歴をローカルファイルに永続化保存"""
    try:
        with open(MODEL_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "learned_weights": weights,
                "training_history": history
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"学習データの保存に失敗しました: {e}")

# セッション状態の初期化 (ファイルから前回の学習内容を自動復元)
if "learned_weights" not in st.session_state:
    saved_weights, saved_history = load_ai_model()
    st.session_state["learned_weights"] = saved_weights
    st.session_state["training_history"] = saved_history

if "predicted_results" not in st.session_state:
    st.session_state["predicted_results"] = None

# 主要騎手の参考勝率データベース (%)
JOCKEY_WIN_RATES = {
    # トップ層 (20%〜)
    "ルメール": 26.5, "C.ルメール": 26.5, "川田": 25.0, "川田将雅": 25.0,
    "モレイラ": 28.0, "J.モレイラ": 28.0, "レーン": 27.0, "D.レーン": 27.0, "ムーア": 25.0, "R.ムーア": 25.0,
    # 上位層 (13%〜19%)
    "戸崎": 16.5, "戸崎圭太": 16.5, "坂井": 15.5, "坂井瑠星": 15.5, "武豊": 15.0,
    "横山武": 14.5, "横山武史": 14.5, "松山": 13.5, "松山弘平": 13.5,
    "岩田望": 13.0, "岩田望来": 13.0, "西村淳": 12.5, "西村淳也": 12.5,
    "デムーロ": 13.0, "M.デムーロ": 13.0, "佐々木": 12.5, "佐々木大輔": 12.5,
    "鮫島駿": 11.5, "鮫島克駿": 11.5,
    # 中堅実力派 (8%〜12%)
    "横山典": 11.5, "横山典弘": 11.5, "菅原明": 11.0, "菅原明良": 11.0,
    "津村": 10.5, "津村明秀": 10.5, "丹内": 10.5, "丹内祐次": 10.5,
    "三浦": 10.0, "三浦皇成": 10.0, "田辺": 10.0, "田辺裕信": 10.0,
    "団野": 10.0, "団野大成": 10.0, "浜中": 10.0, "浜中俊": 10.0,
    "池添": 10.0, "池添謙一": 10.0, "吉田隼": 9.5, "吉田隼人": 9.5,
    "藤岡佑": 9.5, "藤岡佑介": 9.5, "北村友": 9.5, "北村友一": 9.5,
    "北村宏": 8.5, "北村宏司": 8.5, "和田竜": 8.5, "和田竜二": 8.5,
    "幸": 8.5, "幸英明": 8.5, "石橋脩": 8.5, "柴田善": 6.5,
}

def get_jockey_rate(name: str, default_rate: float = 9.0) -> float:
    """騎手名から勝率(%)を取得"""
    if not name or pd.isna(name):
        return default_rate
    name_clean = str(name).strip()
    for j_name, rate in JOCKEY_WIN_RATES.items():
        if j_name in name_clean:
            return rate
    return default_rate

def popularity_to_index(pop: int) -> int:
    """出馬表の単勝人気(1〜18)から能力指数を自動推定"""
    mapping = {
        1: 93, 2: 89, 3: 86, 4: 83, 5: 80,
        6: 77, 7: 75, 8: 73, 9: 71, 10: 69,
        11: 67, 12: 65, 13: 63, 14: 62, 15: 60,
    }
    try:
        p = int(pop)
        return mapping.get(p, max(55, 65 - (p - 10)))
    except:
        return 75

# ==========================================
# 2. サイドバー (レース設定 & 重み設定)
# ==========================================
st.sidebar.header("🏁 レース設定")

if "current_distance" not in st.session_state:
    st.session_state["current_distance"] = 1600

# レース距離の手動数値入力（キーボードで直接入力可能）
race_distance = st.sidebar.number_input(
    "📏 レース距離 (m) [手動入力]",
    min_value=800,
    max_value=4000,
    value=int(st.session_state["current_distance"]),
    step=100,
    help="キーボードで直接好きな数値を手動入力できます（例: 1150, 1700, 2500など）"
)
st.session_state["current_distance"] = race_distance

# 定番距離のクイック選択ボタン
cols_d = st.sidebar.columns(4)
with cols_d[0]:
    if st.button("1200m", use_container_width=True):
        st.session_state["current_distance"] = 1200
        st.rerun()
with cols_d[1]:
    if st.button("1600m", use_container_width=True):
        st.session_state["current_distance"] = 1600
        st.rerun()
with cols_d[2]:
    if st.button("2000m", use_container_width=True):
        st.session_state["current_distance"] = 2000
        st.rerun()
with cols_d[3]:
    if st.button("2400m", use_container_width=True):
        st.session_state["current_distance"] = 2400
        st.rerun()

course_type = st.sidebar.selectbox("コース種別", ["芝", "ダート"], index=0)
pace_prediction = st.sidebar.selectbox("想定ペース", ["スローペース", "ミドルペース", "ハイペース"], index=1)

# 距離カテゴリの自動判定表示
if race_distance <= 1400:
    dist_label = "⚡ 短距離 (スプリント) : スピード・前残り有利"
elif race_distance <= 2000:
    dist_label = "🎯 マイル〜中距離 : スピード＆スタミナの総合力"
else:
    dist_label = "👑 長距離 (ステイヤー) : 騎手の腕前・折り合い重視"
st.sidebar.caption(f"**距離特性**: {dist_label}")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ 予想要素の重視度 (重み)")
st.sidebar.caption("※ AIが学習した最適値が自動セットされます（手動微調整も可能）")

col_rst1, col_rst2 = st.sidebar.columns(2)
with col_rst1:
    if st.button("🔄 標準値に戻す", use_container_width=True):
        st.session_state["learned_weights"] = DEFAULT_WEIGHTS.copy()
        st.rerun()

current_w = st.session_state["learned_weights"]
weight_ability = st.sidebar.slider(
    "1️⃣ 基礎能力（人気・指数）", min_value=0.0, max_value=3.0,
    value=float(current_w.get("ability", 1.5)), step=0.1, key="slider_ability"
)
weight_pace = st.sidebar.slider(
    "2️⃣ 展開・脚質適合度", min_value=0.0, max_value=3.0,
    value=float(current_w.get("pace", 1.2)), step=0.1, key="slider_pace"
)
weight_jockey = st.sidebar.slider(
    "3️⃣ 騎手能力", min_value=0.0, max_value=3.0,
    value=float(current_w.get("jockey", 0.9)), step=0.1, key="slider_jockey"
)

# AI非線形・ゲーム理論パラメータ（自動最適化される深層パラメータ）
with st.sidebar.expander("⚡ **AI非線形・ゲーム理論パラメータ**"):
    st.caption("他馬との駆け引きや距離の掛け算を制御（AI学習で自動最適化）")
    weight_escape_dynamic = st.slider(
        "🔥 単騎逃げ・ハナ争い力学", min_value=0.5, max_value=2.5,
        value=float(current_w.get("escape_dynamic", 1.2)), step=0.1, key="slider_escape_dynamic",
        help="単騎逃げの特大ボーナス、および逃げ馬競り合い時の共倒れペナルティの強度"
    )
    weight_synergy = st.slider(
        "✨ 距離×騎手クロスシナジー", min_value=0.5, max_value=2.5,
        value=float(current_w.get("synergy", 1.1)), step=0.1, key="slider_synergy",
        help="長距離×名手、短距離×先行など条件相乗効果の強度"
    )

# サイドバーに学習履歴の要約と永続化ステータスを表示
if len(st.session_state["training_history"]) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 AI学習データ (保存中)")
    hist = st.session_state["training_history"]
    win_cnt = sum(1 for h in hist if h.get("単勝的中") == "⭕ 的中" or h.get("win_hit", False))
    st.sidebar.write(f"📚 **通算学習レース**: {len(hist)} 戦")
    st.sidebar.write(f"🎯 **単勝的中率**: {win_cnt / len(hist) * 100:.1f}% ({win_cnt}/{len(hist)})")
    st.sidebar.caption("✅ `ai_model.json` に保存中（ブラウザを閉じても消えません）")
    
    if st.sidebar.button("🗑️ 学習データをリセット", help="保存された学習済み重みと履歴を初期化します"):
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        st.session_state["learned_weights"] = DEFAULT_WEIGHTS.copy()
        st.session_state["training_history"] = []
        st.success("学習データを初期化しました！")
        st.rerun()

# スマホ接続ガイド (サイドバー)
local_ip = get_local_ip()
mobile_url = f"http://{local_ip}:8501"

with st.sidebar.expander("📱 **スマホで使う (QRコード)**", expanded=False):
    qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={mobile_url}"
    st.image(qr_img_url, caption="スマホのカメラで読み取って接続", width=180)
    st.code(mobile_url, language="text")
    st.caption("※ PCと同じWi-Fi（またはテザリング）にスマホを接続してアクセスしてください。")

with st.sidebar.expander("🔑 AI画像認識 API設定 (Gemini)"):
    st.caption("出馬表スクショを自動解析するためのGoogle Gemini APIキー（任意）")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Google AI Studioで無料発行できるAPIキー"
    )
    gemini_model_choice = st.selectbox(
        "使用モデル",
        options=["自動切替 (混雑時に別モデルへ自動移行)", "gemini-2.0-flash (高速・安定)", "gemini-1.5-flash (実績多数)", "gemini-2.5-flash (最新・高精度)"],
        index=0,
        help="Googleサーバーの混雑時（503エラー等）は「自動切替」にしておくと、混雑していない安定したモデルで自動再試行します。"
    )

# ==========================================
# 3. メインエリア (タイトル & 画像アップロード)
# ==========================================
st.title("🏇 競馬着順予測AIシミュレータ")
st.caption(f"📱 **スマホでもそのまま使えます！** 同じWi-Fiに繋いだスマホで `{mobile_url}` を開くか、左上メニュー（>）内のQRコードをカメラで読み取ってください。")
st.markdown(
    """
    出馬表の**「人気」「騎手名」**を入力（または**出馬表スクショをアップロード**）して着順を予想！
    レース後は**「実際の結果」を入力することでAIがレース結果を自己学習**し、最適な予想アルゴリズムへと自動進化します。
    """
)

# セッション状態の出走馬データ初期化 (新聞を見て選べる5項目のみ)
if "horses_data" not in st.session_state or "適性距離" in st.session_state["horses_data"].columns:
    st.session_state["horses_data"] = pd.DataFrame([
        {"馬番": 1, "馬名": "グランアレグリア", "単勝人気": 1, "騎手名": "ルメール", "脚質": "先行"},
        {"馬番": 2, "馬名": "コントレイル", "単勝人気": 2, "騎手名": "川田将雅", "脚質": "差し"},
        {"馬番": 3, "馬名": "キセキ", "単勝人気": 4, "騎手名": "武豊", "脚質": "逃げ"},
        {"馬番": 4, "馬名": "クロノジェネシス", "単勝人気": 3, "騎手名": "坂井瑠星", "脚質": "先行"},
        {"馬番": 5, "馬名": "メイケイエール", "単勝人気": 6, "騎手名": "池添謙一", "脚質": "追込"},
    ])

def parse_horses_from_text(raw_text: str) -> list[dict]:
    """Gemini/ChatGPT等のJSONまたはテキスト出力から出走馬データを柔軟に抽出"""
    raw = raw_text.strip()
    if not raw:
        raise ValueError("入力テキストが空です。")

    # ```json ... ``` などのコードブロック抽出
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                p_strip = p_strip[4:].strip()
            if p_strip.startswith("[") and p_strip.endswith("]"):
                raw = p_strip
                break

    # [ から ] までのJSON配列を探索
    start_idx = raw.find("[")
    end_idx = raw.rfind("]")
    extracted_data = []

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_candidate = raw[start_idx:end_idx + 1]
        try:
            extracted_data = json.loads(json_candidate)
        except Exception:
            extracted_data = []

    parsed_list = []

    if isinstance(extracted_data, list) and len(extracted_data) > 0:
        for item in extracted_data:
            if not isinstance(item, dict):
                continue
            parsed_list.append({
                "馬番": int(item.get("馬番", len(parsed_list) + 1)),
                "馬名": str(item.get("馬名", f"馬{len(parsed_list) + 1}")),
                "単勝人気": int(item.get("単勝人気", len(parsed_list) + 1)),
                "騎手名": str(item.get("騎手名", "騎手")),
                "脚質": item.get("脚質", "先行") if item.get("脚質") in ["逃げ", "先行", "差し", "追込"] else "先行",
            })

    # JSONパースに失敗した場合の行ベースフォールバック（netkeibaテキスト等）
    if not parsed_list:
        lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
        for ln in lines:
            if "馬番" in ln and "馬名" in ln:
                continue
            tokens = [tk.strip() for tk in ln.replace("\t", " ").replace(",", " ").replace("|", " ").split(" ") if tk.strip()]
            if len(tokens) >= 2:
                try:
                    h_no = int(tokens[0])
                    h_name = tokens[1]
                    h_pop = int(tokens[2]) if len(tokens) >= 3 and tokens[2].isdigit() else (len(parsed_list) + 1)
                    parsed_list.append({
                        "馬番": h_no,
                        "馬名": h_name,
                        "単勝人気": h_pop,
                        "騎手名": tokens[3] if len(tokens) >= 4 else "騎手",
                        "脚質": "先行",
                    })
                except Exception:
                    continue

    if not parsed_list:
        raise ValueError("出走馬データを抽出できませんでした。JSON形式または出馬表テキストをご確認ください。")

    return parsed_list

# ------------------------------------------
# 出馬表画像アップロード & AI自動読み取り機能
# ------------------------------------------
with st.expander("📷 **【画像・AIから出走馬を一括入力】スクショ解析 または Web版Gemini貼り付け**", expanded=False):
    tab_img_api, tab_paste_text = st.tabs([
        "📷 出馬表スクショをアップロード (API自動抽出)",
        "📋 Web版Gemini/ChatGPTの出力を貼り付け (無料無制限・レートリミット回避！)"
    ])

    with tab_img_api:
        col_upload, col_preview = st.columns([1.2, 1.0])

        with col_upload:
            uploaded_file = st.file_uploader(
                "競馬新聞やnetkeiba・JRA公式サイトの出馬表スクショをアップロード",
                type=["png", "jpg", "jpeg", "webp"]
            )
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_analyze = st.button("✨ 画像から出走馬を自動抽出", type="secondary", disabled=(uploaded_file is None))
            with col_btn2:
                btn_demo = st.button("💡 サンプル出馬表で試す", help="G1有力馬6頭のデモデータをセット", key="btn_demo_api")

        with col_preview:
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="アップロードされた出馬表", use_container_width=True)

        if btn_demo:
            st.session_state["horses_data"] = pd.DataFrame([
                {"馬番": 1, "馬名": "ドウデュース", "単勝人気": 2, "騎手名": "武豊", "脚質": "差し"},
                {"馬番": 2, "馬名": "イクイノックス", "単勝人気": 1, "騎手名": "ルメール", "脚質": "先行"},
                {"馬番": 3, "馬名": "リバティアイランド", "単勝人気": 3, "騎手名": "川田将雅", "脚質": "差し"},
                {"馬番": 4, "馬名": "ジャスティンパレス", "単勝人気": 5, "騎手名": "坂井瑠星", "脚質": "追込"},
                {"馬番": 5, "馬名": "タイトルホルダー", "単勝人気": 4, "騎手名": "横山和生", "脚質": "逃げ"},
                {"馬番": 6, "馬名": "スターズオンアース", "単勝人気": 6, "騎手名": "モレイラ", "脚質": "先行"},
            ])
            st.session_state["predicted_results"] = None
            st.session_state["has_run_prediction"] = False
            st.success("🎉 デモ出馬表（G1有力馬6頭）を読み込みました！")
            st.rerun()

        if btn_analyze and uploaded_file is not None:
            if not gemini_api_key:
                st.warning("⚠️ 画像から直接自動解析するには、サイドバーの「Gemini API Key」が必要です。キーがない・API制限時は「📋 Web版Geminiの出力を貼り付け」タブをご利用ください。")
            else:
                with st.spinner("🔍 AIが出馬表画像を解析中...（混雑時は自動で別モデルに切り替えて再試行します）"):
                    try:
                        from google import genai
                        from google.genai import types

                        client = genai.Client(api_key=gemini_api_key)
                        uploaded_file.seek(0)
                        img_bytes = uploaded_file.read()

                        prompt = """
                        この画像は競馬の出馬表です。
                        画像から各出走馬の情報を読み取り、以下のJSON形式のみ（純粋なJSON配列）で出力してください。

                        [
                          {
                            "馬番": 1,
                            "馬名": "馬名",
                            "単勝人気": 1 (単勝人気。不明ならオッズや実績から1〜18の整数),
                            "騎手名": "騎手名 (例: ルメール、川田、武豊)",
                            "脚質": "逃げ" または "先行" または "差し" または "追込" (推定)
                          }
                        ]
                        注意: 最大18頭まで抽出してください。マークダウンタグはつけず、純粋なJSON文字列のみ出力してください。
                        """

                        if "2.0" in gemini_model_choice:
                            models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
                        elif "1.5" in gemini_model_choice:
                            models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]
                        elif "2.5" in gemini_model_choice:
                            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                        else:
                            models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]

                        response = None
                        last_error = None
                        used_model = None

                        for m_name in models_to_try:
                            for attempt in range(2):
                                try:
                                    response = client.models.generate_content(
                                        model=m_name,
                                        contents=[
                                            types.Part.from_bytes(data=img_bytes, mime_type=uploaded_file.type or "image/png"),
                                            prompt
                                        ]
                                    )
                                    if response and response.text:
                                        used_model = m_name
                                        break
                                except Exception as ex:
                                    last_error = ex
                                    err_str = str(ex)
                                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                                        time.sleep(1.5)
                                        continue
                                    else:
                                        break
                            if response and response.text:
                                break

                        if response is None or not response.text:
                            if last_error:
                                raise last_error
                            else:
                                raise RuntimeError("AIモデルからの応答を取得できませんでした。")

                        parsed_list = parse_horses_from_text(response.text)
                        st.session_state["horses_data"] = pd.DataFrame(parsed_list)
                        st.session_state["predicted_results"] = None
                        st.session_state["has_run_prediction"] = False
                        st.success(f"🎉 出馬表から {len(parsed_list)} 頭の出走馬データを抽出しました！（モデル: {used_model}）")
                        st.rerun()
                    except Exception as e:
                        err_msg = str(e)
                        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                            st.error(
                                "⚠️ **API利用枠（レートリミット）に達しました。**\n\n"
                                "💡 **解決策**: 隣のタブ **「📋 Web版Gemini/ChatGPTの出力を貼り付け」** をご利用ください！\n"
                                "ブラウザの [Google Gemini (無料)](https://gemini.google.com/) に出馬表画像を貼り付けて出力されたテキストをコピペするだけで、**API制限なし・完全無料・無制限**で一瞬で取り込めます。"
                            )
                        elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                            st.error("⚠️ **Google Geminiサーバーが混雑中です（503）。** 隣の「📋 Web版Geminiの出力を貼り付け」タブのご利用もおすすめです。")
                        elif "API_KEY_INVALID" in err_msg or "400" in err_msg:
                            st.error(f"⚠️ **APIキーの確認が必要です**: {err_msg}")
                        else:
                            st.error(f"解析中にエラーが発生しました: {err_msg}")

    # ------------------------------------------
    # タブ2: Web版Gemini/ChatGPTの貼り付け入力
    # ------------------------------------------
    with tab_paste_text:
        st.markdown(
            """
            #### 🌟 完全無料・無制限！Web版AIからコピペで一発入力
            APIキーの利用制限（429エラー）を完全に回避できます。  
            ブラウザで **[Google Gemini (無料)](https://gemini.google.com/)** または **ChatGPT** に出馬表スクショを送り、出力されたテキストをここに貼り付けてください。
            """
        )

        with st.expander("📋 **【手順 1】ブラウザ版AIに送る指示文 (プロンプトをコピー)**", expanded=False):
            st.caption("以下の指示文をコピーして、ブラウザ版Geminiに出馬表画像と一緒に送信してください：")
            prompt_for_web = """この画像は競馬の出馬表です。
画像から全出走馬の情報を読み取り、以下のJSON形式のみ（純粋なJSON配列）で出力してください。

[
  {
    "馬番": 1,
    "馬名": "馬名",
    "単勝人気": 1,
    "騎手名": "騎手名",
    "脚質": "逃げ/先行/差し/追込のいずれか"
  }
]
解説文やマークダウンタグはつけず、純粋なJSON配列のみを出力してください。"""
            st.code(prompt_for_web, language="text")

        st.caption("【手順 2】ブラウザ版AIから返ってきた出力結果をここに貼り付けてください：")
        pasted_content = st.text_area(
            "貼り付けエリア",
            height=160,
            placeholder='ここに [ {"馬番": 1, "馬名": "グランアレグリア", "単勝人気": 1, "騎手名": "ルメール", "脚質": "先行"} ] を貼り付けます',
            key="text_area_pasted_horses"
        )

        col_p1, col_p2 = st.columns([1.5, 1.0])
        with col_p1:
            btn_apply_paste = st.button("📥 貼り付けたデータを出走馬一覧に反映する", type="primary", use_container_width=True)
        with col_p2:
            btn_demo_paste = st.button("💡 デモ出馬表をセット", key="btn_demo_paste", use_container_width=True)

        if btn_demo_paste:
            st.session_state["horses_data"] = pd.DataFrame([
                {"馬番": 1, "馬名": "ドウデュース", "単勝人気": 2, "騎手名": "武豊", "脚質": "差し"},
                {"馬番": 2, "馬名": "イクイノックス", "単勝人気": 1, "騎手名": "ルメール", "脚質": "先行"},
                {"馬番": 3, "馬名": "リバティアイランド", "単勝人気": 3, "騎手名": "川田将雅", "脚質": "差し"},
                {"馬番": 4, "馬名": "ジャスティンパレス", "単勝人気": 5, "騎手名": "坂井瑠星", "脚質": "追込"},
                {"馬番": 5, "馬名": "タイトルホルダー", "単勝人気": 4, "騎手名": "横山和生", "脚質": "逃げ"},
                {"馬番": 6, "馬名": "スターズオンアース", "単勝人気": 6, "騎手名": "モレイラ", "脚質": "先行"},
            ])
            st.session_state["predicted_results"] = None
            st.session_state["has_run_prediction"] = False
            st.success("🎉 デモ出馬表を読み込みました！")
            st.rerun()

        if btn_apply_paste:
            if not pasted_content.strip():
                st.warning("⚠️ 貼り付けエリアにテキストを入力してください。")
            else:
                try:
                    parsed_list = parse_horses_from_text(pasted_content)
                    st.session_state["horses_data"] = pd.DataFrame(parsed_list)
                    st.session_state["predicted_results"] = None
                    st.session_state["has_run_prediction"] = False
                    st.success(f"🎉 貼り付けたテキストから {len(parsed_list)} 頭の出走馬データを一括反映しました！")
                    st.rerun()
                except Exception as ex:
                    st.error(f"データの読み込みに失敗しました: {ex}")

# ==========================================
# 4. 出走馬データ入力テーブル
# ==========================================
st.subheader("📋 出走馬データ")
st.caption("💡 **脚質（逃げ・先行・差し・追込）は、新聞やnetkeibaの通過順（1-1-1なら逃げ、10-10-8なら差し/追込等）を見て自分で選ぶと的中率が大幅に向上します！**")

current_data = st.session_state["horses_data"].copy()
current_data = current_data.loc[:, ~current_data.columns.duplicated()]
required_cols = ["馬番", "馬名", "単勝人気", "騎手名", "脚質"]
existing_cols = [c for c in required_cols if c in current_data.columns]
current_data = current_data[existing_cols]

column_config = {
    "馬番": st.column_config.NumberColumn("馬番", min_value=1, max_value=18, step=1, format="%d", required=True),
    "馬名": st.column_config.TextColumn("馬名", required=True),
    "単勝人気": st.column_config.NumberColumn("単勝人気 (1〜18)", min_value=1, max_value=18, step=1, format="%d位", required=True),
    "騎手名": st.column_config.TextColumn("騎手名", required=True),
    "脚質": st.column_config.SelectboxColumn("脚質 (自分で選択)", options=["逃げ", "先行", "差し", "追込"], required=True, help="競馬新聞の脚質や通過順を見て選んでください"),
}

edited_df = st.data_editor(
    current_data,
    column_config=column_config,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 5. スコアリング＆予測関数
# ==========================================
PACE_MATRIX = {
    "スローペース": {"逃げ": 1.35, "先行": 1.20, "差し": 0.85, "追込": 0.60},
    "ミドルペース": {"逃げ": 1.05, "先行": 1.10, "差し": 1.05, "追込": 0.90},
    "ハイペース":   {"逃げ": 0.60, "先行": 0.80, "差し": 1.25, "追込": 1.35},
}

def get_distance_modifiers(dist: int):
    """
    レース距離に応じた展開補正・騎手重要度（本格競馬モデル）:
    - 短距離 (〜1400m): 前残り絶対有利（逃げ・先行が特大ボーナス、差し・追込は届かず減点）
    - マイル〜中距離 (1500m〜2000m): スピード＆スタミナの総合力勝負
    - 長距離 (2100m〜): スタミナ温存・末脚重視（逃げスタミナ切れ失速減点、差し・追込特大ボーナス、名手騎手重要度急上昇）
    """
    if dist <= 1400:
        ability_factor = 1.15
        jockey_factor = 0.85
        pace_dist_bonus = {"逃げ": 25.0, "先行": 16.0, "差し": -12.0, "追込": -24.0}
        pace_dist_mult = {"逃げ": 1.20, "先行": 1.15, "差し": 0.85, "追込": 0.70}
    elif dist <= 2000:
        ability_factor = 1.00
        jockey_factor = 1.00
        pace_dist_bonus = {"逃げ": 4.0, "先行": 6.0, "差し": 5.0, "追込": 0.0}
        pace_dist_mult = {"逃げ": 1.00, "先行": 1.00, "差し": 1.00, "追込": 1.00}
    else:
        ability_factor = 0.90
        jockey_factor = 1.45
        pace_dist_bonus = {"逃げ": -24.0, "先行": 4.0, "差し": 20.0, "追込": 16.0}
        pace_dist_mult = {"逃げ": 0.75, "先行": 0.95, "差し": 1.20, "追込": 1.15}
    return ability_factor, jockey_factor, pace_dist_bonus, pace_dist_mult

def get_style_distance_eval(style: str, dist: int) -> tuple[float, str]:
    """各馬の脚質とレース距離との展開相性を評価"""
    if dist <= 1400:
        if style == "逃げ":
            return 25.0, "⚡ 前残り絶対有利 (+25)"
        elif style == "先行":
            return 16.0, "◎ 好位抜け出し (+16)"
        elif style == "差し":
            return -12.0, "▲ 届かず不利 (-12)"
        else:
            return -24.0, "✕ 直線短く大不利 (-24)"
    elif dist <= 2000:
        if style in ["先行", "差し"]:
            return 6.0, "○ 総合力適合 (+6)"
        elif style == "逃げ":
            return 4.0, "○ ペース次第 (+4)"
        else:
            return 0.0, "▲ 展開待ち (±0)"
    else:
        if style == "逃げ":
            return -24.0, "✕ スタミナ切れ失速 (-24)"
        elif style == "先行":
            return 4.0, "○ 折り合い要 (+4)"
        elif style == "差し":
            return 20.0, "👑 直線末脚特大有利 (+20)"
        else:
            return 16.0, "◎ 後方脚溜め急襲 (+16)"

def calculate_predictions(
    df: pd.DataFrame,
    w_ability: float,
    w_pace: float,
    w_jockey: float,
    w_escape_dynamic: float = 1.2,
    w_synergy: float = 1.1
) -> tuple[pd.DataFrame, dict]:
    """
    最先端の非線形・ゲーム理論競馬AIロジック:
    1. 他馬の脚質構成による【展開力学（単騎逃げの利 vs 競り合い共倒れ）】
    2. レース距離 × 脚質による【距離展開適合（短距離前残り vs 長距離スタミナ末脚）】
    3. 距離 × 騎手 / 距離 × 脚質の【クロスシナジー】
    4. 能力突出馬の【非線形シグモイド格差拡大】
    5. 期待値【激走穴馬（バリューホース）】自動検知
    """
    if df.empty:
        return df, {}

    dist_ab_factor, dist_jk_factor, pace_dist_bonus, pace_dist_mult = get_distance_modifiers(race_distance)

    # 1. レース内の脚質分布（他馬とのゲーム理論的相互作用）を集計
    escape_count = int((df["脚質"] == "逃げ").sum())
    lead_count = int((df["脚質"] == "先行").sum())

    # 展開力学の自動診断
    if escape_count == 1:
        pace_diag = "🎯 【単騎逃げ濃厚】 誰も競りかけてこずマイペース楽逃げ！逃げ馬が圧倒的有利（前残り警戒）"
    elif escape_count >= 2:
        pace_diag = f"🔥 【ハナ争い激化】 逃げ馬が{escape_count}頭競合！超ハイペース必至で逃げ馬は共倒れ、後方の差し・追込馬が大有利"
    else:
        pace_diag = "🐢 【逃げ馬不在】 押し出された先行馬がスロー単騎！前残り＆位置取り勝負"

    scores = []
    ability_scores = []
    pace_scores = []
    jockey_scores = []
    dynamic_bonuses = []
    synergy_bonuses = []
    dist_bonus_scores = []
    dist_bonus_labels = []
    estimated_indices = []
    jockey_rates = []

    for _, row in df.iterrows():
        pop = row.get("単勝人気", 5)
        idx_val = popularity_to_index(pop)
        estimated_indices.append(idx_val)

        # ① 基礎能力 (上位突出馬の非線形格差補正)
        extra_ability = 0.0
        if idx_val >= 92:
            extra_ability = ((idx_val - 90) ** 1.3) * 1.5
        ability = (float(idx_val) * w_ability * dist_ab_factor) + extra_ability

        # ② 展開・脚質ベース
        style = str(row.get("脚質", "先行"))
        pace_mult = PACE_MATRIX.get(pace_prediction, {}).get(style, 1.0) * pace_dist_mult.get(style, 1.0)
        base_pace = 50.0 * w_pace * pace_mult

        # ③ 【展開力学】他馬との相互作用（単騎逃げの利 vs 競り合い共倒れ）
        dynamic_score = 0.0
        if style == "逃げ":
            if escape_count == 1:
                dynamic_score += 26.0 * w_escape_dynamic  # 単騎逃げ特大ボーナス
            elif escape_count >= 2:
                dynamic_score -= 22.0 * w_escape_dynamic  # 競り合い共倒れ減点
        elif style in ["差し", "追込"]:
            if escape_count >= 2:
                dynamic_score += 20.0 * w_escape_dynamic  # ハイペース前崩れの漁夫の利
            elif escape_count <= 1:
                dynamic_score -= 14.0 * w_escape_dynamic  # 前が止まらず届かない減点
        elif style == "先行":
            if escape_count == 0:
                dynamic_score += 15.0 * w_escape_dynamic  # 逃げ不在で楽に先手を取れる利

        # ④ 騎手能力ベース
        j_name = str(row.get("騎手名", ""))
        j_rate = get_jockey_rate(j_name)
        jockey_rates.append(j_rate)
        base_jockey = j_rate * w_jockey * 3.0 * dist_jk_factor

        # ⑤ 【クロスシナジー】掛け算相互作用
        synergy_score = 0.0
        # 長距離 × トップ騎手 (14%以上: ルメール、川田、武豊、モレイラ等)
        if race_distance >= 2100 and j_rate >= 14.0:
            synergy_score += (j_rate - 11.0) * 2.2 * w_synergy
        # 短距離 × 先行逃げ
        if race_distance <= 1400 and style in ["逃げ", "先行"]:
            synergy_score += 14.0 * w_synergy
        # 長距離 × 競り合い逃げ（スタミナ切れ必至）
        if race_distance >= 2200 and style == "逃げ" and escape_count >= 2:
            synergy_score -= 25.0 * w_synergy

        # ⑥ 【レース距離 × 脚質】の展開適合ボーナス
        dist_bonus_val, dist_bonus_lbl = get_style_distance_eval(style, race_distance)

        total = ability + base_pace + dynamic_score + base_jockey + synergy_score + dist_bonus_val

        ability_scores.append(round(ability, 2))
        pace_scores.append(round(base_pace + dynamic_score, 2))
        jockey_scores.append(round(base_jockey + synergy_score, 2))
        dynamic_bonuses.append(round(dynamic_score, 2))
        synergy_bonuses.append(round(synergy_score, 2))
        dist_bonus_scores.append(round(dist_bonus_val, 2))
        dist_bonus_labels.append(dist_bonus_lbl)
        scores.append(total)

    result_df = df.copy()
    result_df["推定能力指数"] = estimated_indices
    result_df["騎手勝率(%)"] = jockey_rates
    result_df["距離展開評価"] = dist_bonus_labels
    result_df["距離展開点"] = dist_bonus_scores
    result_df["総合スコア"] = np.round(scores, 2)
    result_df["能力スコア"] = ability_scores
    result_df["展開スコア"] = pace_scores
    result_df["騎手スコア"] = jockey_scores
    result_df["展開力学"] = dynamic_bonuses
    result_df["シナジー"] = synergy_bonuses

    scores_array = np.array(scores, dtype=float)
    if len(scores_array) > 0:
        score_std = np.std(scores_array)
        temperature = max(score_std, 10.0)
        exp_scores = np.exp((scores_array - np.max(scores_array)) / temperature)
        win_rates = (exp_scores / np.sum(exp_scores)) * 100.0
        result_df["勝率(%)"] = np.round(win_rates, 1)
    else:
        result_df["勝率(%)"] = 0.0

    result_df = result_df.sort_values(by="総合スコア", ascending=False).reset_index(drop=True)
    result_df["予想着順数値"] = range(1, len(result_df) + 1)
    result_df.insert(0, "予想着順", [f"{i}着" for i in range(1, len(result_df) + 1)])

    # 激走穴馬（単勝人気4位以下だがAI上位3着以内）の判定
    result_df["特注穴馬"] = False
    for idx, row in result_df.iterrows():
        if int(row["単勝人気"]) >= 4 and int(row["予想着順数値"]) <= 3:
            result_df.at[idx, "特注穴馬"] = True

    meta = {
        "escape_count": escape_count,
        "lead_count": lead_count,
        "pace_diagnosis": pace_diag
    }
    return result_df, meta

# ==========================================
# 6. 予測実行
# ==========================================
st.markdown("---")
btn_predict = st.button("🚀 着順を予測する", type="primary", use_container_width=True)

if "has_run_prediction" not in st.session_state:
    st.session_state["has_run_prediction"] = False

if btn_predict:
    st.session_state["has_run_prediction"] = True

# 一度予測を実行した後は、サイドバーの距離変更・スライダー変更・データ変更時に自動でリアルタイム再計算
if st.session_state["has_run_prediction"]:
    valid_df = edited_df.dropna(subset=["馬名"]).copy()
    if valid_df.empty:
        st.warning("⚠️ 出走馬データを入力してください。")
    else:
        results, meta = calculate_predictions(
            valid_df, weight_ability, weight_pace, weight_jockey,
            weight_escape_dynamic, weight_synergy
        )
        st.session_state["predicted_results"] = results
        st.session_state["predicted_meta"] = meta

        # 確定着順入力用データの同期・整合性維持
        if "actual_results_df" not in st.session_state or set(st.session_state["actual_results_df"]["馬番"]) != set(results["馬番"]):
            st.session_state["actual_results_df"] = results[["馬番", "馬名", "予想着順", "予想着順数値"]].copy()
            st.session_state["actual_results_df"]["実際の着順"] = results["予想着順数値"]
        else:
            # 既存の入力値を保持しながら予想着順を最新化
            curr_map = st.session_state["actual_results_df"].set_index("馬番")["実際の着順"].to_dict()
            new_actual = results[["馬番", "馬名", "予想着順", "予想着順数値"]].copy()
            new_actual["実際の着順"] = new_actual["馬番"].map(curr_map).fillna(new_actual["予想着順数値"]).astype(int)
            st.session_state["actual_results_df"] = new_actual

# ------------------------------------------
# 予測結果の表示エリア
# ------------------------------------------
if st.session_state["predicted_results"] is not None:
    results = st.session_state["predicted_results"]
    meta = st.session_state.get("predicted_meta", {})

    # レース条件バッジ
    st.info(f"📍 **設定レース条件**: 【{course_type} {race_distance}m】 / 想定ペース: **{pace_prediction}** ({dist_label}) ｜ ⚡ *サイドバーの距離変更に即座に連動して自動再計算中*")

    # 🤖 AI展開力学シミュレーション診断
    pace_diag_msg = meta.get("pace_diagnosis", "")
    if pace_diag_msg:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 12px 18px; border-radius: 8px; color: white; margin-bottom: 12px;">
                <b>🤖 AI展開力学シミュレータの診断結果:</b><br>{pace_diag_msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    # 激走穴馬（オッズ妙味馬）のアラート検知
    upset_horses = results[results["特注穴馬"] == True]
    if len(upset_horses) > 0:
        upset_list = [f"**{r['馬番']}番 {r['馬名']}**（単勝{r['単勝人気']}位 → AI予測{r['予想着順']}）" for _, r in upset_horses.iterrows()]
        st.warning(f"🔥 **【AI特注・激走穴馬を発掘！】** 世間の人気は控えめですが、展開利・シナジーにより高評価： {' / '.join(upset_list)}")

    st.subheader("🏆 予測結果サマリー (上位3頭)")
    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    medals = ["🥇 1着候補", "🥈 2着候補", "🥉 3着候補"]

    for i in range(min(3, len(results))):
        horse = results.iloc[i]
        with cols[i]:
            st.metric(
                label=medals[i],
                value=f"{horse['馬番']}番 {horse['馬名']} ({horse['騎手名']})",
                delta=f"勝率 {horse['勝率(%)']}% (総合スコア: {horse['総合スコア']})"
            )

    st.subheader("📊 全頭の予測着順一覧")
    display_columns = [
        "予想着順", "馬番", "馬名", "単勝人気", "騎手名", "脚質", "距離展開評価",
        "勝率(%)", "総合スコア", "推定能力指数", "騎手勝率(%)",
        "能力スコア", "展開スコア", "騎手スコア", "展開力学", "シナジー"
    ]
    filtered_display = results[[c for c in display_columns if c in results.columns]]

    st.dataframe(
        filtered_display,
        column_config={
            "勝率(%)": st.column_config.ProgressColumn("勝率(%)", format="%.1f%%", min_value=0.0, max_value=100.0),
            "単勝人気": st.column_config.NumberColumn("人気", format="%d位"),
            "脚質": st.column_config.TextColumn("選択脚質"),
            "距離展開評価": st.column_config.TextColumn("距離×脚質適合度"),
            "総合スコア": st.column_config.NumberColumn("総合スコア", format="%.2f"),
            "推定能力指数": st.column_config.NumberColumn("推定能力値", format="%d"),
            "騎手勝率(%)": st.column_config.NumberColumn("騎手勝率", format="%.1f%%"),
            "展開力学": st.column_config.NumberColumn("展開力学(単騎/崩壊)", format="%+.1f"),
            "シナジー": st.column_config.NumberColumn("掛け算シナジー", format="%+.1f")
        },
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------
    # 🎫 AI推奨！おすすめ買い目 3選
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🎫 AI推奨！おすすめ買い目 3選")
    st.caption("AIが算出した勝率と総合スコアに基づき、スタイル別に最適化された3つの馬券戦略を提示します。")

    if len(results) >= 2:
        h1 = results.iloc[0]
        h2 = results.iloc[1]
        h3 = results.iloc[2] if len(results) >= 3 else None
        h4 = results.iloc[3] if len(results) >= 4 else None

        col_bet1, col_bet2, col_bet3 = st.columns(3)

        # プラン1: 堅実・本命派
        with col_bet1:
            st.markdown("#### 🛡️ ①【堅実・的中重視】")
            st.markdown(
                f"""
                **推奨馬券: 単勝 ＆ ワイド**
                * 🎯 **単勝**: **{h1['馬番']}番** {h1['馬名']}
                * 🤝 **ワイド**: **{h1['馬番']} - {h2['馬番']}** (本線1点)
                
                ---
                * **推奨度**: ⭐⭐⭐⭐⭐
                * **狙い**: 勝率トップの◎{h1['馬名']}と対抗○{h2['馬名']}による手堅い的中狙い。
                * **資金配分例 (1,000円)**:
                  - 単勝 {h1['馬番']}: 500円
                  - ワイド {h1['馬番']}-{h2['馬番']}: 500円
                """
            )

        # プラン2: バランス・王道派
        with col_bet2:
            st.markdown("#### ⚖️ ②【王道・バランス】")
            if h3 is not None:
                st.markdown(
                    f"""
                    **推奨馬券: 馬連 流し (2点)**
                    * 🎯 **軸馬**: **{h1['馬番']}番** {h1['馬名']}
                    * 相手: **{h2['馬番']}番** ({h2['馬名']})、**{h3['馬番']}番** ({h3['馬名']})
                    * **買い目**: **{h1['馬番']} - {h2['馬番']}, {h3['馬番']}** (計2点)
                    
                    ---
                    * **推奨度**: ⭐⭐⭐⭐☆
                    * **狙い**: 的中率とオッズ妙味を両立した競馬ファンの王道スタイル。
                    * **資金配分例 (1,000円)**:
                      - 馬連 {h1['馬番']}-{h2['馬番']}: 700円 (本線)
                      - 馬連 {h1['馬番']}-{h3['馬番']}: 300円 (抑え)
                    """
                )
            else:
                st.markdown(
                    f"""
                    **推奨馬券: 馬連 (1点)**
                    * **買い目**: **{h1['馬番']} - {h2['馬番']}**
                    * **資金配分例**: 1,000円 (1点勝負)
                    """
                )

        # プラン3: 高配当・一撃狙い
        with col_bet3:
            st.markdown("#### 🚀 ③【高配当・一撃狙い】")
            if h3 is not None:
                bet3_text = f"**{h1['馬番']} - {h2['馬番']} - {h3['馬番']}** (1点集中)"
                if h4 is not None:
                    bet3_extra = f"\n  - 3連複 {h1['馬番']}-{h2['馬番']}-{h4['馬番']}: 300円 (高配当抑え)"
                    bet3_main_alloc = "700円"
                else:
                    bet3_extra = ""
                    bet3_main_alloc = "1,000円"

                st.markdown(
                    f"""
                    **推奨馬券: 3連複 集中勝負**
                    * 🎯 **本線**: {bet3_text}
                    
                    ---
                    * **推奨度**: ⭐⭐⭐☆☆
                    * **狙い**: AI総合スコア上位馬による上位独占を狙い撃つ高回収率特化プラン。
                    * **資金配分例 (1,000円)**:
                      - 3連複 {h1['馬番']}-{h2['馬番']}-{h3['馬番']}: {bet3_main_alloc} (本命){bet3_extra}
                    """
                )
            else:
                st.markdown(
                    f"""
                    **推奨馬券: 馬単 (表裏2点)**
                    * **買い目**: **{h1['馬番']} ⇄ {h2['馬番']}**
                    * **資金配分例**: 各500円
                    """
                )

    # ==========================================
    # 7. 実際の結果入力 & 答え合わせ & AI自己学習
    # ==========================================
    st.markdown("---")
    st.header("🎯 レース結果の入力 ＆ AI自己学習（パラメータ最適化）")
    st.markdown(
        """
        レース終了後、**「実際の結果（着順）」を入力してください。**
        AIが予想と実際の結果を照合・比較し、**次回以降の的中率が最大になるように重視度（重み）を自動学習・最適化**します。
        """
    )

    with st.expander("📝 **全頭の確定着順（4着以下も含む）を入力する**", expanded=True):
        actual_input_df = st.session_state["actual_results_df"].copy()
        total_horses = len(actual_input_df)
        horse_names = actual_input_df["馬名"].tolist()

        tab_input_quick, tab_input_num, tab_input_table = st.tabs([
            "⚡ クイック選択 (1着〜5着掲示板)",
            "⌨️ 馬番順で一括入力 (簡単)",
            "📋 テーブルで直接編集 (全頭)"
        ])

        # ----------------------------------
        # タブ1: 1着〜5着のクイック選択
        # ----------------------------------
        with tab_input_quick:
            st.caption("上位馬（掲示板）をプルダウンで指定します。指定されなかった馬は自動的に下位着順に採番されます。")
            cols_pick = st.columns(min(5, total_horses))
            picked_horses = []

            for i in range(min(5, total_horses)):
                rank_num = i + 1
                with cols_pick[i]:
                    avail_options = [h for h in horse_names if h not in picked_horses]
                    default_idx = 0 if avail_options else 0
                    p_horse = st.selectbox(
                        f"{rank_num}着馬",
                        options=avail_options if avail_options else horse_names,
                        index=default_idx,
                        key=f"pick_rank_{rank_num}"
                    )
                    picked_horses.append(p_horse)

            if st.button("⚡ 上記の着順を反映する", key="btn_apply_quick"):
                # 選択された馬に着順を割り当て
                assigned_ranks = {}
                for r_idx, h_name in enumerate(picked_horses):
                    assigned_ranks[h_name] = r_idx + 1

                # 残りの馬には6着以降を連番で割り当て
                rem_rank = len(picked_horses) + 1
                for h_name in horse_names:
                    if h_name not in assigned_ranks:
                        assigned_ranks[h_name] = rem_rank
                        rem_rank += 1

                for idx, r in actual_input_df.iterrows():
                    actual_input_df.at[idx, "実際の着順"] = assigned_ranks.get(r["馬名"], total_horses)

                st.session_state["actual_results_df"] = actual_input_df
                st.success("🎉 全頭の確定着順を反映しました！")
                st.rerun()

        # ----------------------------------
        # タブ2: 馬番順で一括入力
        # ----------------------------------
        with tab_input_num:
            st.caption("着順通りの馬番をカンマ区切りで入力（例: `2, 4, 1, 5, 3`）。1着から最下位まで一瞬でセットできます。")
            default_seq = ", ".join(actual_input_df.sort_values(by="実際の着順")["馬番"].astype(str).tolist())
            seq_text = st.text_input("確定着順の馬番順 (1着から順に入力)", value=default_seq)
            
            if st.button("📥 馬番順で確定着順を一括反映", key="btn_apply_seq"):
                try:
                    num_list = [int(x.strip()) for x in seq_text.replace("、", ",").split(",") if x.strip()]
                    if len(num_list) != total_horses:
                        st.warning(f"⚠️ 出走頭数（{total_horses}頭）と同じ数の馬番を入力してください（現在: {len(num_list)}頭）")
                    else:
                        for rank_i, horse_no in enumerate(num_list, start=1):
                            match_idx = actual_input_df[actual_input_df["馬番"] == horse_no].index
                            if len(match_idx) > 0:
                                actual_input_df.at[match_idx[0], "実際の着順"] = rank_i
                        st.session_state["actual_results_df"] = actual_input_df
                        st.success("🎉 全頭の馬番着順を反映しました！")
                        st.rerun()
                except Exception as ex:
                    st.error(f"入力フォーマットを確認してください: {ex}")

        # ----------------------------------
        # タブ3: テーブル直接入力 (全頭対応)
        # ----------------------------------
        with tab_input_table:
            st.caption("各馬の「実際の着順」セルをダブルクリックして、1着〜最下位までの数値を直接変更できます。")

        edited_actual_df = st.data_editor(
            actual_input_df,
            column_config={
                "馬番": st.column_config.NumberColumn("馬番", disabled=True),
                "馬名": st.column_config.TextColumn("馬名", disabled=True),
                "予想着順": st.column_config.TextColumn("予想着順", disabled=True),
                "実際の着順": st.column_config.NumberColumn(
                    f"実際の着順 (1〜{total_horses}着)",
                    min_value=1,
                    max_value=total_horses,
                    step=1,
                    format="%d着",
                    required=True,
                    help="4着以下も含め、すべての馬の確定着順を指定できます"
                )
            },
            use_container_width=True,
            hide_index=True
        )

    # ------------------------------------------
    # 予想 vs 実際結果の比較 & 的中判定
    # ------------------------------------------
    st.subheader("🔍 予想 vs 実際結果の答え合わせ")
    
    comp_df = edited_actual_df.copy()
    comp_df["着順誤差"] = (comp_df["予想着順数値"] - comp_df["実際の着順"]).abs()
    
    # 的中判定
    top1_pred = comp_df[comp_df["予想着順数値"] == 1].iloc[0]
    win_hit = (top1_pred["実際の着順"] == 1)

    top3_pred_names = set(comp_df[comp_df["予想着順数値"] <= 3]["馬名"])
    top3_actual_names = set(comp_df[comp_df["実際の着順"] <= 3]["馬名"])
    place_hits = len(top3_pred_names.intersection(top3_actual_names))
    
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        if win_hit:
            st.success("🎯 **単勝予想: 【的中！】**\n1着候補が見事1着に入線しました！")
        else:
            st.warning(f"❌ **単勝予想: 【不的中】**\n1着候補: {top1_pred['馬名']} (実際は{int(top1_pred['実際の着順'])}着)")
    with col_v2:
        st.info(f"🏅 **3着以内好走率**: {place_hits} / 3頭的中\n予想上位3頭中、{place_hits}頭が3着以内に入線")
    with col_v3:
        avg_err = comp_df["着順誤差"].mean()
        st.metric("全頭の平均着順誤差", f"{avg_err:.1f}着", help="数値が小さいほど全頭の順位予測の精度が高い")

    # 全頭の着順対比表 (確定着順順)
    comp_display = comp_df[["馬番", "馬名", "予想着順", "実際の着順", "着順誤差"]].copy()
    comp_display = comp_display.sort_values(by="実際の着順").reset_index(drop=True)
    comp_display["判定"] = comp_display["着順誤差"].apply(
        lambda e: "🎯 完全一致" if e == 0 else ("👍 誤差1着以内" if e == 1 else f"差 {int(e)}着")
    )
    comp_display["実際の着順"] = comp_display["実際の着順"].apply(lambda x: f"{int(x)}着")

    st.dataframe(
        comp_display[["実際の着順", "馬番", "馬名", "予想着順", "着順誤差", "判定"]],
        column_config={
            "着順誤差": st.column_config.NumberColumn("着順ズレ", format="%d"),
        },
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------
    # 🧠 AI自己学習・重み最適化アルゴリズム
    # ------------------------------------------
    st.markdown("---")
    st.subheader("🧠 AI重み学習・自動最適化")
    st.markdown(
        """
        実際の結果をもとに、**「このレース結果を当てるには、各ファクターをどの重みに設定すべきだったか」**を
        最適化アルゴリズム（ランキング順位最適化）によって逆算・学習します。
        """
    )

    if st.button("🚀 実際の結果からAI重みを学習・最適化する", type="primary"):
        with st.spinner("🤖 AIがレース結果から最適な重みパラメータを学習・最適化中..."):
            # 学習対象データ
            features_for_opt = results.copy()
            merged_opt = pd.merge(
                features_for_opt[["馬番", "馬名", "推定能力指数", "脚質", "騎手勝率(%)"]],
                comp_df[["馬番", "実際の着順"]],
                on="馬番"
            )

            # ランキング損失関数 (実着順のペアに対して、非線形パラメータも含めて予測スコアの逆転を最小化)
            dist_ab_factor, dist_jk_factor, pace_dist_bonus, pace_dist_mult = get_distance_modifiers(race_distance)
            escape_count = int((merged_opt["脚質"] == "逃げ").sum())

            def ranking_loss(weights):
                w_a, w_p, w_j, w_esc, w_syn = weights
                calc_scores = []
                for _, r in merged_opt.iterrows():
                    idx_val = float(r["推定能力指数"])
                    extra_ab = ((idx_val - 90) ** 1.3) * 1.5 if idx_val >= 92 else 0.0
                    ability = (idx_val * w_a * dist_ab_factor) + extra_ab

                    style = str(r["脚質"])
                    pace_mult = PACE_MATRIX.get(pace_prediction, {}).get(style, 1.0) * pace_dist_mult.get(style, 1.0)
                    base_pace = 50.0 * w_p * pace_mult

                    # 展開力学
                    dynamic_score = 0.0
                    if style == "逃げ":
                        if escape_count == 1:
                            dynamic_score += 26.0 * w_esc
                        elif escape_count >= 2:
                            dynamic_score -= 22.0 * w_esc
                    elif style in ["差し", "追込"]:
                        if escape_count >= 2:
                            dynamic_score += 20.0 * w_esc
                        elif escape_count <= 1:
                            dynamic_score -= 14.0 * w_esc
                    elif style == "先行" and escape_count == 0:
                        dynamic_score += 15.0 * w_esc

                    j_rate = float(r["騎手勝率(%)"])
                    base_jockey = j_rate * w_j * 3.0 * dist_jk_factor

                    # クロスシナジー
                    synergy_score = 0.0
                    if race_distance >= 2100 and j_rate >= 14.0:
                        synergy_score += (j_rate - 11.0) * 2.2 * w_syn
                    if race_distance <= 1400 and style in ["逃げ", "先行"]:
                        synergy_score += 14.0 * w_syn
                    if race_distance >= 2200 and style == "逃げ" and escape_count >= 2:
                        synergy_score -= 25.0 * w_syn

                    # 距離×脚質展開ボーナス
                    dist_bonus_val, _ = get_style_distance_eval(style, race_distance)

                    calc_scores.append(ability + base_pace + dynamic_score + base_jockey + synergy_score + dist_bonus_val)

                scores_arr = np.array(calc_scores)
                actual_ranks = merged_opt["実際の着順"].values

                # ペアワイズ損失: 実際着順 i < j ならば score_i > score_j であるべき
                loss = 0.0
                n = len(actual_ranks)
                for i in range(n):
                    for j in range(i + 1, n):
                        if actual_ranks[i] < actual_ranks[j]:
                            margin = (scores_arr[j] - scores_arr[i]) + 2.0
                            if margin > 0:
                                loss += margin ** 2
                        elif actual_ranks[i] > actual_ranks[j]:
                            margin = (scores_arr[i] - scores_arr[j]) + 2.0
                            if margin > 0:
                                loss += margin ** 2

                # 正則化項 (極端な偏りを防ぐ安定化項)
                reg = 0.04 * (
                    (w_a - 1.5)**2 + (w_p - 1.2)**2 + (w_j - 0.9)**2 +
                    (w_esc - 1.2)**2 + (w_syn - 1.1)**2
                )
                return loss + reg

            # 最適化の実行 (5次元パラメータ探索)
            initial_guess = [weight_ability, weight_pace, weight_jockey, weight_escape_dynamic, weight_synergy]
            bounds = [(0.1, 3.0), (0.1, 3.0), (0.1, 3.0), (0.3, 2.5), (0.3, 2.5)]

            opt_res = minimize(ranking_loss, initial_guess, method="L-BFGS-B", bounds=bounds)

            new_w_ability = round(float(opt_res.x[0]), 2)
            new_w_pace = round(float(opt_res.x[1]), 2)
            new_w_jockey = round(float(opt_res.x[2]), 2)
            new_w_escape = round(float(opt_res.x[3]), 2)
            new_w_synergy = round(float(opt_res.x[4]), 2)

            # 学習結果の保存 (セッション & ローカルファイルに永続化)
            st.session_state["learned_weights"] = {
                "ability": new_w_ability,
                "pace": new_w_pace,
                "jockey": new_w_jockey,
                "escape_dynamic": new_w_escape,
                "synergy": new_w_synergy
            }
            hist_entry = {
                "レース番号": f"第{len(st.session_state['training_history']) + 1}戦",
                "学習日時": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "コース・距離": f"{course_type} {race_distance}m",
                "想定ペース": pace_prediction,
                "単勝的中": "⭕ 的中" if win_hit else "❌ 不的中",
                "3着内好走": f"{place_hits}/3頭",
                "学習後重み(能力/展開/騎手/力学/相乗)": f"{new_w_ability:.2f}/{new_w_pace:.2f}/{new_w_jockey:.2f}/{new_w_escape:.2f}/{new_w_synergy:.2f}"
            }
            st.session_state["training_history"].append(hist_entry)
            
            # ローカルファイルに永久保存
            save_ai_model(st.session_state["learned_weights"], st.session_state["training_history"])

            st.success("🎉 **最先端AIの多次元最適化学習が完了しました！最適化結果は永久保存されました。**")

            # ビフォーアフターの比較表示
            col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
            with col_b1:
                delta_a = round(new_w_ability - weight_ability, 2)
                st.metric("1️⃣ 基礎能力", f"{new_w_ability:.2f}", delta=f"{delta_a:+.2f}")
            with col_b2:
                delta_p = round(new_w_pace - weight_pace, 2)
                st.metric("2️⃣ 展開・脚質", f"{new_w_pace:.2f}", delta=f"{delta_p:+.2f}")
            with col_b3:
                delta_j = round(new_w_jockey - weight_jockey, 2)
                st.metric("3️⃣ 騎手能力", f"{new_w_jockey:.2f}", delta=f"{delta_j:+.2f}")
            with col_b4:
                delta_e = round(new_w_escape - weight_escape_dynamic, 2)
                st.metric("🔥 展開力学", f"{new_w_escape:.2f}", delta=f"{delta_e:+.2f}")
            with col_b5:
                delta_s = round(new_w_synergy - weight_synergy, 2)
                st.metric("✨ シナジー", f"{new_w_synergy:.2f}", delta=f"{delta_s:+.2f}")

            # AIの分析解説インサイト
            insights = []
            if delta_e > 0.05:
                insights.append("🔥 **【展開力学の学習】** 逃げ馬の頭数による影響（単騎逃げの利、または逃げ争いによる共倒れ・差し台頭）をより敏感に評価するよう進化しました。")
            if delta_s > 0.05:
                insights.append("✨ **【クロスシナジーの学習】** 「長距離×トップ騎手」や「短距離×先行」などの条件掛け算シナジーをより強力に評価するように学習しました。")
            if delta_p > 0.1:
                insights.append("🏇 **展開（脚質）が勝敗を大きく左右したレース**でした。展開・位置取りの基本重要度を引き上げました。")
            if delta_a > 0.1:
                insights.append("💪 **馬の絶対的な基礎能力・人気通りの強さが目立ったレース**でした。基礎能力の重視度を引き上げました。")
            if delta_j > 0.1:
                insights.append("🏇 **トップ騎手の腕前・勝負勘が大きく貢献したレース**でした。騎手能力の重視度を引き上げました。")

            if insights:
                st.info("💡 **AIの学習分析コメント**:\n" + "\n".join(insights))

            st.balloons()

# ==========================================
# 8. 永続化された学習レース履歴の一覧表示
# ==========================================
if len(st.session_state["training_history"]) > 0:
    st.markdown("---")
    with st.expander("📚 **【永久保存】これまでのAI学習レース履歴一覧**", expanded=False):
        history_df = pd.DataFrame(st.session_state["training_history"])
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.caption("※ すべての学習データはローカルの `ai_model.json` に安全に保管されており、次回起動時にも引き継がれます。")
