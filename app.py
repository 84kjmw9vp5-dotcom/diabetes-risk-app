# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 01:50:11 2026

@author: hiu
"""

# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import numpy as np

# ==========================================
# 1. AIモデルとスケーラーの読み込み
# ==========================================
with open('xgboost_model_app.pkl', 'rb') as f:
    model_9 = pickle.load(f)
with open('scaler_app.pkl', 'rb') as f:
    scaler_9 = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    model_21 = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler_21 = pickle.load(f)

# ==========================================
# 2. 画面UIの構築
# ==========================================
st.title("糖尿病リスク予測シミュレータ")
st.warning("※ 本システムは機械学習モデルによる予測の傾向を示すものであり、医学的な診断を行うものではありません。")

# ==========================================
# 3. 第1段階：簡易予測（9項目）の入力
# ==========================================
st.header("Step 1: 基本情報（9項目）")

col1, col2 = st.columns(2)
with col1:
    age_category = st.slider("年齢カテゴリ (1: 18-24歳 〜 13: 80歳以上)", 1, 13, 7)
    sex = st.radio("性別", options=[0, 1], format_func=lambda x: "女性" if x==0 else "男性")
    height = st.number_input("身長 (cm)", 100.0, 250.0, 170.0)
    weight = st.number_input("体重 (kg)", 30.0, 200.0, 65.0)
    bmi = weight / ((height/100)**2)
    st.info(f"自動計算されたBMI: {bmi:.1f}")

with col2:
    high_bp = st.selectbox("高血圧ですか？", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
    high_chol = st.selectbox("高コレステロールですか？", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
    gen_hlth = st.slider("自身の健康状態の評価 (1:非常に良い 〜 5:悪い)", 1, 5, 3)
    phys_act = st.selectbox("過去30日間に運動をしましたか？", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
    smoker = st.selectbox("喫煙習慣（これまでに100本以上吸った）", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
    hvy_alcohol = st.selectbox("過度な飲酒習慣がありますか？", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")

# ==========================================
# 4. 詳細予測への切り替えスイッチ
# ==========================================
st.write("---")
# ボタンではなくチェックボックスにすることで、UIの切り替えを直感的にしました
show_details = st.checkbox("さらに詳しく予測する（残り12項目を追加）")

# ==========================================
# 5. 第2段階：詳細予測（全21項目）の入力
# ==========================================
if show_details:
    st.header("Step 2: 詳細情報（追加12項目）")
    
    col3, col4 = st.columns(2)
    with col3:
        chol_check = st.selectbox("5年以内にコレステロール検査を受けた", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        stroke = st.selectbox("脳卒中の病歴", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        heart_disease = st.selectbox("心疾患の病歴", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        fruits = st.selectbox("毎日フルーツを食べる", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        veggies = st.selectbox("毎日野菜を食べる", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        healthcare = st.selectbox("医療保険に加入している", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
    
    with col4:
        no_doc_cost = st.selectbox("過去1年で費用が理由で受診できなかった", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        ment_hlth = st.slider("過去30日間のメンタル不調の日数", 0, 30, 0)
        phys_hlth = st.slider("過去30日間の身体的疾患・ケガの日数", 0, 30, 0)
        diff_walk = st.selectbox("歩行や階段に困難がある", options=[0, 1], format_func=lambda x: "いいえ" if x==0 else "はい")
        education = st.slider("教育レベル (1:未修了 〜 6:大卒以上)", 1, 6, 4)
        income = st.slider("所得レベル (1:低い 〜 8:高い)", 1, 8, 5)

# ==========================================
# 6. 予測結果の表示（画面の一番下に1つだけ表示）
# ==========================================
st.write("---")
st.header("予測結果")

# アラートを出す閾値（もし0.65の方が性能が良ければ、ここを65.0に変更します）
THRESHOLD = 60.0

if show_details:
    # 21項目版で計算
    input_data_21 = np.array([[
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease, 
        phys_act, fruits, veggies, hvy_alcohol, healthcare, no_doc_cost, 
        gen_hlth, ment_hlth, phys_hlth, diff_walk, sex, age_category, education, income
    ]])
    input_scaled_21 = scaler_21.transform(input_data_21)
    prob = model_21.predict_proba(input_scaled_21)[0][1] * 100
    mode_text = "詳細予測（21項目）"
else:
    # 9項目版で計算
    input_data_9 = np.array([[age_category, sex, bmi, high_bp, high_chol, gen_hlth, phys_act, smoker, hvy_alcohol]])
    input_scaled_9 = scaler_9.transform(input_data_9)
    prob = model_9.predict_proba(input_scaled_9)[0][1] * 100
    mode_text = "簡易予測（9項目）"

# 最終的な判定を1つだけ出力
if prob >= THRESHOLD:
    st.error(f"【{mode_text}】 あなたの糖尿病リスク: {prob:.1f} % （※注意: 統計的にリスクが高めの傾向にあります）")
else:
    st.success(f"【{mode_text}】 あなたの糖尿病リスク: {prob:.1f} % （※現在のところリスクは低めの傾向です）")