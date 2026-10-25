import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 設定網頁標題與排版
st.set_page_config(page_title="心情點唱機", page_icon="🎵", layout="centered")
st.title("🎵心情點唱機")
st.write("輸入你今天的心情，讓系統為你推薦最適合的一首歌曲吧！")

# 讀取資料與 K-Means 分群
@st.cache_data
def load_data():
    songs = pd.read_csv("songs_data.csv", encoding="utf-8-sig")
    model = KMeans(n_clusters=min(4, len(songs)), n_init=10, random_state=0)
    songs["群"] = model.fit_predict(songs[["valence", "energy"]])
    
    def name_cluster(valence, energy):
        if valence >= 0.5 and energy >= 0.5:
            return "開心"
        elif valence >= 0.5:
            return "平靜"
        elif energy >= 0.5:
            return "壓力"
        else:
            return "難過"
            
    centers = songs.groupby("群")[["valence", "energy"]].mean()
    names = {group_id: name_cluster(row["valence"], row["energy"]) for group_id, row in centers.iterrows()}
    songs["心情"] = songs["群"].map(names)
    return songs

songs = load_data()

@st.cache_resource
def load_classifier():
    train = pd.read_csv("mood_train.csv", encoding="utf-8-sig")
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000, C=10),
    )
    clf.fit(train["句子"], train["心情"])
    return clf


clf = load_classifier()

# 關鍵字與否定詞設定
KEYWORDS = {
    "開心": ["開心", "快樂", "高興", "爽", "太棒", "好玩", "興奮", "棒", "讚"],
    "難過": ["難過", "傷心", "想哭", "失戀", "孤單", "失落", "心痛", "沮喪", "低落", "心情不好", "分手"],
    "壓力": ["壓力", "焦慮", "緊張", "煩", "考試", "報告", "趕", "崩潰", "快瘋了", "累"],
    "平靜": ["平靜", "放鬆", "無聊", "安靜", "還好", "想睡", "放空"],
}
NEGATED = {"開心": "難過", "難過": "平靜", "壓力": "平靜"}

def is_negated(text, i):
    before = text[max(0, i - 2):i]
    return "不" in before or "沒" in before

def detect_mood(text):
    probs = clf.predict_proba([text])[0]
    best = probs.argmax()
    if probs[best] < 0.4:
        return None
    return clf.classes_[best]

#  網頁互動介面
user_input = st.text_input("你現在的心情是？", placeholder="例如：今天考試好累、心情真好...")

if st.button("✨ 幫我推薦歌曲"):
    if not user_input.strip():
        st.warning("請先輸入一些心情描述喔！")
    else:
        mood = detect_mood(user_input)
        
        # 如果偵測不到心情
        if mood is None:
            st.warning("我不太確定你的心情，可以再多描述一點嗎？")
        else:
            candidates = songs[songs["心情"] == mood]
        
        # 若該分類剛好沒歌，從全體隨機挑一首
        if len(candidates) == 0:
            candidates = songs
            
        pick = candidates.sample(1).iloc[0]
        
        # 顯示推薦結果
        st.info(f"🔍 偵測到你的心情偏向：**{mood}**")
        st.success(f"### 推薦你聽：{pick['歌名']} - {pick['歌手']}")
        
        if '歌曲連結' in pick and pd.notna(pick['歌曲連結']):
            st.markdown(f"🔗 [點此前往聆聽歌曲]({pick['歌曲連結']})")