"""Small offline visual layer built on native Streamlit components."""

import base64
from pathlib import Path

import streamlit as st

CSS = """
<style>
.stApp { background: #F8FBFA; color: #264653; }
[data-testid="stHeader"] { background: rgba(248,251,250,.94); }
[data-testid="stSidebar"] { background: #EEF7F3; border-right: 1px solid #dceae4; }
.stMainBlockContainer { max-width: 1240px; padding-top: 2.5rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #264653; letter-spacing: -.025em; }
[data-testid="stMetric"] {
  background: #fff; border: 1px solid #e0ece6; border-radius: 18px;
  padding: 20px 22px; box-shadow: 0 5px 22px rgba(38,70,83,.045);
}
[data-testid="stMetricValue"] { color: #247e73; }
[data-testid="stMetricValue"] > div { white-space: normal; overflow-wrap: anywhere;
  font-size: clamp(1.3rem, 2.2vw, 2rem); line-height: 1.25; }
.st-key-nutrition_estimate [data-testid="stMetric"] { border-top: 3px solid #9bc9ba; }
.st-key-nutrition_estimate [data-testid="stColumn"]:first-child [data-testid="stMetric"] {
  background: #fff8ed; border-color: #f0d2a8; border-top-color: #F4A261;
}
.st-key-nutrition_estimate [data-testid="stColumn"]:first-child [data-testid="stMetricValue"] {
  color: #875423;
}
[data-testid="stExpander"], [data-testid="stDataFrame"] {
  border-radius: 16px; overflow: hidden; background: #fff;
}
[data-testid="stExpander"] { border: 1px solid #dce9e3; }
[data-testid="stExpander"] summary { background: #f0f7f3; }
[data-testid="stFileUploader"] {
  padding: 12px; border-radius: 20px; background: #fff; border: 1px solid #dce9e3;
}
.stButton > button, .stFormSubmitButton > button { border-radius: 12px; min-height: 42px; }
.stButton > button:hover { border-color: #2A9D8F; box-shadow: 0 4px 12px #26465312; }
.stButton > button:focus-visible { outline: 3px solid #f4a261; outline-offset: 3px; }
[data-testid="stAlert"] { border-radius: 14px; }
.food-hero {
  display: flex; align-items: center; gap: 20px; border: 1px solid #dcebe0;
  background: linear-gradient(115deg, #eef7f3, #fffdf8); border-radius: 28px;
  padding: 0; margin-bottom: 28px; overflow: hidden; min-height: 390px;
}
.food-hero-copy { flex: 1.1; padding: 36px 40px; }
.food-eyebrow { color: #247e73; font-size: .78rem; font-weight: 700; letter-spacing: .13em; }
.food-hero h1 { font-size: clamp(2rem, 3.5vw, 3.2rem); line-height: 1.14; margin: 14px 0; }
.food-hero p { line-height: 1.8; color: #506b72; max-width: 540px; }
.food-photo { width: 47%; align-self: stretch; position: relative; min-height: 390px; }
.food-photo img { width: 100%; height: 100%; object-fit: cover;
  object-position: 55% center; position: absolute; inset: 0; }
.food-photo::after { content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, #f5faf300, #153e3114); pointer-events: none; }
.food-photo-caption { position: absolute; z-index: 1; bottom: 22px; left: 22px; right: 22px;
  background: #ffffffed; border: 1px solid #fff; border-radius: 14px;
  padding: 14px 18px; color: #264653; font-size: .88rem; }
.food-hero-copy a { display: inline-block; background: #237f73; color: white;
  padding: 12px 20px; border-radius: 12px; font-weight: 600; text-decoration: none;
  margin-top: 20px; }
.food-hero-copy a:hover { background: #1c695f; }
.food-hero-copy a:focus-visible { outline: 3px solid #F4A261; outline-offset: 3px; }
.st-key-dashboard_nutrients [data-testid="stMetric"] { border-top: 3px solid #2A9D8F; }
.st-key-dashboard_nutrients [data-testid="stColumn"]:first-child [data-testid="stMetric"] {
  background: #fff8ed; border-top-color: #F4A261;
}
[data-testid="stPlotlyChart"] { border-radius: 18px; overflow: hidden; }
[data-testid="stCameraInput"] { border-radius: 18px; overflow: hidden; }
[data-testid="stImage"] img { max-width: 100%; height: auto; border-radius: 18px; }
.food-section-label { font-size: .78rem; font-weight: 700; letter-spacing: .13em;
  color: #247e73; margin: 10px 0 6px; }
.food-chip { display: inline-block; background: #fffaf0; color: #875423;
  border: 1px solid #f3debd; border-radius: 30px; padding: 6px 12px; font-size: .8rem; }
.food-feature { height: 100%; padding: 22px; background: white; border: 1px solid #e0ece6;
  border-radius: 18px; margin: 12px 0; }
.food-feature strong { display: block; color: #264653; margin-bottom: 8px; font-size: 1.05rem; }
.food-feature p { color: #506b72; font-size: .94rem; line-height: 1.7; margin: 0; }
@media (max-width: 760px) {
  .food-hero { flex-direction: column; align-items: stretch; gap: 0; min-height: 0; }
  .food-hero-copy { padding: 26px 22px; }
  .food-photo { width: 100%; min-height: 240px; }
  .food-photo-caption { bottom: 14px; left: 14px; right: 14px; padding: 10px 14px; }
  .stMainBlockContainer { padding-top: 1.5rem; }
  [data-testid="stMetric"] { padding: 14px; }
}
@media (prefers-reduced-motion: no-preference) {
  .stButton > button { transition: box-shadow .18s ease, border-color .18s ease; }
}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def render_hero():
    artwork = Path(__file__).with_name("assets") / "food_photography.jpg"
    encoded = base64.b64encode(artwork.read_bytes()).decode("ascii")
    st.markdown(
        '<section class="food-hero"><div class="food-hero-copy">'
        '<span class="food-eyebrow">FOOD &amp; NUTRITION · PROJECT 18</span>'
        "<h1>Hiểu món ăn.<br>Chăm bữa mỗi ngày.</h1>"
        "<p>Nhận diện món ăn – Ước tính dinh dưỡng – Lưu nhật ký bữa ăn.<br>"
        "Từ một tấm ảnh, bắt đầu theo dõi bữa ăn theo cách đơn giản hơn.</p>"
        '<span class="food-chip">Món bạn chọn · Khẩu phần bạn xác nhận</span><br>'
        '<a href="recognize" target="_self">Bắt đầu nhận diện ↗</a></div>'
        '<div class="food-photo">'
        f'<img src="data:image/jpeg;base64,{encoded}" '
        'alt="Ảnh thực phẩm gốc: rau củ, trái cây và các nguyên liệu trên bàn ăn">'
        '<div class="food-photo-caption">🥗 Hiểu bữa ăn hôm nay.<br>'
        "Theo dõi thói quen mỗi ngày.</div></div></section>",
        unsafe_allow_html=True,
    )
