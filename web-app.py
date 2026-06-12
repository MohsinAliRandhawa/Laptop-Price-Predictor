import pickle
import re
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS — premium dark-mode look
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Root background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.20));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(12px);
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Section cards ── */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.6rem 1.6rem 0.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(8px);
}
.card-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Labels ── */
label, .stSelectbox label, .stSlider label, .stRadio label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
}

/* ── Selectbox / input styling ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
}
.stSelectbox > div > div:hover {
    border-color: #818cf8 !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    color: white !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.85rem 1.5rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35);
    margin-top: 1rem;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(168,85,247,0.55);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.18));
    border: 2px solid rgba(99,102,241,0.6);
    border-radius: 20px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}
.result-label {
    color: #94a3b8;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-price {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.result-range {
    color: #64748b;
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

/* ── Feature pill ── */
.pill {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 0.25rem 0.8rem;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 0.2rem;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.07);
}

/* ── Fade in ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL BUNDLE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path="laptop_price_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model    = bundle["model"]
encoders = bundle["label_encoders"]
features = bundle["features"]
cats     = bundle["categorical"]

# ──────────────────────────────────────────────────────────────────────────────
#  HELPER — run inference
# ──────────────────────────────────────────────────────────────────────────────
def predict_price(sample: dict) -> float:
    row = pd.DataFrame([sample])
    for col in cats:
        le  = encoders[col]
        val = str(row[col].iloc[0])
        row[col] = le.transform([val])[0] if val in le.classes_ else -1
    return float(model.predict(row[features])[0])

# ──────────────────────────────────────────────────────────────────────────────
#  LIVE CURRENCY RATES  (cached 1 hour — free API, no key needed)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_live_rates():
    """
    Fetch live INR -> PKR and INR -> USD rates from open.er-api.com.
    Falls back to hardcoded rates if internet/API is unavailable.
    Returns: (pkr_rate, usd_rate, fetched_at, is_live)
    """
    FALLBACK_PKR = 3.52
    FALLBACK_USD = 0.012
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/INR",
            timeout=5
        )
        data = resp.json()
        if data.get("result") == "success":
            pkr = data["rates"]["PKR"]
            usd = data["rates"]["USD"]
            fetched_at = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
            return pkr, usd, fetched_at, True
    except Exception:
        pass
    # Fallback if API fails
    return FALLBACK_PKR, FALLBACK_USD, "Unavailable (using fallback)", False

# ──────────────────────────────────────────────────────────────────────────────
#  STATIC OPTION LISTS  (derived from training data knowledge)
# ──────────────────────────────────────────────────────────────────────────────
BRANDS = ["Acer", "Apple", "Asus", "Dell", "HP", "Infinix",
          "Lenovo", "MSI", "Samsung", "Other"]

RAM_OPTIONS     = [4, 8, 16, 32, 64]          # GB
RAM_TYPES       = ["DDR3", "DDR4", "DDR5", "LPDDR4", "LPDDR4X", "LPDDR5", "LPDDR5X", "Unified"]
STORAGE_OPTIONS = [32, 64, 128, 256, 512, 1024, 2048]  # GB (1024=1TB, 2048=2TB)
STORAGE_LABELS  = {32:"32 GB", 64:"64 GB", 128:"128 GB", 256:"256 GB",
                   512:"512 GB", 1024:"1 TB", 2048:"2 TB"}
STORAGE_TYPES   = ["SSD", "Hard-Disk"]
DISPLAY_SIZES   = [11.6, 13.3, 13.4, 13.6, 14.0, 15.3, 15.6, 16.0, 16.1, 17.3]
RESOLUTIONS     = {
    "1366 × 768  (HD)"       : (1366, 768),
    "1920 × 1080 (FHD)"      : (1920, 1080),
    "1920 × 1200 (FHD+)"     : (1920, 1200),
    "2160 × 1440 (2K)"       : (2160, 1440),
    "2560 × 1440 (QHD)"      : (2560, 1440),
    "2560 × 1600 (QHD+)"     : (2560, 1600),
    "2880 × 1800 (3K)"       : (2880, 1800),
    "2880 × 1864 (3K)"       : (2880, 1864),
    "3200 × 2000 (3.2K)"     : (3200, 2000),
    "3840 × 2400 (4K)"       : (3840, 2400),
}
GPU_VRAM_OPTIONS = {
    "Integrated (0 GB)"  : 0,
    "2 GB Dedicated"     : 2,
    "4 GB Dedicated"     : 4,
    "6 GB Dedicated"     : 6,
    "8 GB Dedicated"     : 8,
    "16 GB Dedicated"    : 16,
}
OS_OPTIONS = ["Windows 11", "Windows 10", "macOS", "Chrome OS", "Android", "Other"]

# ──────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>💻 Laptop Price Predictor</h1>
    <p>Configure your ideal laptop below and get an AI-powered price estimate instantly.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN LAYOUT — 3 columns
# ──────────────────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([1.1, 1.1, 1.0], gap="large")

# ╔══════════════════════════════╗
# ║   LEFT — Brand & Memory      ║
# ╚══════════════════════════════╝
with col_left:
    st.markdown('<div class="card"><div class="card-title">🏷️ &nbsp;Brand & Identity</div>', unsafe_allow_html=True)

    brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Asus"))

    os_family = st.selectbox("Operating System", OS_OPTIONS, index=0)

    warranty = st.selectbox("Warranty (Years)", [1, 2, 3], index=0,
                             format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Memory card ──
    st.markdown('<div class="card"><div class="card-title">🧠 &nbsp;Memory & Storage</div>', unsafe_allow_html=True)

    ram_gb = st.selectbox("RAM", RAM_OPTIONS, index=2,
                           format_func=lambda x: f"{x} GB")

    ram_type = st.selectbox("RAM Type", RAM_TYPES, index=RAM_TYPES.index("DDR5"))

    storage_val = st.selectbox("Storage Capacity", STORAGE_OPTIONS, index=4,
                                format_func=lambda x: STORAGE_LABELS[x])

    storage_type = st.selectbox("Storage Type", STORAGE_TYPES, index=0)

    st.markdown('</div>', unsafe_allow_html=True)

# ╔══════════════════════════════╗
# ║   MIDDLE — CPU & GPU         ║
# ╚══════════════════════════════╝
with col_mid:
    st.markdown('<div class="card"><div class="card-title">⚡ &nbsp;Processor</div>', unsafe_allow_html=True)

    proc_gen = st.slider("Processor Generation", min_value=7, max_value=14,
                          value=12, step=1,
                          help="e.g. 12 = 12th Gen Intel / 7th Gen AMD Ryzen")

    cpu_cores = st.selectbox("CPU Cores", [2, 4, 6, 8, 10, 12, 14, 16, 24],
                              index=3, format_func=lambda x: f"{x} Cores")

    spec_rating = st.slider("Overall Spec Rating", min_value=50.0, max_value=90.0,
                             value=70.0, step=0.5,
                             help="Aggregate performance score (50 = budget, 90 = flagship)")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GPU card ──
    st.markdown('<div class="card"><div class="card-title">🎮 &nbsp;Graphics</div>', unsafe_allow_html=True)

    gpu_label = st.selectbox("GPU / VRAM", list(GPU_VRAM_OPTIONS.keys()), index=2)
    gpu_vram  = GPU_VRAM_OPTIONS[gpu_label]
    has_dedicated_gpu = 1 if gpu_vram > 0 else 0

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Display card ──
    st.markdown('<div class="card"><div class="card-title">🖥️ &nbsp;Display</div>', unsafe_allow_html=True)

    display_size = st.selectbox("Screen Size (inches)", DISPLAY_SIZES,
                                 index=DISPLAY_SIZES.index(15.6),
                                 format_func=lambda x: f'{x}"')

    res_label = st.selectbox("Resolution", list(RESOLUTIONS.keys()),
                              index=list(RESOLUTIONS.keys()).index("1920 × 1080 (FHD)"))
    res_w, res_h = RESOLUTIONS[res_label]
    total_pixels = res_w * res_h

    st.markdown("</div>", unsafe_allow_html=True)

# ╔══════════════════════════════╗
# ║   RIGHT — Summary & Result   ║
# ╚══════════════════════════════╝
with col_right:
    st.markdown('<div class="card"><div class="card-title">📋 &nbsp;Configuration Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="line-height: 2; color: #cbd5e1; font-size: 0.88rem;">
    <b style="color:#a5b4fc">Brand</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{brand}<br>
    <b style="color:#a5b4fc">OS</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{os_family}<br>
    <b style="color:#a5b4fc">CPU Gen</b> &nbsp;&nbsp;&nbsp;&nbsp;{proc_gen}th Gen<br>
    <b style="color:#a5b4fc">Cores</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{cpu_cores}<br>
    <b style="color:#a5b4fc">RAM</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ram_gb} GB {ram_type}<br>
    <b style="color:#a5b4fc">Storage</b> &nbsp;&nbsp;&nbsp;&nbsp;{STORAGE_LABELS[storage_val]} {storage_type}<br>
    <b style="color:#a5b4fc">GPU</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{gpu_label}<br>
    <b style="color:#a5b4fc">Display</b> &nbsp;&nbsp;&nbsp;&nbsp;{display_size}" — {res_w}×{res_h}<br>
    <b style="color:#a5b4fc">Rating</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{spec_rating}<br>
    <b style="color:#a5b4fc">Warranty</b> &nbsp;&nbsp;{warranty} Yr
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict button ──
    predict_clicked = st.button("🔮  Predict Price", use_container_width=True)

    if predict_clicked:
        sample = {
            "Ram_GB"           : ram_gb,
            "ROM_GB"           : storage_val,
            "GPU_VRAM_GB"      : gpu_vram,
            "Has_Dedicated_GPU": has_dedicated_gpu,
            "Processor_Gen"    : proc_gen,
            "CPU_Cores"        : cpu_cores,
            "Total_Pixels"     : total_pixels,
            "display_size"     : display_size,
            "spec_rating"      : spec_rating,
            "warranty"         : warranty,
            "brand_clean"      : brand,
            "Ram_type"         : ram_type,
            "ROM_type"         : storage_type,
            "OS_Family"        : os_family,
        }

        with st.spinner("Fetching live rates & calculating..."):
            price_inr = predict_price(sample)
            INR_TO_PKR, INR_TO_USD, rate_time, is_live = get_live_rates()

        price_pkr = price_inr * INR_TO_PKR
        price_usd = price_inr * INR_TO_USD

        low_pkr  = max(0, price_pkr * 0.90)
        high_pkr = price_pkr * 1.10

        # ── Price tier (based on PKR) ──
        if price_pkr < 100_000:
            tier, tier_color, tier_icon = "Budget", "#34d399", "💚"
        elif price_pkr < 200_000:
            tier, tier_color, tier_icon = "Mid-Range", "#60a5fa", "💙"
        elif price_pkr < 350_000:
            tier, tier_color, tier_icon = "Premium", "#c084fc", "💜"
        else:
            tier, tier_color, tier_icon = "Flagship", "#f472b6", "🩷"

        rate_badge_color = "#34d399" if is_live else "#f59e0b"
        rate_badge_text  = "LIVE" if is_live else "OFFLINE"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">PKR {price_pkr:,.0f}</div>
            <div style="margin-top:0.8rem;">
                <span style="background:rgba(0,0,0,0.3); border:1px solid {tier_color};
                             border-radius:999px; padding:0.2rem 0.9rem;
                             color:{tier_color}; font-size:0.82rem; font-weight:600;">
                    {tier_icon} {tier} Tier
                </span>
            </div>
            <div class="result-range">
                PKR range &nbsp;|&nbsp; {low_pkr:,.0f} &mdash; {high_pkr:,.0f}
            </div>
            <div style="margin-top:1.2rem; display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
                <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
                            border-radius:12px; padding:0.6rem 1.2rem; text-align:center;">
                    <div style="color:#94a3b8; font-size:0.72rem; letter-spacing:0.08em;">INDIAN RUPEE</div>
                    <div style="color:#e2e8f0; font-size:1.1rem; font-weight:700;">INR {price_inr:,.0f}</div>
                </div>
                <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
                            border-radius:12px; padding:0.6rem 1.2rem; text-align:center;">
                    <div style="color:#94a3b8; font-size:0.72rem; letter-spacing:0.08em;">US DOLLAR</div>
                    <div style="color:#e2e8f0; font-size:1.1rem; font-weight:700;">$ {price_usd:,.0f}</div>
                </div>
            </div>
            <div style="margin-top:0.9rem; display:flex; align-items:center;
                        justify-content:center; gap:0.5rem; flex-wrap:wrap;">
                <span style="background:rgba(0,0,0,0.3); border:1px solid {rate_badge_color};
                             border-radius:999px; padding:0.15rem 0.6rem;
                             color:{rate_badge_color}; font-size:0.7rem; font-weight:700;">
                    {rate_badge_text}
                </span>
                <span style="color:#475569; font-size:0.72rem;">
                    1 INR = {INR_TO_PKR:.4f} PKR &bull; $ {INR_TO_USD:.5f} &nbsp;|&nbsp; {rate_time}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Placeholder before prediction ──
    else:
        st.markdown("""
        <div style="margin-top:1.5rem; text-align:center; padding:2rem 1rem;
                    border:1px dashed rgba(99,102,241,0.35); border-radius:16px;
                    color:#475569;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">🔮</div>
            <div style="font-size:0.9rem;">
                Configure the specs on the left<br>then click <b style="color:#818cf8">Predict Price</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER INFO
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.82rem;">
        <div style="color:#818cf8; font-weight:600; margin-bottom:0.3rem;">Model</div>
        Random Forest Regressor<br>300 trees &bull; R² = 0.86
    </div>""", unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.82rem;">
        <div style="color:#818cf8; font-weight:600; margin-bottom:0.3rem;">Training Data</div>
        893 Indian market laptops<br>across 15+ brands
    </div>""", unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.82rem;">
        <div style="color:#818cf8; font-weight:600; margin-bottom:0.3rem;">Accuracy</div>
        Avg. error ~ PKR 42,832<br>within &plusmn;10% range shown
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.76rem; margin-top:1rem; padding-bottom:1.5rem;">
    Prices converted from Indian market data. 1 INR = 3.52 PKR = $0.012 USD. Rates may vary.
</div>
""", unsafe_allow_html=True)
