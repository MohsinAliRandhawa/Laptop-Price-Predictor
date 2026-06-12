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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #e2e8f0;
}

/* ── Root background ── */
.stApp {
    background: #050505 !important;
}

/* Add an absolute aurora glow behind the main content */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%; left: -10%;
    width: 60vw; height: 60vw;
    background: radial-gradient(circle, rgba(45,212,191,0.06) 0%, rgba(0,0,0,0) 60%);
    z-index: -1;
    pointer-events: none;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -20%; right: -10%;
    width: 70vw; height: 70vw;
    background: radial-gradient(circle, rgba(139,92,246,0.07) 0%, rgba(0,0,0,0) 60%);
    z-index: -1;
    pointer-events: none;
}

/* ── Hero banner ── */
.hero {
    background: rgba(15, 15, 17, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.05), 0 8px 32px rgba(0, 0, 0, 0.6);
    border-radius: 24px;
    padding: 3rem 2rem;
    margin-bottom: 2.5rem;
    text-align: center;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    position: relative;
    overflow: hidden;
}

/* Top accent line on hero */
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.5), rgba(139,92,246,0.5), transparent);
}

.hero h1 {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f8fafc 20%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #64748b;
    font-size: 1.15rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Section cards ── */
.card {
    background: rgba(18, 18, 20, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.03);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.card:hover {
    border-color: rgba(255, 255, 255, 0.08);
}
.card-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b5cf6;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Labels ── */
label, .stSelectbox label, .stSlider label, .stRadio label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
}

/* ── Selectbox / input styling ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease;
}
.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:hover {
    border-color: rgba(139,92,246,0.5) !important;
    background: rgba(255,255,255,0.04) !important;
}
.stSelectbox > div > div:focus,
.stSelectbox > div > div[data-baseweb="select"]:focus-within {
    box-shadow: 0 0 0 1px #8b5cf6 !important;
    border-color: #8b5cf6 !important;
}

/* ── Dropdown Menu Styling (Tricky in Streamlit but targeting general baseWeb elements) ── */
ul[data-baseweb="menu"] {
    background-color: #121214 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}
li[data-baseweb="menu-item"] {
    color: #e2e8f0 !important;
}
li[data-baseweb="menu-item"]:hover {
    background-color: rgba(139,92,246,0.15) !important;
}

/* ── Slider ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #2dd4bf, #8b5cf6) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: #ffffff !important;
    color: #050505 !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    padding: 0.9rem 1.5rem;
    border: 1px solid #ffffff !important;
    border-radius: 14px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-top: 1rem;
    box-shadow: 0 0 20px rgba(255,255,255,0.1);
}
.stButton > button:hover {
    background: transparent !important;
    color: #ffffff !important;
    box-shadow: 0 0 30px rgba(139,92,246,0.4);
    border-color: rgba(139,92,246,0.8) !important;
    transform: translateY(-2px);
}
.stButton > button:active {
    transform: translateY(1px);
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    margin-top: 2rem;
    position: relative;
    overflow: hidden;
    animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), inset 0 1px 1px rgba(255,255,255,0.05);
}
.result-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(45,212,191,0.8), rgba(139,92,246,0.8), transparent);
}
.result-label {
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.result-price {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #2dd4bf 0%, #8b5cf6 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 1rem;
    filter: drop-shadow(0 4px 12px rgba(139,92,246,0.2));
}
.result-range {
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.8rem;
}

/* ── Feature pill ── */
.pill {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #f8fafc;
    margin: 0.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.05);
    margin: 3rem 0;
}

/* ── Animations ── */
@keyframes slideUpFade {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}

/* ── Hide Streamlit branding ── */
#MainMenu, header, footer { visibility: hidden !important; }
.stApp > header { display: none !important; }

/* ── Reduce main padding ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1000px !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL BUNDLE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_bundle(path="laptop_price_model.pkl", cache_buster=4):
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
    Fetch live USD -> PKR rate from open.er-api.com.
    Falls back to hardcoded rates if internet/API is unavailable.
    Returns: (usd_to_pkr, fetched_at, is_live)
    """
    FALLBACK_PKR = 280.0
    try:
        resp = requests.get(
            "https://open.er-api.com/v6/latest/USD",
            timeout=5
        )
        data = resp.json()
        if data.get("result") == "success":
            pkr = data["rates"]["PKR"]
            fetched_at = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
            return pkr, fetched_at, True
    except Exception:
        pass
    return FALLBACK_PKR, "Unavailable (using fallback)", False

# ──────────────────────────────────────────────────────────────────────────────
#  STATIC OPTION LISTS  (derived from training data knowledge)
# ──────────────────────────────────────────────────────────────────────────────
BRANDS = ["Acer", "Apple", "Asus", "Dell", "HP", "Infinix",
          "Lenovo", "MSI", "Samsung", "Other"]

RAM_OPTIONS     = [4, 8, 16, 32, 64]          # GB
STORAGE_OPTIONS = [32, 64, 128, 256, 512, 1024, 2048]  # GB
STORAGE_LABELS  = {32:"32 GB", 64:"64 GB", 128:"128 GB", 256:"256 GB",
                   512:"512 GB", 1024:"1 TB", 2048:"2 TB"}
STORAGE_TYPES   = ["SSD", "Hard-Disk"]
DISPLAY_SIZES   = [11.6, 13.3, 13.4, 13.6, 14.0, 15.3, 15.6, 16.0, 16.1, 17.3]
RESOLUTIONS     = {
    "1366 x 768  (HD)"       : (1366, 768),
    "1920 x 1080 (FHD)"      : (1920, 1080),
    "1920 x 1200 (FHD+)"     : (1920, 1200),
    "2160 x 1440 (2K)"       : (2160, 1440),
    "2560 x 1440 (QHD)"      : (2560, 1440),
    "2560 x 1600 (QHD+)"     : (2560, 1600),
    "2880 x 1800 (3K)"       : (2880, 1800),
    "2880 x 1864 (3K)"       : (2880, 1864),
    "3200 x 2000 (3.2K)"     : (3200, 2000),
    "3840 x 2400 (4K)"       : (3840, 2400),
}

# CPU options: label -> (processor_gen, cpu_cores)
CPU_OPTIONS = {
    "Intel Celeron N-Series (2 Cores)":        (0,  2),
    "Intel Core i3 - 11th Gen (4 Cores)":      (11, 4),
    "Intel Core i3 - 12th Gen (6 Cores)":      (12, 6),
    "Intel Core i3 - 13th Gen (6 Cores)":      (13, 6),
    "Intel Core i5 - 11th Gen (4 Cores)":      (11, 4),
    "Intel Core i5 - 12th Gen (8 Cores)":      (12, 8),
    "Intel Core i5 - 13th Gen (10 Cores)":     (13, 10),
    "Intel Core i5 - 14th Gen (10 Cores)":     (14, 10),
    "Intel Core Ultra 5 (14 Cores)":           (14, 14),
    "Intel Core i7 - 11th Gen (8 Cores)":      (11, 8),
    "Intel Core i7 - 12th Gen (10 Cores)":     (12, 10),
    "Intel Core i7 - 13th Gen (14 Cores)":     (13, 14),
    "Intel Core i7 - 14th Gen (20 Cores)":     (14, 20),
    "Intel Core Ultra 7 (16 Cores)":           (14, 16),
    "Intel Core i9 - 13th Gen (24 Cores)":     (13, 24),
    "Intel Core i9 - 14th Gen (24 Cores)":     (14, 24),
    "Intel Core Ultra 9 (16 Cores)":           (14, 16),
    "AMD Athlon (2 Cores, Entry)":             (0,  2),
    "AMD Ryzen 3 - 5th Gen (4 Cores)":         (5,  4),
    "AMD Ryzen 3 - 7th Gen (4 Cores)":         (7,  4),
    "AMD Ryzen 5 - 5th Gen (6 Cores)":         (5,  6),
    "AMD Ryzen 5 - 6th Gen (6 Cores)":         (6,  6),
    "AMD Ryzen 5 - 7th Gen (6 Cores)":         (7,  6),
    "AMD Ryzen 5 - 8th Gen (6 Cores)":         (8,  6),
    "AMD Ryzen 7 - 5th Gen (8 Cores)":         (5,  8),
    "AMD Ryzen 7 - 6th Gen (8 Cores)":         (6,  8),
    "AMD Ryzen 7 - 7th Gen (8 Cores)":         (7,  8),
    "AMD Ryzen 7 - 8th Gen (8 Cores)":         (8,  8),
    "AMD Ryzen 9 - 7th Gen (16 Cores)":        (7, 16),
    "AMD Ryzen 9 - 8th Gen (16 Cores)":        (8, 16),
    "AMD Ryzen AI 9 (12 Cores)":               (9, 12),
    "Apple M1 (8 Cores)":                      (0,  8),
    "Apple M2 (8 Cores)":                      (0,  8),
    "Apple M2 Pro (12 Cores)":                 (0, 12),
    "Apple M3 (8 Cores)":                      (0,  8),
    "Apple M3 Pro (12 Cores)":                 (0, 12),
    "Apple M3 Max (16 Cores)":                 (0, 16),
    "Apple M4 (10 Cores)":                     (0, 10),
    "Apple M4 Pro (14 Cores)":                 (0, 14),
    "Apple Intel Core i9 9th Gen (8 Cores)":   (9,  8),
}

# GPU options: label -> (vram_gb, has_dedicated)
GPU_OPTIONS_DEFAULT = {
    "Integrated Graphics":                     (0,  0),
    "Intel Arc Graphics":                      (0,  0),
    "NVIDIA GeForce GTX 1650 (4 GB)":          (4,  1),
    "NVIDIA GeForce RTX 2050 (4 GB)":          (4,  1),
    "NVIDIA GeForce RTX 3050 (4 GB)":          (4,  1),
    "NVIDIA GeForce RTX 3050 (6 GB)":          (6,  1),
    "NVIDIA GeForce RTX 3050 Ti (4 GB)":       (4,  1),
    "NVIDIA GeForce RTX 3060 (6 GB)":          (6,  1),
    "NVIDIA GeForce RTX 3070 Ti (8 GB)":       (8,  1),
    "NVIDIA GeForce RTX 4050 (6 GB)":          (6,  1),
    "NVIDIA GeForce RTX 4060 (8 GB)":          (8,  1),
    "NVIDIA GeForce RTX 4070 (8 GB)":          (8,  1),
    "NVIDIA GeForce RTX 4080 (12 GB)":         (12, 1),
    "NVIDIA GeForce RTX 4090 (16 GB)":         (16, 1),
    "AMD Radeon RX 6500M (4 GB)":              (4,  1),
    "AMD Radeon RX 6650M (8 GB)":              (8,  1),
    "AMD Radeon RX 7600S (8 GB)":              (8,  1),
}

GPU_OPTIONS_APPLE = {
    "Apple Integrated GPU (7-Core)":           (0,  0),
    "Apple Integrated GPU (8-Core)":           (0,  0),
    "Apple Integrated GPU (10-Core)":          (0,  0),
    "Apple Integrated GPU (14-Core)":          (0,  0),
    "Apple Integrated GPU (18-Core)":          (0,  0),
    "Apple Integrated GPU (40-Core)":          (0,  0),
    "AMD Radeon Pro 5500M (4 GB)":             (4,  1),
}

# Brand-specific OS filter
BRAND_OS_MAP = {
    "Apple":   ["macOS"],
    "default": ["Windows 11", "Windows 10", "Chrome OS", "Other"],
}

# Brand-specific RAM type filter
BRAND_RAM_MAP = {
    "Apple":   ["Unified", "LPDDR5"],
    "default": ["DDR4", "DDR5", "LPDDR4", "LPDDR4X", "LPDDR5", "LPDDR5X", "DDR3"],
}

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

    # Filter options based on selected brand
    os_opts    = BRAND_OS_MAP.get(brand, BRAND_OS_MAP["default"])
    ram_t_opts = BRAND_RAM_MAP.get(brand, BRAND_RAM_MAP["default"])

    os_family = st.selectbox("Operating System", os_opts, index=0)

    warranty = st.selectbox("Warranty (Years)", [1, 2, 3], index=0,
                             format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Memory card ──
    st.markdown('<div class="card"><div class="card-title">🧠 &nbsp;Memory & Storage</div>', unsafe_allow_html=True)

    ram_gb = st.selectbox("RAM", RAM_OPTIONS, index=2,
                           format_func=lambda x: f"{x} GB")

    ram_type = st.selectbox("RAM Type", ram_t_opts, index=0)

    storage_val = st.selectbox("Storage Capacity", STORAGE_OPTIONS, index=4,
                                format_func=lambda x: STORAGE_LABELS[x])

    storage_type = st.selectbox("Storage Type", STORAGE_TYPES, index=0)

    st.markdown('</div>', unsafe_allow_html=True)

# ╔══════════════════════════════╗
# ║   MIDDLE — CPU & GPU         ║
# ╚══════════════════════════════╝
with col_mid:
    st.markdown('<div class="card"><div class="card-title">⚡ &nbsp;Processor</div>', unsafe_allow_html=True)

    # CPU list filtered by brand
    cpu_keys    = [k for k in CPU_OPTIONS if "Apple" in k] if brand == "Apple" \
                  else [k for k in CPU_OPTIONS if "Apple" not in k]
    default_cpu = "Apple M2 (8 Cores)" if brand == "Apple" else "Intel Core i5 - 12th Gen (8 Cores)"
    cpu_label   = st.selectbox("CPU Model", cpu_keys,
                                index=cpu_keys.index(default_cpu) if default_cpu in cpu_keys else 0)
    proc_gen, cpu_cores = CPU_OPTIONS[cpu_label]

    st.markdown('</div>', unsafe_allow_html=True)

    # ── GPU card ──
    st.markdown('<div class="card"><div class="card-title">🎮 &nbsp;Graphics Card</div>', unsafe_allow_html=True)

    gpu_options = GPU_OPTIONS_APPLE if brand == "Apple" else GPU_OPTIONS_DEFAULT
    gpu_label   = st.selectbox("GPU Model", list(gpu_options.keys()), index=0)
    gpu_vram, has_dedicated_gpu = gpu_options[gpu_label]

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Display card ──
    st.markdown('<div class="card"><div class="card-title">🖥️ &nbsp;Display</div>', unsafe_allow_html=True)

    display_size = st.selectbox("Screen Size (inches)", DISPLAY_SIZES,
                                 index=DISPLAY_SIZES.index(15.6),
                                 format_func=lambda x: f'{x}"')

    res_label = st.selectbox("Resolution", list(RESOLUTIONS.keys()),
                              index=list(RESOLUTIONS.keys()).index("1920 x 1080 (FHD)"))
    res_w, res_h = RESOLUTIONS[res_label]
    total_pixels = res_w * res_h

    st.markdown("</div>", unsafe_allow_html=True)

# ╔══════════════════════════════╗
# ║   RIGHT — Summary & Result   ║
# ╚══════════════════════════════╝
with col_right:
    # ── Auto-calculate Spec Rating ──
    score = 50.0
    score += min(ram_gb / 64 * 15, 15)
    score += min(storage_val / 2048 * 10, 10)
    score += min(gpu_vram / 16 * 10, 10)
    score += min(proc_gen / 14 * 8, 8)
    score += min(cpu_cores / 24 * 7, 7)
    spec_rating = round(min(score, 90.0), 1)

    st.markdown('<div class="card"><div class="card-title">📋 &nbsp;Configuration Summary</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="line-height: 2; color: #cbd5e1; font-size: 0.85rem;">
    <b style="color:#a5b4fc">Brand</b> &nbsp;&nbsp;&nbsp;&nbsp;{brand}<br>
    <b style="color:#a5b4fc">OS</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{os_family}<br>
    <b style="color:#a5b4fc">CPU</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{cpu_label}<br>
    <b style="color:#a5b4fc">RAM</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ram_gb} GB {ram_type}<br>
    <b style="color:#a5b4fc">Storage</b> {STORAGE_LABELS[storage_val]} {storage_type}<br>
    <b style="color:#a5b4fc">GPU</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{gpu_label}<br>
    <b style="color:#a5b4fc">Display</b> {display_size}" - {res_w}x{res_h}<br>
    <b style="color:#a5b4fc">Rating</b> &nbsp;{spec_rating}<br>
    <b style="color:#a5b4fc">Warranty</b> {warranty} Yr
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
            price_usd = predict_price(sample)
            USD_TO_PKR, rate_time, is_live = get_live_rates()

        price_pkr = price_usd * USD_TO_PKR

        low_usd  = max(0, price_usd * 0.90)
        high_usd = price_usd * 1.10
        low_pkr  = low_usd  * USD_TO_PKR
        high_pkr = high_usd * USD_TO_PKR

        # Price tier (based on USD)
        if price_usd < 400:
            tier, tier_color, tier_icon = "Budget", "#34d399", "💚"
        elif price_usd < 900:
            tier, tier_color, tier_icon = "Mid-Range", "#60a5fa", "💙"
        elif price_usd < 1600:
            tier, tier_color, tier_icon = "Premium", "#c084fc", "💜"
        else:
            tier, tier_color, tier_icon = "Flagship", "#f472b6", "🩷"

        rate_badge_color = "#34d399" if is_live else "#f59e0b"
        rate_badge_text  = "LIVE" if is_live else "OFFLINE"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Price</div>
            <div class="result-price">$ {price_usd:,.0f} USD</div>
            <div style="margin-top:0.8rem;">
                <span style="background:rgba(0,0,0,0.3); border:1px solid {tier_color};
                             border-radius:999px; padding:0.2rem 0.9rem;
                             color:{tier_color}; font-size:0.82rem; font-weight:600;">
                    {tier_icon} {tier} Tier
                </span>
            </div>
            <div class="result-range">
                USD range &nbsp;|&nbsp; ${low_usd:,.0f} &mdash; ${high_usd:,.0f}
            </div>
            <div style="margin-top:1.2rem; display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
                <div style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
                            border-radius:12px; padding:0.6rem 1.2rem; text-align:center;">
                    <div style="color:#94a3b8; font-size:0.72rem; letter-spacing:0.08em;">PAKISTANI RUPEE</div>
                    <div style="color:#e2e8f0; font-size:1.1rem; font-weight:700;">PKR {price_pkr:,.0f}</div>
                    <div style="color:#64748b; font-size:0.65rem;">{low_pkr:,.0f} - {high_pkr:,.0f}</div>
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
                    1 USD = {USD_TO_PKR:.2f} PKR &nbsp;|&nbsp; {rate_time}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Placeholder before prediction ──
    else:
        st.markdown("""
        <div style="margin-top:1.5rem; text-align:center; padding:3rem 1rem;
                    border:1px dashed rgba(139,92,246,0.3); border-radius:24px;
                    background: rgba(15, 15, 17, 0.4);
                    color:#64748b; backdrop-filter: blur(12px);">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.8;">✨</div>
            <div style="font-size:0.95rem; font-weight:500; letter-spacing:0.02em;">
                Configure the specs on the left<br>then click <b style="color:#f8fafc">Predict Price</b>
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
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 1rem; background:rgba(255,255,255,0.02); border-radius:16px; border:1px solid rgba(255,255,255,0.05);">
        <div style="color:#8b5cf6; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; font-size:0.75rem;">Model Core</div>
        <span style="color:#e2e8f0; font-weight:600;">Random Forest</span><br>300 Trees &bull; R² = 0.73
    </div>""", unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 1rem; background:rgba(255,255,255,0.02); border-radius:16px; border:1px solid rgba(255,255,255,0.05);">
        <div style="color:#8b5cf6; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; font-size:0.75rem;">Dataset Volume</div>
        <span style="color:#e2e8f0; font-weight:600;">5,480 Profiles</span><br>Cross-referenced from 5 APIs
    </div>""", unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 1rem; background:rgba(255,255,255,0.02); border-radius:16px; border:1px solid rgba(255,255,255,0.05);">
        <div style="color:#8b5cf6; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; font-size:0.75rem;">Precision</div>
        <span style="color:#e2e8f0; font-weight:600;">$274 MAE</span><br>&plusmn;10% Tolerance bounds
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:2rem; padding-bottom:1.5rem; letter-spacing:0.02em;">
    Prices computed natively via ML. Real-time forex rates dynamically applied.
</div>
""", unsafe_allow_html=True)
