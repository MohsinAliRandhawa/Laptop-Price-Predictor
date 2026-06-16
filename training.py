"""
=============================================================
 Laptop Price Predictor — Multi-Dataset Training Script
 Datasets: data.csv (INR) | laptopPrice.csv (INR) |
           laptops.csv (EUR) | global_laptop_selling_data.csv (EUR)
 Output  : laptop_price_model.pkl  (predicts in USD)
=============================================================
"""

import sys
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  CURRENCY RATES  (to USD)
# ─────────────────────────────────────────────────────────────
INR_TO_USD = 0.012   # 1 INR = 0.012 USD
EUR_TO_USD = 1.08    # 1 EUR = 1.08 USD

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def clean_brand(b):
    b = str(b).strip().lower()
    brand_map = {
        "asus": "Asus", "hp": "HP", "lenovo": "Lenovo", "dell": "Dell",
        "acer": "Acer", "msi": "MSI", "apple": "Apple", "samsung": "Samsung",
        "infinix": "Infinix", "lg": "LG", "razer": "Razer",
        "gigabyte": "Gigabyte", "avita": "Other", "medion": "Other",
        "alurin": "Other", "pccom": "Other", "wings": "Other",
        "primebook": "Other", "realme": "Other", "mi": "Other",
    }
    for key, val in brand_map.items():
        if key in b:
            return val
    return "Other"

def parse_gen(txt):
    """Extract processor generation number."""
    txt = str(txt).lower()
    # "12th", "12700h" etc.
    m = re.search(r'(\d{1,2})(?:th|st|nd|rd)', txt)
    if m:
        return int(m.group(1))
    # e.g. "Core i7-12700H" -> 12
    m = re.search(r'i[3579]-(\d{2})', txt)
    if m:
        return int(m.group(1)[:2])
    # Ryzen generation: "ryzen 5 5600h" -> 5th gen
    m = re.search(r'ryzen\s+[3579]\s+(\d)', txt)
    if m:
        gen_digit = int(m.group(1))
        return gen_digit  # 5xxx=5, 6xxx=6, 7xxx=7
    # "not available" or apple -> 0
    return 0

def parse_cores(txt):
    """Estimate CPU core count from text."""
    txt = str(txt).lower()
    core_map = {"dual": 2, "quad": 4, "hexa": 6, "octa": 8, "deca": 10}
    for k, v in core_map.items():
        if k in txt:
            return v
    m = re.search(r'(\d+)\s*core', txt)
    if m:
        return int(m.group(1))
    # Intel i3 ~ 4-6 cores, i5 ~ 8, i7 ~ 10, i9 ~ 14+
    if "i9" in txt or "ryzen 9" in txt:
        return 14
    if "i7" in txt or "ryzen 7" in txt:
        return 10
    if "i5" in txt or "ryzen 5" in txt:
        return 8
    if "i3" in txt or "ryzen 3" in txt:
        return 4
    if "celeron" in txt or "athlon" in txt or "pentium" in txt:
        return 2
    if "apple m" in txt:
        return 8
    return 4

def parse_gpu_vram(txt):
    """Extract GPU VRAM in GB and dedicated flag."""
    txt = str(txt).lower()
    if not txt or txt in ("nan", "none", "", "0", "0 gb"):
        return 0, 0
    m = re.search(r'(\d+)\s*gb', txt)
    vram = int(m.group(1)) if m else 0
    dedicated = 0 if vram == 0 else 1
    # Check for integrated keywords
    if any(k in txt for k in ["integrated", "intel uhd", "intel iris", "radeon graphics", "apple"]):
        dedicated = 0
    return vram, dedicated

def parse_storage(val):
    """Return storage in GB as int."""
    val = str(val).strip().lower().replace(" ", "")
    if "tb" in val:
        m = re.search(r'(\d+\.?\d*)', val)
        return int(float(m.group(1)) * 1024) if m else 0
    m = re.search(r'(\d+)', val)
    return int(m.group(1)) if m else 0

def parse_ram(val):
    m = re.search(r'(\d+)', str(val))
    return int(m.group(1)) if m else 0

def parse_warranty(val):
    val = str(val).lower()
    if "no" in val or "0" in val:
        return 0
    m = re.search(r'(\d+)', val)
    return int(m.group(1)) if m else 1

def normalize_os(val):
    val = str(val).lower()
    if "mac" in val or "ios" in val:
        return "macOS"
    if "chrome" in val:
        return "Chrome OS"
    if "dos" in val or "linux" in val or "ubuntu" in val or "nos" in val:
        return "Other"
    return "Windows"

def normalize_ram_type(val):
    val = str(val).upper().replace(" ", "")
    types = ["LPDDR5X", "LPDDR5", "LPDDR4X", "LPDDR4", "DDR5", "DDR4", "DDR3", "UNIFIED"]
    for t in types:
        if t in val:
            return t.replace("UNIFIED", "Unified")
    return "DDR4"

def normalize_storage_type(val):
    val = str(val).lower()
    if "ssd" in val or "nvme" in val or "emmc" in val or "flash" in val:
        return "SSD"
    return "Hard-Disk"

def calc_spec_rating(ram, storage, gpu_vram, proc_gen, cpu_cores):
    """Approximate spec rating 50-90."""
    score = 50
    score += min(ram / 64 * 15, 15)
    score += min(storage / 2048 * 10, 10)
    score += min(gpu_vram / 16 * 10, 10)
    score += min(proc_gen / 14 * 8, 8)
    score += min(cpu_cores / 24 * 7, 7)
    return round(min(score, 90), 1)

# ─────────────────────────────────────────────────────────────
#  DATASET 1: data.csv  (India, INR)
# ─────────────────────────────────────────────────────────────
print("Loading dataset 1: data.csv ...")
df1_raw = pd.read_csv("data.csv")
rows1 = []
for _, r in df1_raw.iterrows():
    try:
        price_inr = float(str(r.get("price", 0)).replace(",", ""))
        if price_inr < 10000:
            continue
        price_usd = price_inr * INR_TO_USD

        cpu_txt = str(r.get("CPU", ""))
        gpu_txt = str(r.get("GPU", ""))

        # Storage
        rom_txt = str(r.get("ROM", "0"))
        storage_gb = parse_storage(rom_txt)

        # Resolution
        rw = int(r.get("resolution_width", 1920) or 1920)
        rh = int(r.get("resolution_height", 1080) or 1080)
        total_px = rw * rh

        ram  = parse_ram(r.get("Ram", 8))
        pgen = parse_gen(cpu_txt)
        pcores = parse_cores(cpu_txt)
        vram, has_ded = parse_gpu_vram(gpu_txt)
        disp = float(r.get("display_size", 15.6) or 15.6)
        warr = parse_warranty(r.get("warranty", 1))
        spec = float(r.get("spec_rating", 70) or 70)
        rtype = normalize_ram_type(r.get("Ram_type", "DDR4"))
        stype = normalize_storage_type(r.get("ROM_type", "SSD"))
        os_f  = normalize_os(r.get("OS", "Windows"))
        brand = clean_brand(r.get("brand", "Other"))

        rows1.append({
            "brand_clean": brand, "Ram_GB": ram, "Ram_type": rtype,
            "ROM_GB": storage_gb, "ROM_type": stype,
            "GPU_VRAM_GB": vram, "Has_Dedicated_GPU": has_ded,
            "Processor_Gen": pgen, "CPU_Cores": pcores,
            "Total_Pixels": total_px, "display_size": disp,
            "spec_rating": spec, "warranty": warr, "OS_Family": os_f,
            "price_usd": price_usd
        })
    except Exception:
        continue
df1 = pd.DataFrame(rows1)
print(f"  -> {len(df1)} valid rows from data.csv")

# ─────────────────────────────────────────────────────────────
#  DATASET 2: laptopPrice.csv  (India, INR)
# ─────────────────────────────────────────────────────────────
print("Loading dataset 2: laptopPrice.csv ...")
df2_raw = pd.read_csv("laptopPrice.csv")
rows2 = []
for _, r in df2_raw.iterrows():
    try:
        price_inr = float(str(r.get("Price", 0)).replace(",", ""))
        if price_inr < 10000:
            continue
        price_usd = price_inr * INR_TO_USD

        cpu_name = str(r.get("processor_name", ""))
        cpu_gen  = str(r.get("processor_gnrtn", ""))
        gpu_vram_raw = str(r.get("graphic_card_gb", "0 GB"))

        pgen  = parse_gen(cpu_gen) if "not available" not in cpu_gen.lower() else 0
        pcores = parse_cores(cpu_name)

        # SSD + HDD storage
        ssd = parse_storage(str(r.get("ssd", "0")))
        hdd = parse_storage(str(r.get("hdd", "0")))
        storage_gb = max(ssd + hdd, ssd, hdd)

        ram   = parse_ram(r.get("ram_gb", 8))
        vram  = parse_ram(gpu_vram_raw)
        has_ded = 1 if vram > 0 else 0
        rtype = normalize_ram_type(r.get("ram_type", "DDR4"))
        stype = "SSD" if ssd >= hdd else "Hard-Disk"
        os_f  = normalize_os(r.get("os", "Windows"))
        warr  = parse_warranty(r.get("warranty", "1 year"))
        brand = clean_brand(r.get("brand", "Other"))
        spec  = calc_spec_rating(ram, storage_gb, vram, pgen, pcores)

        rows2.append({
            "brand_clean": brand, "Ram_GB": ram, "Ram_type": rtype,
            "ROM_GB": storage_gb, "ROM_type": stype,
            "GPU_VRAM_GB": vram, "Has_Dedicated_GPU": has_ded,
            "Processor_Gen": pgen, "CPU_Cores": pcores,
            "Total_Pixels": 1920 * 1080, "display_size": 15.6,
            "spec_rating": spec, "warranty": warr, "OS_Family": os_f,
            "price_usd": price_usd
        })
    except Exception:
        continue
df2 = pd.DataFrame(rows2)
print(f"  -> {len(df2)} valid rows from laptopPrice.csv")

# ─────────────────────────────────────────────────────────────
#  DATASET 3: laptops.csv  (Europe, EUR)
# ─────────────────────────────────────────────────────────────
print("Loading dataset 3: laptops.csv ...")
df3_raw = pd.read_csv("laptops.csv")
rows3 = []
for _, r in df3_raw.iterrows():
    try:
        price_eur = float(str(r.get("Final Price", 0)).replace(",", ""))
        if price_eur < 100:
            continue
        price_usd = price_eur * EUR_TO_USD

        cpu_txt = str(r.get("CPU", ""))
        gpu_txt = str(r.get("GPU", ""))

        ram   = parse_ram(r.get("RAM", 8))
        storage_gb = parse_storage(str(r.get("Storage", "512")))
        pgen  = parse_gen(cpu_txt)
        pcores = parse_cores(cpu_txt)
        vram, has_ded = parse_gpu_vram(gpu_txt)
        disp  = float(r.get("Screen", 15.6) or 15.6)
        warr  = 1  # not available in this dataset
        rtype = "DDR4"  # not available
        stype = normalize_storage_type(str(r.get("Storage type", "SSD")))
        os_f  = normalize_os("")  # not available, default Windows
        brand = clean_brand(r.get("Brand", "Other"))
        spec  = calc_spec_rating(ram, storage_gb, vram, pgen, pcores)

        rows3.append({
            "brand_clean": brand, "Ram_GB": ram, "Ram_type": rtype,
            "ROM_GB": storage_gb, "ROM_type": stype,
            "GPU_VRAM_GB": vram, "Has_Dedicated_GPU": has_ded,
            "Processor_Gen": pgen, "CPU_Cores": pcores,
            "Total_Pixels": 1920 * 1080, "display_size": disp,
            "spec_rating": spec, "warranty": warr, "OS_Family": os_f,
            "price_usd": price_usd
        })
    except Exception:
        continue
df3 = pd.DataFrame(rows3)
print(f"  -> {len(df3)} valid rows from laptops.csv")

# ─────────────────────────────────────────────────────────────
#  DATASET 4: global_laptop_selling_data.csv  (Europe, EUR)
# ─────────────────────────────────────────────────────────────
print("Loading dataset 4: global_laptop_selling_data.csv ...")
df4_raw = pd.read_csv("global_laptop_selling_data.csv", encoding="latin1")
rows4 = []
for _, r in df4_raw.iterrows():
    try:
        price_eur = float(str(r.get("Price_euros", 0)).replace(",", "."))
        if price_eur < 100:
            continue
        price_usd = price_eur * EUR_TO_USD

        cpu_txt = str(r.get("Cpu", ""))
        gpu_txt = str(r.get("Gpu", ""))
        mem_txt = str(r.get("Memory", "512GB SSD"))
        ram_txt = str(r.get("Ram", "8GB"))

        # Screen resolution
        res_txt = str(r.get("ScreenResolution", "1920x1080"))
        res_m   = re.search(r'(\d{3,4})[xX](\d{3,4})', res_txt)
        if res_m:
            rw, rh = int(res_m.group(1)), int(res_m.group(2))
        else:
            rw, rh = 1920, 1080

        ram   = parse_ram(ram_txt)
        storage_gb = parse_storage(mem_txt)
        pgen  = parse_gen(cpu_txt)
        pcores = parse_cores(cpu_txt)
        vram, has_ded = parse_gpu_vram(gpu_txt)
        disp  = float(r.get("Inches", 15.6) or 15.6)
        warr  = 1
        rtype = "DDR4"
        stype = normalize_storage_type(mem_txt)
        os_f  = normalize_os(str(r.get("OpSys", "")))
        brand = clean_brand(r.get("Company", "Other"))
        spec  = calc_spec_rating(ram, storage_gb, vram, pgen, pcores)

        rows4.append({
            "brand_clean": brand, "Ram_GB": ram, "Ram_type": rtype,
            "ROM_GB": storage_gb, "ROM_type": stype,
            "GPU_VRAM_GB": vram, "Has_Dedicated_GPU": has_ded,
            "Processor_Gen": pgen, "CPU_Cores": pcores,
            "Total_Pixels": rw * rh, "display_size": disp,
            "spec_rating": spec, "warranty": warr, "OS_Family": os_f,
            "price_usd": price_usd
        })
    except Exception:
        continue
df4 = pd.DataFrame(rows4)
print(f"  -> {len(df4)} valid rows from global_laptop_selling_data.csv")

# ─────────────────────────────────────────────────────────────
#  DATASET 5: priceoye_laptops_version_2.csv  (Pakistan, PKR)
# ─────────────────────────────────────────────────────────────
print("Loading dataset 5: priceoye_laptops_version_2.csv ...")
df5_raw = pd.read_csv("priceoye_laptops_version_2.csv")
PKR_TO_USD = 1 / 278.24
rows5 = []
for _, r in df5_raw.iterrows():
    try:
        price_pkr = float(str(r.get("Discounted Price", 0)).replace(",", ""))
        if pd.isna(price_pkr) or price_pkr < 50000:
            continue
        price_usd = price_pkr * PKR_TO_USD

        name_txt = str(r.get("Name", ""))
        brand = clean_brand(str(r.get("Brand", "Other")))
        
        ram = parse_ram(name_txt)
        storage_gb = parse_storage(name_txt)
        pgen = parse_gen(name_txt)
        pcores = parse_cores(name_txt)
        vram, has_ded = parse_gpu_vram(name_txt)
        
        disp = 15.6
        rw, rh = 1920, 1080
        warr = 1
        rtype = "DDR4"
        stype = "SSD"
        os_f = "Windows" if brand.lower() != "apple" else "macOS"
        spec = calc_spec_rating(ram, storage_gb, vram, pgen, pcores)

        rows5.append({
            "brand_clean": brand, "Ram_GB": ram, "Ram_type": rtype,
            "ROM_GB": storage_gb, "ROM_type": stype,
            "GPU_VRAM_GB": vram, "Has_Dedicated_GPU": has_ded,
            "Processor_Gen": pgen, "CPU_Cores": pcores,
            "Total_Pixels": rw * rh, "display_size": disp,
            "spec_rating": spec, "warranty": warr, "OS_Family": os_f,
            "price_usd": price_usd
        })
    except Exception:
        continue
df5 = pd.DataFrame(rows5)
print(f"  -> {len(df5)} valid rows from priceoye_laptops_version_2.csv")

# ─────────────────────────────────────────────────────────────
#  MERGE ALL DATASETS
# ─────────────────────────────────────────────────────────────
combined = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)
print(f"\nTotal combined rows: {len(combined)}")

# Remove extreme outliers (price < $100 or > $8000)
combined = combined[(combined["price_usd"] >= 100) & (combined["price_usd"] <= 8000)]
combined.dropna(inplace=True)
print(f"After cleaning: {len(combined)} rows")
print(f"Price range: ${combined['price_usd'].min():.0f} - ${combined['price_usd'].max():.0f}")

# ─────────────────────────────────────────────────────────────
#  ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────────────────────
CATEGORICAL = ["brand_clean", "Ram_type", "ROM_type", "OS_Family"]
FEATURES    = ["Ram_GB", "ROM_GB", "GPU_VRAM_GB", "Has_Dedicated_GPU",
               "Processor_Gen", "CPU_Cores", "Total_Pixels", "display_size",
               "spec_rating", "warranty",
               "brand_clean", "Ram_type", "ROM_type", "OS_Family"]

label_encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# ─────────────────────────────────────────────────────────────
#  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X = combined[FEATURES]
y = combined["price_usd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"\nTraining set: {len(X_train)} rows")
print(f"Test set    : {len(X_test)} rows")

# ─────────────────────────────────────────────────────────────
#  TRAIN MODEL — Random Forest (300 trees)
# ─────────────────────────────────────────────────────────────
print("\nTraining Random Forest (300 trees) ...")
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n  R2 Score : {r2:.4f}")
print(f"  MAE      : ${mae:.2f}")
print(f"  Avg error: ${mae:.0f} USD  (~PKR {mae*280:.0f}  ~INR {mae/0.012:.0f})")

# 5-fold cross validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
print(f"  CV R2    : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Feature importance
feat_imp = pd.Series(model.feature_importances_, index=FEATURES)
print("\nTop 5 features:")
for feat, imp in feat_imp.nlargest(5).items():
    print(f"  {feat:25s} {imp:.3f}")

# ─────────────────────────────────────────────────────────────
#  SAVE MODEL BUNDLE
# ─────────────────────────────────────────────────────────────
bundle = {
    "model"          : model,
    "label_encoders" : label_encoders,
    "features"       : FEATURES,
    "categorical"    : CATEGORICAL,
    "price_unit"     : "USD",
    "mae_usd"        : mae,
    "r2"             : r2,
    "training_rows"  : len(combined),
}
with open("laptop_price_model.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nModel saved: laptop_price_model.pkl")
print(f"Total training rows: {len(combined)}")
print("Price unit: USD (web app will convert to PKR/INR via live API)")
print("\nDone!")
