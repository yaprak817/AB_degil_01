import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="İstanbul Emlak Fırsat Dedektörü", layout="centered")

MODEL_PATH = os.path.join("models", "model_bundle.pkl")
eps = 1e-9

st.title("🏠 İstanbul Emlak Fırsat Dedektörü")

# --- Model yükleme ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model bulunamadı: {MODEL_PATH}")
    st.code("deneme/\n  app.py\n  models/\n    model_bundle.pkl")
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
threshold_pct = float(bundle.get("threshold_pct", 10.0))
te_maps = bundle.get("te_maps", {}) or {}
expected_features = bundle.get("expected_features", [])
default_row = bundle.get("default_row", {})
ui_cols = bundle.get("ui_cols", {}) or {}

if not expected_features or not default_row:
    st.error("model_bundle.pkl içinde expected_features/default_row eksik. Eğitim çıktını güncellemen gerekiyor.")
    st.stop()

st.info(f"🎯 Karar eşiği: ±{threshold_pct:.2f}%")
st.caption(f"Model feature sayısı: {len(expected_features)}")

# ---------------- Helpers ----------------
def to_num(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        s = s.replace(",", ".")
        s = "".join(ch for ch in s if (ch.isdigit() or ch in ".-"))
        return float(s) if s else np.nan
    except:
        return np.nan

def parse_rooms(v):
    if v is None:
        return np.nan
    t = str(v).lower().strip()
    if t == "":
        return np.nan
    if "studio" in t or "1+0" in t:
        return 1.0
    if "+" in t:
        parts = [p.strip() for p in t.split("+")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return float(parts[0]) + float(parts[1])
    return to_num(t)

def apply_te(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    def apply_one(key):
        obj = te_maps.get(key)
        if not obj:
            return
        col = obj.get("col")
        maps = obj.get("maps") or {}
        if not col or col not in df.columns or not maps:
            return

        m_mean = maps.get("mean", {}) or {}
        m_med  = maps.get("med", {}) or {}
        m_cnt  = maps.get("cnt", {}) or {}
        g_mean = float(maps.get("global_mean", 0.0))
        g_med  = float(maps.get("global_med", 0.0))

        k = df[col].astype(str)
        df[f"te_{col}_mean"] = k.map(m_mean).fillna(g_mean).astype(float)
        df[f"te_{col}_med"]  = k.map(m_med).fillna(g_med).astype(float)
        df[f"te_{col}_cnt"]  = k.map(m_cnt).fillna(0.0).astype(float)

    apply_one("district")
    apply_one("neighborhood")
    return df

def investment_advice(listing_price: float, fair_value: float):
    if listing_price <= 0:
        return "NORMAL", 0.0
    delta_pct = (fair_value - listing_price) / (listing_price + eps) * 100
    if delta_pct > threshold_pct:
        return "FIRSAT", delta_pct
    if delta_pct < -threshold_pct:
        return "PAHALI", delta_pct
    return "NORMAL", delta_pct

def safe_set(X: pd.DataFrame, col: str, value):
    if col and col in X.columns and value is not None:
        X.at[0, col] = value

def build_input(district, neigh, area, rooms, baths, age) -> pd.DataFrame:
    # 1) default_row’dan başla
    X = pd.DataFrame([default_row]).copy()

    # 2) Kullanıcı girdilerini yaz (boşsa default kalsın)
    if ui_cols.get("district") and str(district).strip():
        safe_set(X, ui_cols["district"], str(district).strip())

    if ui_cols.get("neighborhood") and str(neigh).strip():
        safe_set(X, ui_cols["neighborhood"], str(neigh).strip())

    a = to_num(area)
    r = parse_rooms(rooms)
    b = to_num(baths)
    ag = to_num(age)

    if ui_cols.get("area") and not np.isnan(a):
        safe_set(X, ui_cols["area"], float(a))

    if ui_cols.get("rooms") and not np.isnan(r):
        safe_set(X, ui_cols["rooms"], float(r))

    if ui_cols.get("baths") and not np.isnan(b):
        safe_set(X, ui_cols["baths"], float(b))

    if ui_cols.get("age") and not np.isnan(ag):
        safe_set(X, ui_cols["age"], float(ag))

    # 3) FE kolonları modelde varsa yeniden hesapla
    a_col = ui_cols.get("area")
    r_col = ui_cols.get("rooms")
    b_col = ui_cols.get("baths")
    age_col = ui_cols.get("age")

    try:
        if a_col and a_col in X.columns and "log_area" in X.columns and pd.notna(X.at[0, a_col]):
            X.at[0, "log_area"] = float(np.log1p(float(X.at[0, a_col])))

        if a_col and r_col and a_col in X.columns and r_col in X.columns and "area_per_room" in X.columns:
            X.at[0, "area_per_room"] = float(X.at[0, a_col]) / (float(X.at[0, r_col]) + 1.0)

        if r_col and b_col and r_col in X.columns and b_col in X.columns and "room_bath_ratio" in X.columns:
            X.at[0, "room_bath_ratio"] = float(X.at[0, r_col]) / (float(X.at[0, b_col]) + 1e-3)

        if age_col and age_col in X.columns and "age_bucket" in X.columns and pd.notna(X.at[0, age_col]):
            X.at[0, "age_bucket"] = str(pd.cut(
                pd.Series([float(X.at[0, age_col])]),
                bins=[-1, 5, 15, 30, 200],
                labels=["0-5", "6-15", "16-30", "30+"]
            ).astype(str).iloc[0])
    except:
        pass

    # 4) TE ekle
    X = apply_te(X)

    # 5) Kolon sırası
    X = X[expected_features]
    return X

# ---------------- Dropdown seçenekleri (modelden) ----------------
district_options = []
neigh_options = []

try:
    d = te_maps.get("district", {})
    if d and d.get("maps") and d["maps"].get("cnt"):
        district_options = sorted(list(d["maps"]["cnt"].keys()))
except:
    district_options = []

try:
    n = te_maps.get("neighborhood", {})
    if n and n.get("maps") and n["maps"].get("cnt"):
        neigh_options = sorted(list(n["maps"]["cnt"].keys()))
except:
    neigh_options = []

# İlk seçenek boş + diğer
district_list = [""] + district_options
neigh_list = ["", "Diğer (Elle yaz)"] + neigh_options

# ---------------- UI State Init ----------------
keys = ["district_sel", "neigh_sel", "neigh_custom", "area", "rooms", "baths", "age", "listing_price"]
for k in keys:
    if k not in st.session_state:
        st.session_state[k] = ""

colA, colB = st.columns([1, 1])
with colA:
    if st.button("🧹 Formu Temizle"):
        for k in keys:
            st.session_state[k] = ""
        st.rerun()
with colB:
    st.caption("")

# ---------------- Form ----------------
st.subheader("Ev Bilgileri")

with st.form("predict_form", clear_on_submit=False):
    # District dropdown
    st.caption("📌 İlçe'yi listeden seçmek yazım hatalarını engeller.")
    district = st.selectbox(
        "District / İlçe (Dropdown)",
        options=district_list,
        index=0,
        key="district_sel",
        help="Boş bırakma; mümkünse listeden seç."
    )

    # Neighborhood dropdown + custom
    if ui_cols.get("neighborhood"):
        neigh_choice = st.selectbox(
            "Neighborhood / Mahalle (opsiyonel)",
            options=neigh_list,
            index=0,
            key="neigh_sel",
        )
        if neigh_choice == "Diğer (Elle yaz)":
            neigh_custom = st.text_input("Mahalle (Elle yaz)", value=st.session_state["neigh_custom"], key="neigh_custom")
            neighborhood = neigh_custom
        else:
            neighborhood = neigh_choice
    else:
        neighborhood = ""  # dataset'te yoksa boş

    c1, c2 = st.columns(2)
    with c1:
        area = st.text_input("m²", value=st.session_state["area"], key="area")
        rooms = st.text_input("Oda (örn: 3+1)", value=st.session_state["rooms"], key="rooms")
    with c2:
        baths = st.text_input("Banyo", value=st.session_state["baths"], key="baths")
        age = st.text_input("Bina Yaşı", value=st.session_state["age"], key="age")

    listing_price = st.text_input("İlan Fiyatı (TL)", value=st.session_state["listing_price"], key="listing_price")

    submitted = st.form_submit_button("🚀 FIRSAT MI?")

# ---------------- Predict ----------------
if submitted:
    lp = to_num(listing_price)

    if not str(district).strip():
        st.error("Lütfen District/İlçe seç.")
        st.stop()

    if np.isnan(lp) or lp <= 0:
        st.error("Lütfen geçerli bir 'İlan Fiyatı (TL)' gir.")
        st.stop()

    if lp > 2_500_000:
        st.warning("⚠️ Model 2020 fiyat seviyesinde eğitildi. Çok yüksek ilan fiyatlarında 'PAHALI' çıkması normal olabilir.")

    try:
        X_in = build_input(district, neighborhood, area, rooms, baths, age)
        pred_log = model.predict(X_in)
        fair_value = float(np.expm1(pred_log)[0])

        adv, delta = investment_advice(float(lp), fair_value)

        st.success(f"💰 Fair Value: {fair_value:,.0f} TL")
        st.write(f"📌 İlan: {float(lp):,.0f} TL")
        st.subheader(f"📣 Tavsiye: **{adv}**")
        st.caption(f"Fark: {delta:.2f}% (eşik: ±{threshold_pct:.2f}%)")

    except Exception as e:
        st.error(f"Tahmin hatası: {e}")

