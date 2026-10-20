
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Liman Kıyı Elektriği Tahmini",
    page_icon="⚓",
    layout="wide"
)

st.title("⚓ Limanlar için Kıyıdan Elektrik Talep Tahmin Aracı")
st.write(
    "2018 verileriyle geliştirilen deneysel bir makine öğrenmesi "
    "prototipi. Sonuçlar gerçek zamanlı ölçüm değildir."
)

@st.cache_data
def load_data():
    df = pd.read_csv("AE_power_use_at_berth.csv")
    hour_cols = [str(i) for i in range(1, 8761)]
    return df, hour_cols

@st.cache_data
def evaluate_record(idx):
    df, hour_cols = load_data()
    row = df.loc[idx]
    series = row[hour_cols].astype(float).to_numpy()

    data = pd.DataFrame({"power": series})
    data["lag_1"] = data["power"].shift(1)
    data["lag_24"] = data["power"].shift(24)
    data["lag_168"] = data["power"].shift(168)
    data = data.dropna()

    split = int(len(series) * 0.8)
    train_data = data.loc[data.index < split]
    test_data = data.loc[data.index >= split]

    features = ["lag_1", "lag_24", "lag_168"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(train_data[features], train_data["power"])
    predictions = model.predict(test_data[features])

    mae = mean_absolute_error(test_data["power"], predictions)
    rmse = np.sqrt(
        mean_squared_error(test_data["power"], predictions)
    )

    return mae, rmse, test_data["power"].to_numpy(), predictions

df, hour_cols = load_data()

st.sidebar.header("Tahmin ayarları")

ports = sorted(df["Port name"].dropna().unique())
selected_port = st.sidebar.selectbox("Liman", ports)

categories = sorted(
    df.loc[df["Port name"] == selected_port, "Ship category"]
    .dropna().unique()
)
selected_category = st.sidebar.selectbox("Gemi kategorisi", categories)

matches = df[
    (df["Port name"] == selected_port)
    & (df["Ship category"] == selected_category)
]

if matches.empty:
    st.error("Bu seçim için veri bulunamadı.")
    st.stop()

idx = int(matches.index[0])
row = df.loc[idx]
series = row[hour_cols].astype(float).to_numpy()

st.subheader(f"{selected_port} — {selected_category}")

st.caption(
    "Saatlik değerler 2018 veri setinden alınmıştır. "
    "Model, ilk %80 ile eğitilir ve son %20 üzerinde değerlendirilir."
)

col1, col2, col3 = st.columns(3)
col1.metric("Saatlik ortalama güç (kW)", f"{series.mean():,.1f}")
col2.metric("En yüksek güç (kW)", f"{series.max():,.1f}")
col3.metric("Pozitif değer bulunan saat", f"{int((series > 0).sum()):,}")

st.subheader("Saatlik elektrik gücü ihtiyacı")

start_hour = st.slider(
    "Başlangıç saati",
    min_value=1,
    max_value=8593,
    value=4207
)

end_hour = start_hour + 167

x = np.arange(start_hour, end_hour + 1)
y = series[start_hour - 1:end_hour]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, y, linewidth=1)
ax.set_xlabel("2018 yılındaki saat")
ax.set_ylabel("Güç ihtiyacı (kW)")
ax.set_ylim(bottom=0)
ax.ticklabel_format(axis="y", style="plain", useOffset=False)
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

st.subheader("Model değerlendirmesi")

with st.spinner("Seçilen liman ve gemi kategorisi için model eğitiliyor..."):
    mae, rmse, actual, predictions = evaluate_record(idx)

m1, m2 = st.columns(2)
m1.metric("Test MAE (kW)", f"{mae:,.2f}")
m2.metric("Test RMSE (kW)", f"{rmse:,.2f}")

st.write(
    "Model, geçmiş 1, 24 ve 168 saatlik güç değerlerini kullanır. "
    "Test verisi yılın son %20'sidir. Bu sonuç yalnızca seçilen "
    "liman-gemi kaydı için geçerlidir; tüm limanlar için doğrulanmamıştır."
)

st.subheader("Test döneminde gerçek değer ve model tahmini")

fig2, ax2 = plt.subplots(figsize=(12, 4))
n = min(168, len(actual))
ax2.plot(actual[:n], label="Veri setindeki değer", linewidth=1.5)
ax2.plot(predictions[:n], label="Model tahmini", linewidth=1.5)
ax2.set_xlabel("Test dönemindeki saat")
ax2.set_ylabel("Güç (kW)")
ax2.set_ylim(bottom=0)
ax2.ticklabel_format(axis="y", style="plain", useOffset=False)
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)
plt.close(fig2)

st.info(
    "Veri kaynağı: Stolz ve arkadaşlarının (2021) Avrupa limanlarında "
    "kıyı elektriği çalışmasının ek verileri. Değerler modellenmiş "
    "tahminlerdir; doğrudan sayaç ölçümü değildir."
)
