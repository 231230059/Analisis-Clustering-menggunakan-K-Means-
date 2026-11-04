# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Earthquake Clustering Dashboard", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv('D:\Clustering menggunakan K-Means\Alma_Test3_ClusteringK-Means.csv')
    return df

df = load_data()
fitur_asli = ['latitude', 'longitude', 'depth', 'mag', 'sig']
fitur_scaled = [f'{c}_scaled' for c in fitur_asli]
X = df[fitur_scaled].values

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Pengaturan")
K = st.sidebar.slider("Pilih jumlah klaster (K)", min_value=2, max_value=10, value=4)

fitur_vis = st.sidebar.multiselect(
    "Fitur untuk visualisasi scatter:",
    options=fitur_asli,
    default=['longitude', 'latitude']
)

# ---------- Judul ----------
st.title("🌍 Dashboard Clustering Gempa Bumi")
st.markdown("""
Proyek ini melakukan **clustering gempa bumi global** menggunakan algoritma **K-Means**
berdasarkan fitur `latitude`, `longitude`, `depth`, `mag`, dan `sig`.
""")

# ---------- Ringkasan dataset ----------
st.header("1. Ringkasan Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Jumlah Data", f"{len(df):,}")
c2.metric("Jumlah Fitur Numerik", len(fitur_asli))
c3.metric("Rata-rata Magnitudo", f"{df['mag'].mean():.2f}")
st.dataframe(df[fitur_asli].describe().T)

# ---------- Distribusi & Korelasi ----------
st.header("2. Distribusi & Korelasi Fitur")
tab1, tab2 = st.tabs(["Histogram", "Correlation Heatmap"])
with tab1:
    fig, ax = plt.subplots(1, len(fitur_asli), figsize=(18, 3))
    for i, col in enumerate(fitur_asli):
        ax[i].hist(df[col], bins=40, color='steelblue')
        ax[i].set_title(col)
    st.pyplot(fig)
with tab2:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[fitur_asli].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---------- Clustering ----------
st.header("3. Hasil Clustering")
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
df_plot = df.copy()
df_plot['cluster'] = labels

# Elbow & Silhouette
st.subheader("Elbow Method & Silhouette Score")
Ks = list(range(2, 11))
inertias, sils = [], []
for k in Ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, lab))

col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    ax.plot(Ks, inertias, 'bo-')
    ax.set_xlabel('K'); ax.set_ylabel('Inertia'); ax.set_title('Elbow Curve')
    st.pyplot(fig)
    st.caption("Nilai inertia turun seiring bertambahnya K. Titik 'siku' menandakan K optimal.")
with col2:
    fig, ax = plt.subplots()
    ax.plot(Ks, sils, 'ro-')
    ax.set_xlabel('K'); ax.set_ylabel('Silhouette'); ax.set_title('Silhouette Score')
    st.pyplot(fig)
    st.caption("Silhouette mendekati 1 = klaster terpisah baik.")

# Scatter
st.subheader("Visualisasi Scatter Klaster")
if len(fitur_vis) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_plot, x=fitur_vis[0], y=fitur_vis[1],
                    hue='cluster', palette='Set1', ax=ax, s=30)
    ax.set_title(f'Klaster: {fitur_vis[0]} vs {fitur_vis[1]}')
    st.pyplot(fig)
    st.caption("Warna berbeda = klaster berbeda. Titik yang mengelompok erat menandakan anggota klaster yang mirip.")
else:
    st.info("Pilih minimal 2 fitur di sidebar untuk scatter plot.")

# Tabel karakteristik klaster
st.subheader("Tabel Karakteristik Klaster")
summary = df_plot.groupby('cluster')[fitur_asli].mean().round(2)
summary['jumlah_anggota'] = df_plot['cluster'].value_counts().sort_index()
st.dataframe(summary)

fig, ax = plt.subplots(figsize=(10, 4))
summary[['depth','mag','sig']].plot(kind='bar', ax=ax)
ax.set_title('Rata-rata Fitur per Klaster')
plt.tight_layout()
st.pyplot(fig)
st.caption("Bar chart menunjukkan perbedaan karakteristik tiap klaster (mis. klaster dengan kedalaman tinggi = gempa dalam).")