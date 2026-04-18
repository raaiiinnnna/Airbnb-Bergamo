import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Airbnb Asset Analytics", layout="wide", page_icon="🏢")

# --- DATA ENGINE ---
@st.cache_data
def load_and_process_data():
    url = "https://data.insideairbnb.com/italy/lombardia/bergamo/2025-09-29/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    cols = ['id', 'room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude']
    df_proc = df[cols].copy()
    df_proc.dropna(subset=['room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude'], inplace=True)
    
    # PERBAIKAN: Pastikan accommodates & price adalah angka murni untuk mencegah Error Scaling
    df_proc['accommodates'] = pd.to_numeric(df_proc['accommodates'], errors='coerce')
    df_proc['price'] = pd.to_numeric(df_proc['price'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    df_proc.dropna(subset=['price', 'accommodates'], inplace=True)
    
    def parse_am(val):
        cleaned = str(val).strip('[]{}').replace('"', '').replace("'", "")
        return [i.strip() for i in cleaned.split(',') if i.strip()]
    
    df_proc['am_list'] = df_proc['amenities'].apply(parse_am)
    
    mlb = MultiLabelBinarizer()
    am_encoded = mlb.fit_transform(df_proc['am_list'])
    
    am_df = pd.DataFrame(am_encoded, columns=mlb.classes_, index=df_proc.index)
    valid_cols = am_df.columns[am_df.mean() > 0.05]
    am_df_filtered = am_df[valid_cols]
    
    return df_proc, am_df_filtered

with st.spinner("Mensinkronkan Data dengan Server Airbnb..."):
    df_main, df_amenities = load_and_process_data()

# --- MODELING ENGINE ---
def run_kmeans(k_val):
    # PERBAIKAN: Paksa output get_dummies menjadi float agar dibaca oleh StandardScaler
    room_encoded = pd.get_dummies(df_main['room_type'], prefix='room', dtype=float)
    X = pd.concat([df_main[['accommodates']], room_encoded, df_amenities], axis=1)
    
    # PERBAIKAN UTAMA: Paksa seluruh isi matriks X menjadi angka desimal
    X = X.astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    clusters = km.fit_predict(X_pca)
    
    return X_pca, clusters, km.inertia_

# --- UI LAYOUT ---
st.title("🏛️ Dashboard Manajemen Aset: Airbnb Bergamo")
st.sidebar.header("Sinkronisasi Orange")
k_select = st.sidebar.selectbox("Pilih Jumlah Klaster (Set ke 2 untuk Orange)", options=[2, 3, 4, 5, 6], index=0)

X_pca, clusters, inertia_val = run_kmeans(k_select)
df_main['Cluster'] = clusters
df_main['Cluster_Label'] = "Kelompok " + df_main['Cluster'].astype(str)

tabs = st.tabs(["📋 Persiapan & EDA", "📐 Validasi Model", "🧩 Visualisasi Klaster", "💰 Validasi Harga", "📊 Ringkasan Profil"])

# TAB 1: PERSIAPAN & EDA
with tabs[0]:
    st.header("Fase 1: Pemahaman & Persiapan Data")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data Preparation & Cleaning")
        st.write(f"**Total Aset Setelah Pembersihan:** {len(df_main)} Unit")
        st.markdown("""
        - **Handling Missing Data:** Strategi 'Drop' diterapkan pada baris yang tidak memiliki koordinat, harga, atau fasilitas.
        - **Konversi Tipe Data:** Kolom harga dikonversi dari string ($) ke numerik desimal.
        - **Anomali:** Outlier harga tetap dipertahankan namun dipotong pada visualisasi (95 persentil) agar grafik proporsional.
        """)
    with c2:
        st.subheader("Exploratory Data Analysis (EDA)")
        fig_eda = px.histogram(df_main, x="accommodates", color="room_type", barmode="group",
                               labels={'accommodates': 'Kapasitas Tamu', 'count': 'Jumlah Aset'})
        st.plotly_chart(fig_eda, use_container_width=True)

# TAB 2: VALIDASI MODEL
with tabs[1]:
    st.header("Fase 2: Metrik Evaluasi Matematika")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Elbow Method")
        inertias = [run_kmeans(i)[2] for i in range(2, 7)]
        fig_elb = px.line(x=range(2, 7), y=inertias, markers=True, labels={'x':'K', 'y':'Inertia'})
        st.plotly_chart(fig_elb, use_container_width=True)
    with col_v2:
        st.subheader("Silhouette Score")
        score = silhouette_score(X_pca, clusters)
        st.metric("Skor Kohesi Klaster", f"{score:.3f}")
        st.info("K=2 biasanya memberikan skor tertinggi, menunjukkan pemisahan kelompok yang paling tegas.")

# TAB 3: VISUALISASI KLASTER
with tabs[2]:
    st.header("Fase 3: Geospasial & Karakteristik Aset")
    m1, m2 = st.columns([2, 1])
    with m1:
        st.subheader("Peta Lokasi Properti")
        fig_map = px.scatter_mapbox(df_main, lat="latitude", lon="longitude", color="Cluster_Label",
                                    hover_data=['price', 'room_type'], zoom=10, 
                                    mapbox_style="carto-positron", height=500)
        st.plotly_chart(fig_map, use_container_width=True)
    with m2:
        st.subheader("Visualisasi Tipe Kamar")
        fig_room = px.pie(df_main, names='room_type', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_room, use_container_width=True)

    st.subheader("🛠️ Fasilitas Unggulan Interaktif")
    selected_cl = st.selectbox("Pilih Klaster untuk Melihat Top 10 Fasilitas:", df_main['Cluster_Label'].unique())
    cl_idx = df_main[df_main['Cluster_Label'] == selected_cl].index
    top_10 = df_amenities.loc[cl_idx].mean().sort_values(ascending=False).head(10)
    fig_bar = px.bar(top_10, orientation='h', labels={'value':'Frekuensi Ketersediaan', 'index':'Fasilitas'})
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 4: VALIDASI HARGA
with tabs[3]:
    st.header("Fase 4: Validasi Nilai Ekonomi")
    st.markdown("Harga diisolasi selama pemodelan. Grafik ini membuktikan korelasi antara spesifikasi fisik aset dengan harga pasar.")
    fig_price = px.box(df_main, x="Cluster_Label", y="price", color="Cluster_Label")
    fig_price.update_yaxes(range=[0, df_main['price'].quantile(0.95)])
    st.plotly_chart(fig_price, use_container_width=True)

# TAB 5: RINGKASAN PROFIL
with tabs[4]:
    st.header("Economic Summary & Expected Price")
    summary = df_main.groupby('Cluster_Label').agg({
        'id': 'count',
        'accommodates': 'mean',
        'price': ['mean', 'median']
    }).round(2)
    summary.columns = ['Jumlah Unit', 'Rata-rata Kapasitas', 'Expected Price (Mean)', 'Expected Price (Median)']
    summary['Persentase Pasar'] = (summary['Jumlah Unit'] / len(df_main) * 100).round(2).astype(str) + '%'
    
    st.subheader("Tabel Profiling Portofolio")
    st.dataframe(summary, use_container_width=True)
    
    st.subheader("Narasi Profil Aset")
    for cl in sorted(df_main['Cluster_Label'].unique()):
        p_val = summary.loc[cl, 'Expected Price (Mean)']
        cap_val = summary.loc[cl, 'Rata-rata Kapasitas']
        st.markdown(f"- **{cl}**: Didominasi oleh aset dengan kapasitas rata-rata **{cap_val} tamu**. Berdasarkan fasilitas dan kelengkapan fisiknya, kelompok ini memiliki **Expected Price** di angka **${p_val}**.")
