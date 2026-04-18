import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Airbnb Asset Clustering", layout="wide", page_icon="🏢")

# --- DATA ENGINE ---
@st.cache_data
def get_full_data():
    url = "https://data.insideairbnb.com/italy/lombardia/bergamo/2025-09-29/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    # Pre-processing & Cleaning
    df_proc = df[['id', 'room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude']].copy()
    df_proc.dropna(subset=['room_type', 'accommodates', 'amenities', 'price'], inplace=True)
    df_proc['price'] = pd.to_numeric(df_proc['price'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
    df_proc.dropna(subset=['price'], inplace=True)
    
    # Amenities Extraction
    def parse_am(val):
        return [i.strip() for i in str(val).strip('[]{}').replace('"', '').replace("'", "").split(',') if i.strip()]
    
    df_proc['am_list'] = df_proc['amenities'].apply(parse_am)
    all_am = [item for sublist in df_proc['am_list'] for item in sublist]
    valid_am = (pd.Series(all_am).value_counts() / len(df_proc))
    valid_am = valid_am[valid_am > 0.05].index.tolist()
    
    for am in valid_am:
        df_proc[f"am_{am}"] = df_proc['am_list'].apply(lambda x: 1 if am in x else 0)
        
    return df_proc, valid_am

df_main, amenity_names = get_full_data()

# --- MODELING ENGINE ---
def run_model(k_value):
    room_encoded = pd.get_dummies(df_main['room_type'], prefix='room')
    am_cols = [c for c in df_main.columns if c.startswith('am_')]
    X = pd.concat([df_main[['accommodates']], room_encoded, df_main[am_cols]], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    return X_pca, clusters, X_scaled, kmeans.inertia_

# --- UI LAYOUT ---
st.title("🏛️ Dashboard Manajemen Aset Strategis: Airbnb Bergamo")
st.sidebar.header("Konfigurasi Model")
k_select = st.sidebar.selectbox("Jumlah Klaster (Sinkronisasi Orange)", options=[2, 3, 4, 5], index=0)

X_pca, clusters, X_scaled, inertia_val = run_model(k_select)
df_main['Cluster'] = clusters
df_main['Cluster_Label'] = "Kelompok " + df_main['Cluster'].astype(str)

tabs = st.tabs(["📋 Persiapan & EDA", "📐 Validasi Model", "🧩 Visualisasi Klaster", "💰 Validasi Harga", "📊 Ringkasan Profil"])

# TAB 1: PERSIAPAN & EDA
with tabs[0]:
    st.header("Fase 1: Data Preparation & Exploratory Analysis")
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.subheader("Log Pembersihan Data")
        st.info(f"Metode: Baris dengan data kosong dihapus (Drop).\nTotal Aset Setelah Pembersihan: {len(df_main)} Unit.")
        st.write("Statistik Deskriptif Fitur Utama:")
        st.dataframe(df_main[['accommodates', 'price']].describe())
    with col_eda2:
        st.subheader("Distribusi Kapasitas Aset")
        fig_eda = px.histogram(df_main, x="accommodates", color="room_type", barmode="group")
        st.plotly_chart(fig_eda, use_container_width=True)

# TAB 2: VALIDASI MODEL
with tabs[1]:
    st.header("Fase 2: Metrik Presisi Matematis")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Elbow Method")
        # Pre-calculated for speed
        it_list = [run_model(i)[3] for i in range(2, 7)]
        fig_elb = px.line(x=range(2, 7), y=it_list, markers=True, labels={'x':'K', 'y':'Inertia'})
        st.plotly_chart(fig_elb, use_container_width=True)
    with col_v2:
        st.subheader("Silhouette Analysis")
        score = silhouette_score(X_pca, clusters)
        st.metric("Silhouette Score", f"{score:.3f}")
        st.write("Skor ini menunjukkan seberapa baik tiap aset berada di klasternya dibandingkan klaster lain.")

# TAB 3: VISUALISASI KLASTER
with tabs[2]:
    st.header("Fase 3: Geospasial & Karakteristik Aset")
    
    # Row 1: Map & Room Type
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Peta Persebaran Lokasi")
        fig_map = px.scatter_mapbox(df_main, lat="latitude", lon="longitude", color="Cluster_Label", size="accommodates",
                                    hover_data=['price', 'room_type'], zoom=10, mapbox_style="carto-positron", height=500)
    
        st.plotly_chart(fig_map, use_container_width=True)
    with c2:
        st.subheader("Dominasi Tipe Kamar")
        fig_room = px.pie(df_main, names='room_type', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_room, use_container_width=True)

    # Row 2: Interactive Amenities
    st.subheader("🛠️ Fasilitas Unggulan Interaktif")
    selected_cluster = st.selectbox("Pilih Klaster untuk Analisis Fasilitas:", options=df_main['Cluster_Label'].unique())
    cl_data = df_main[df_main['Cluster_Label'] == selected_cluster]
    am_cols = [c for c in df_main.columns if c.startswith('am_')]
    top_am = cl_data[am_cols].mean().sort_values(ascending=False).head(10)
    top_am.index = [i.replace('am_', '') for i in top_am.index]
    fig_am = px.bar(top_am, orientation='h', labels={'value':'Frekuensi (%)', 'index':'Fasilitas'}, title=f"Top 10 Fasilitas di {selected_cluster}")
    st.plotly_chart(fig_am, use_container_width=True)

# TAB 4: VALIDASI HARGA
with tabs[3]:
    st.header("Uji Realitas: Validasi Menggunakan Harga")
    st.markdown("Harga diisolasi selama pelatihan. Tab ini membuktikan apakah klaster memiliki nilai ekonomi yang berbeda.")
    fig_price = px.box(df_main, x="Cluster_Label", y="price", color="Cluster_Label", 
                       points="outliers", notched=True, title="Distribusi Harga Pasar per Kelompok Aset")
    fig_price.update_yaxes(range=[0, df_main['price'].quantile(0.95)])
    st.plotly_chart(fig_price, use_container_width=True)

# TAB 5: RINGKASAN & PROFIL
with tabs[4]:
    st.header("Strategic Summary & Economic Profiling")
    
    summary = df_main.groupby('Cluster_Label').agg({
        'id': 'count',
        'accommodates': 'mean',
        'price': ['mean', 'median']
    }).round(2)
    summary.columns = ['Jumlah Unit', 'Rata-rata Kapasitas', 'Expected Price (Mean)', 'Expected Price (Median)']
    summary['Persentase Pasar'] = (summary['Jumlah Unit'] / len(df_main) * 100).round(2).astype(str) + '%'
    
    st.subheader("Tabel Profiling Klaster")
    st.dataframe(summary)
    
    st.subheader("Narasi Profil Properti")
    for cl in sorted(df_main['Cluster_Label'].unique()):
        avg_p = summary.loc[cl, 'Expected Price (Mean)']
        avg_c = summary.loc[cl, 'Rata-rata Kapasitas']
        count = summary.loc[cl, 'Jumlah Unit']
        st.write(f"**{cl}**: Terdiri dari {count} aset. Memiliki profil kapasitas rata-rata {avg_c} tamu dengan **Expected Price** di kisaran ${avg_p}.")
