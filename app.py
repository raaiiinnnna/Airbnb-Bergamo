import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="Airbnb Analytics", layout="wide", page_icon="🏡")
st.title("🏡 Dasbor Analisis Klaster Aset Properti Airbnb")

# Mengunduh & Memproses Data dengan Cache (Agar Web Tidak Lemot)
@st.cache_data
def load_and_process_data():
    url = "https://data.insideairbnb.com/italy/lombardia/bergamo/2025-09-29/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    # Pilih Fitur & Lokasi
    df_sel = df[['id', 'room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude']].copy().dropna()
    df_sel['price'] = pd.to_numeric(df_sel['price'].astype(str).str.replace('[\$,]', '', regex=True), errors='coerce')
    df_sel.dropna(subset=['price'], inplace=True)
    
    meta = df_sel[['id', 'price', 'latitude', 'longitude']].copy()
    
    # Amenities Parsing
    def parse_amenities(val):
        cleaned = str(val).strip('[]{}').replace('"', '').replace("'", "")
        return [item.strip() for item in cleaned.split(',') if item.strip()]
        
    df_sel['amenities_list'] = df_sel['amenities'].apply(parse_amenities)
    all_amenities = [item for sublist in df_sel['amenities_list'] for item in sublist]
    
    amenity_percentages = pd.Series(all_amenities).value_counts() / len(df_sel)
    valid_amenities = amenity_percentages[amenity_percentages > 0.05].index.tolist()
    
    for am in valid_amenities:
        safe_name = "".join([c if c.isalnum() else "_" for c in am])
        df_sel[f"am_{safe_name}"] = df_sel['amenities_list'].apply(lambda x: 1 if am in x else 0)
        
    # Encoding & PCA
    room_encoded = pd.get_dummies(df_sel['room_type'], prefix='room', drop_first=True)
    am_cols = [c for c in df_sel.columns if c.startswith('am_')]
    X = pd.concat([df_sel[['accommodates']], room_encoded, df_sel[am_cols]], axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return meta, X_pca, len(valid_amenities)

# Mulai Proses
with st.spinner("Mengunduh dan mempersiapkan data dari server Airbnb..."):
    meta_data, X_pca, total_am = load_and_process_data()

st.sidebar.header("Pengaturan Model")
k = st.sidebar.slider("Pilih Jumlah Klaster (K)", min_value=2, max_value=8, value=3)

# Eksekusi K-Means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
meta_data['Cluster'] = kmeans.fit_predict(X_pca)
meta_data['Cluster_Label'] = "Klaster " + meta_data['Cluster'].astype(str)

# Membuat Tab Visualisasi
tab1, tab2, tab3 = st.tabs(["📊 Analisis Harga & PCA", "📍 Peta Lokasi Geospasial", "⚙️ Ringkasan Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Harga Asli per Klaster")
        fig_box, ax_box = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=meta_data, x='Cluster_Label', y='price', palette='Set2', ax=ax_box)
        ax_box.set_ylim(0, meta_data['price'].quantile(0.95))
        ax_box.set_ylabel("Harga ($)")
        st.pyplot(fig_box)
        
    with col2:
        st.subheader("Sebaran Kedekatan Aset (PCA)")
        df_pca_vis = pd.DataFrame(data=X_pca[:, :2], columns=['PC1', 'PC2'])
        df_pca_vis['Cluster_Label'] = meta_data['Cluster_Label'].values
        fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df_pca_vis, x='PC1', y='PC2', hue='Cluster_Label', palette='Set1', alpha=0.7, ax=ax_scatter)
        ax_scatter.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_scatter)

with tab2:
    st.subheader("Pemetaan Properti Berdasarkan Klaster")
    st.markdown("Peta interaktif ini menunjukkan titik koordinat asli properti. Warna mewakili klaster tempat aset tersebut berada.")
    fig_map = px.scatter_mapbox(
        meta_data, lat="latitude", lon="longitude", color="Cluster_Label", 
        hover_name="id", hover_data={"price": True, "latitude": False, "longitude": False, "Cluster_Label": False}, 
        color_discrete_sequence=px.colors.qualitative.Set1,
        zoom=10, height=600
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.subheader("Informasi Kinerja Mesin")
    colA, colB, colC = st.columns(3)
    colA.metric("Total Aset Dianalisis", f"{len(meta_data)} Unit")
    colB.metric("Fasilitas Valid (>5%)", f"{total_am} Fasilitas")
    colC.metric("Komponen PCA (90%)", f"{X_pca.shape[1]} Dimensi")
    
    st.write("Rata-Rata Harga per Klaster:")
    st.dataframe(meta_data.groupby('Cluster_Label')['price'].agg(['mean', 'median', 'count']).round(2))
