import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Asset Management Analytics", layout="wide", page_icon="🏢")

# --- DATA ENGINE (V5 - Asset Focused) ---
@st.cache_data
def load_and_process_data():
    url = "https://data.insideairbnb.com/italy/lombardia/bergamo/2025-09-29/data/listings.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    cols = ['id', 'room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude']
    df_proc = df[cols].copy()
    
    # Cleaning & Data Integrity
    df_proc.dropna(subset=['room_type', 'accommodates', 'amenities', 'price', 'latitude', 'longitude'], inplace=True)
    df_proc['price'] = pd.to_numeric(df_proc['price'].astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')
    df_proc['accommodates'] = pd.to_numeric(df_proc['accommodates'], errors='coerce')
    df_proc.dropna(subset=['price', 'accommodates'], inplace=True)
    
    # Amenities Extraction
    def parse_am(val):
        cleaned = str(val).strip('[]{}').replace('"', '').replace("'", "")
        return [i.strip() for i in cleaned.split(',') if i.strip()]
    
    df_proc['am_list'] = df_proc['amenities'].apply(parse_am)
    
    mlb = MultiLabelBinarizer()
    am_encoded = mlb.fit_transform(df_proc['am_list'])
    am_df = pd.DataFrame(am_encoded, columns=mlb.classes_, index=df_proc.index)
    
    # Filter > 5% Frequency
    valid_cols = am_df.columns[am_df.mean() > 0.05]
    am_df_filtered = am_df[valid_cols]
    
    return df_proc, am_df_filtered

with st.spinner("Mensinkronkan Data Aset..."):
    df_main, df_amenities = load_and_process_data()

# --- MODELING ENGINE ---
def run_full_modeling(k_val):
    room_encoded = pd.get_dummies(df_main['room_type'], prefix='room', dtype=float)
    X = pd.concat([df_main[['accommodates']], room_encoded, df_amenities], axis=1).astype(float)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate Silhouette Scores for Range 2-6
    sil_scores = []
    for i in range(2, 7):
        km_test = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels = km_test.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))
    
    # Final Model
    km_final = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    clusters = km_final.fit_predict(X_pca)
    
    return X_pca, clusters, sil_scores

# --- UI LAYOUT ---
st.title("🏛️ Sistem Informasi Manajemen Aset Strategis: Airbnb Bergamo")
st.sidebar.header("Konfigurasi Model")
k_select = st.sidebar.selectbox("Sinkronisasi Klaster (K)", options=[2, 3, 4, 5, 6], index=0)

X_pca, clusters, all_sil_scores = run_full_modeling(k_select)
df_main['Cluster'] = clusters
df_main['Cluster_Label'] = "Kelompok " + df_main['Cluster'].astype(str)

tabs = st.tabs(["📊 EDA & Persiapan", "📐 Validasi Model", "📍 Peta Lokasi", "💰 Validasi Ekonomi", "📋 Ringkasan Profil"])

# --- TAB 1: EDA & PERSIAPAN ---
with tabs[0]:
    st.header("Analisis Eksploratif Data (EDA)")
    c_eda1, c_eda2 = st.columns(2)
    with c_eda1:
        fig_eda_p = px.histogram(df_main, x="price", color="room_type", title="Distribusi Harga Pasar")
        fig_eda_p.update_xaxes(range=[0, df_main['price'].quantile(0.95)])
        st.plotly_chart(fig_eda_p, use_container_width=True)
    with c_eda2:
        fig_eda_a = px.box(df_main, x="room_type", y="accommodates", title="Kapasitas Berdasarkan Tipe Kamar")
        st.plotly_chart(fig_eda_a, use_container_width=True)
    
    st.divider()
    st.subheader("Fase Persiapan & Pembersihan Data")
    st.markdown(f"""
    - **Pembersihan Data:** Baris dengan data kosong pada fitur inti telah dihapus. Total aset bersih: **{len(df_main)} unit**.
    - **Penanganan Anomali:** Nilai harga non-numerik telah dibersihkan. Outlier harga di atas persentil 95 dipotong pada visualisasi untuk akurasi interpretasi.
    - **Konversi Fitur:** Fasilitas (Amenities) diurai menjadi matriks biner. Kategori 'Room Type' diubah melalui proses One-Hot Encoding.
    """)

# --- TAB 2: VALIDASI MODEL ---
with tabs[1]:
    st.header("Metrik Validasi & Karakteristik Klaster")
    
    # Row 1: Silhouette & Distribution
    c_val1, c_val2 = st.columns(2)
    with c_val1:
        fig_sil = px.line(x=range(2, 7), y=all_sil_scores, markers=True, title="Silhouette Score per Jumlah K",
                          labels={'x': 'Jumlah Klaster (K)', 'y': 'Silhouette Score'})
        st.plotly_chart(fig_sil, use_container_width=True)
    with c_val2:
        fig_dist = px.bar(df_main['Cluster_Label'].value_counts().reset_index(), x='Cluster_Label', y='count',
                          title="Distribusi Jumlah Aset per Kelompok", color='Cluster_Label')
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Row 2: PCA & Room Type
    c_val3, c_val4 = st.columns(2)
    with c_val3:
        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        df_pca['Cluster'] = df_main['Cluster_Label'].values
        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title="Visualisasi Kedekatan Aset (PCA)")
        st.plotly_chart(fig_pca, use_container_width=True)
    with c_val4:
        fig_room = px.pie(df_main, names='room_type', color_discrete_sequence=px.colors.qualitative.Pastel,
                          title="Komposisi Tipe Kamar")
        st.plotly_chart(fig_room, use_container_width=True)
    
    # Row 3: Amenities
    st.divider()
    st.subheader("🛠️ Fasilitas Unggulan Interaktif")
    sel_cl = st.selectbox("Pilih Kelompok untuk Profil Fasilitas:", sorted(df_main['Cluster_Label'].unique()))
    cl_idx = df_main[df_main['Cluster_Label'] == sel_cl].index
    top_amenities = df_amenities.loc[cl_idx].mean().sort_values(ascending=False).head(10)
    fig_am_bar = px.bar(top_amenities, orientation='h', labels={'value': 'Frekuensi', 'index': 'Fasilitas'},
                        title=f"10 Fasilitas Dominan pada {sel_cl}")
    st.plotly_chart(fig_am_bar, use_container_width=True)

# --- TAB 3: PETA LOKASI ---
with tabs[2]:
    st.header("Peta Lokasi Geospasial Properti")
    fig_map = px.scatter_mapbox(df_main, lat="latitude", lon="longitude", color="Cluster_Label",
                                hover_data={'price': True, 'room_type': True}, zoom=11, 
                                mapbox_style="carto-positron", height=600)
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 4: VALIDASI EKONOMI ---
with tabs[3]:
    st.header("Uji Realitas: Validasi Nilai Ekonomi")
    col_v_econ1, col_v_econ2 = st.columns(2)
    with col_v_econ1:
        fig_box_p = px.box(df_main, x="Cluster_Label", y="price", color="Cluster_Label", title="Rentang Harga per Kelompok")
        fig_box_p.update_yaxes(range=[0, df_main['price'].quantile(0.95)])
        st.plotly_chart(fig_box_p, use_container_width=True)
    with col_v_econ2:
        # Tambahan Visualisasi: Price vs Accommodates
        fig_scat_econ = px.scatter(df_main, x="accommodates", y="price", color="Cluster_Label", 
                                   trendline="ols", title="Korelasi Harga vs Kapasitas")
        fig_scat_econ.update_yaxes(range=[0, df_main['price'].quantile(0.95)])
        st.plotly_chart(fig_scat_econ, use_container_width=True)

# --- TAB 5: RINGKASAN PROFIL ---
with tabs[4]:
    st.header("Ringkasan Strategis & Profil Portofolio")
    
    # Tabel Ringkasan dengan angka dibulatkan
    summary = df_main.groupby('Cluster_Label').agg({
        'id': 'count',
        'accommodates': 'mean',
        'price': 'mean'
    })
    summary.columns = ['Jumlah Unit', 'Rata-rata Kapasitas', 'Expected Price (Mean)']
    
    # Pembulatan
    summary['Jumlah Unit'] = summary['Jumlah Unit'].round(0).astype(int)
    summary['Rata-rata Kapasitas'] = summary['Rata-rata Kapasitas'].round(0).astype(int)
    summary['Expected Price (Mean)'] = summary['Expected Price (Mean)'].round(2)
    summary['Pangsa Pasar (%)'] = ((summary['Jumlah Unit'] / len(df_main)) * 100).round(2)
    
    st.dataframe(summary, use_container_width=True)
    
    st.divider()
    st.subheader("Narasi Karakteristik Aset")
    for cl in sorted(df_main['Cluster_Label'].unique()):
        row = summary.loc[cl]
        # Ambil Top 5 Fasilitas untuk narasi
        idx = df_main[df_main['Cluster_Label'] == cl].index
        top_fs = df_amenities.loc[idx].mean().sort_values(ascending=False).head(5).index.tolist()
        top_fs_str = ", ".join([f.replace('_', ' ') for f in top_fs])
        
        st.markdown(f"""
        ### **{cl}**
        Kelompok ini mencakup **{row['Pangsa Pasar (%)']}%** dari total portofolio dengan total **{row['Jumlah Unit']} unit**. 
        Secara profil fisik, aset ini memiliki kapasitas rata-rata **{row['Rata-rata Kapasitas']} tamu**. 
        
        **Kelengkapan Fasilitas:**
        Aset dalam kelompok ini sangat identik dengan ketersediaan fasilitas unggulan seperti **{top_fs_str}**. 
        Berdasarkan kombinasi karakteristik fisik dan utilitas tersebut, indikasi **Expected Price** yang wajar berada pada angka **${row['Expected Price (Mean)']}** per malam.
        """)
