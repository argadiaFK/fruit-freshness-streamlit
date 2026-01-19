import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN (WAJIB PALING ATAS)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fruit Quality Assurance (Dark)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (DARK MODE, NEON ACCENTS & ANIMASI)
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* --- GLOBAL DARK THEME --- */
    /* Memaksa background gelap dan teks terang */
    .stApp {
        background-color: #0E1117 !important;
        color: #E0E0E0 !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Memastikan semua heading dan teks terbaca */
    h1, h2, h3, h4, h5, h6, label, p, div, span {
        color: #E0E0E0 !important;
    }

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: #262730 !important;
        border-right: 1px solid #333333;
    }
    /* Memperbaiki warna teks di elemen sidebar */
    [data-testid="stSidebar"] * {
        color: #E0E0E0 !important;
    }

    /* --- RESULT CARD (KARTU HASIL) --- */
    .result-container {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        animation: fadeIn 0.6s ease-in-out;
    }

    /* --- BADGES (INDIKATOR STATUS) --- */
    /* Fresh: Latar Hijau Tua, Teks Hijau Neon */
    .badge-fresh {
        background-color: #064E3B;      
        color: #34D399 !important;      
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid #059669;
        display: inline-block;
        box-shadow: 0 0 5px rgba(52, 211, 153, 0.2);
    }
    
    /* Rotten: Latar Merah Tua, Teks Merah Neon */
    .badge-rotten {
        background-color: #7F1D1D;      
        color: #F87171 !important;      
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        border: 1px solid #DC2626;
        display: inline-block;
        box-shadow: 0 0 5px rgba(248, 113, 113, 0.2);
    }

    /* --- TABEL --- */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        font-size: 0.95em;
        font-family: sans-serif;
        background-color: #161B22;
    }
    .styled-table thead tr {
        background-color: #21262D;
        text-align: left;
    }
    .styled-table thead th {
        color: #8B949E !important; /* Header Abu-abu */
        padding: 12px 15px;
        border-bottom: 1px solid #30363D;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #30363D;
    }
    .styled-table td {
        padding: 12px 15px;
        color: #C9D1D9 !important;
    }

    /* --- TOMBOL UTAMA --- */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white !important;
        border: 1px solid rgba(27, 31, 35, 0.15);
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2c974b;
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(46, 160, 67, 0.5);
    }
    /* Fix warna teks di dalam tombol agar tetap putih */
    div.stButton > button:first-child p {
        color: white !important;
    }

    /* --- ANIMASI --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* CATATAN: Baris di bawah ini dikomentari agar MENU (titik tiga) TETAP MUNCUL.
       Jika ingin menyembunyikannya (mode kiosk), hilangkan tanda komentar slash-bintang.
    */
    /* #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;} 
    */
    
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. SIDEBAR (PANEL KONTROL)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h3 style='color: #58A6FF !important;'>⚙️ Control Panel</h3>", unsafe_allow_html=True)
    st.markdown("<div style='height: 2px; background-color: #30363D; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # Upload File
    uploaded_files = st.file_uploader(
        "Upload Gambar (JPG/PNG)", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- PENGATURAN DETEKSI ---
    st.markdown("**Parameter AI:**")
    
    # Slider Confidence
    conf_threshold = st.slider(
        "Confidence (Akurasi)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.40, 
        step=0.05,
        help="Batas minimal keyakinan model agar objek dianggap valid."
    )

    # Slider IoU (Kunci mengatasi Numpuk)
    iou_threshold = st.slider(
        "IoU Threshold (Anti-Numpuk)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.50, 
        step=0.05,
        help="Semakin KECIL nilainya, semakin agresif sistem membuang kotak yang bertumpuk."
    )
    
    st.markdown("---")
    st.caption("ℹ️ **Tips:** Jika hasil deteksi masih menumpuk, **turunkan nilai IoU Threshold**.")

# -----------------------------------------------------------------------------
# 4. FUNGSI LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Pastikan file best.pt ada di folder yang sama
        model = YOLO("best.pt")
        return model
    except Exception as e:
        return None

model = load_model()

# -----------------------------------------------------------------------------
# 5. HALAMAN UTAMA (HEADER)
# -----------------------------------------------------------------------------
# Judul dengan efek glow teks biru
st.markdown("<h1 style='text-align: center; color: #58A6FF !important; margin-bottom: 5px; text-shadow: 0 0 10px rgba(88, 166, 255, 0.3);'>🛡️ Fruit Quality Control</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8B949E !important; margin-bottom: 40px;'>Sistem Analisis Kualitas Buah Berbasis AI (Fresh vs Rotten)</p>", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ ERROR KRITIS: File model 'best.pt' tidak ditemukan. Silakan upload file model terlebih dahulu.")
    st.stop()

# -----------------------------------------------------------------------------
# 6. LOGIKA UTAMA & OUTPUT
# -----------------------------------------------------------------------------
if uploaded_files:
    # Layout tombol di tengah
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        analyze_btn = st.button("🚀 JALANKAN ANALISIS", type="primary", use_container_width=True)

    if analyze_btn:
        # 1. Animasi Progress Bar
        progress_text = "Memindai kualitas objek..."
        my_bar = st.progress(0, text=progress_text)
        
        for p in range(100):
            time.sleep(0.005) # Simulasi loading (bisa dihapus agar lebih cepat)
            my_bar.progress(p + 1)
        
        my_bar.empty() # Hilangkan bar setelah selesai

        # 2. Loop Proses Gambar
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            
            # --- INFERENSI (DETEKSI) ---
            # agnostic_nms=True -> Mencegah deteksi ganda beda kelas (misal Fresh & Rotten di titik sama)
            # iou -> Mengontrol tumpang tindih kotak
            results = model(image, conf=conf_threshold, iou=iou_threshold, agnostic_nms=True)
            
            # Render visualisasi (Plot)
            res_plotted = results[0].plot()[:, :, ::-1] # Konversi warna BGR ke RGB
            
            # Ambil data kotak (Boxes)
            boxes = results[0].boxes
            table_rows = ""
            rotten_detected = False
            
            # Proses Data Teks
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id].upper() # Nama kelas kapital (misal: FRESH_APPLE)
                    conf = float(box.conf[0])
                    
                    # Logika HTML Badge Berdasarkan Nama Kelas
                    if "ROTTEN" in label:
                        rotten_detected = True
                        status_badge = f"<span class='badge-rotten'>{label}</span>"
                    elif "FRESH" in label:
                        status_badge = f"<span class='badge-fresh'>{label}</span>"
                    else:
                        # Fallback jika nama label tidak mengandung fresh/rotten
                        status_badge = f"<span style='background:#21262D; color:#C9D1D9; padding:6px 12px; border-radius:6px; border:1px solid #30363D;'>{label}</span>"
                    
                    # Tambahkan baris ke tabel HTML
                    table_rows += f"<tr><td>{status_badge}</td><td style='color:#C9D1D9 !important;'>{conf:.1%}</td></tr>"
            else:
                table_rows = "<tr><td colspan='2' style='text-align:center; color: #8B949E; font-style:italic;'>Tidak ada objek terdeteksi</td></tr>"

            # Tentukan Warna Kartu (Merah jika ada busuk, Hijau jika aman)
            border_color = "#FF7B72" if rotten_detected else "#238636"
            status_text = "❌ DITEMUKAN CACAT/BUSUK" if rotten_detected else "✅ KUALITAS PRIMA (FRESH)"
            status_text_color = "#FF7B72" if rotten_detected else "#3FB950"

            # 3. TAMPILKAN KARTU HASIL (HTML)
            st.markdown(f"""
            <div class="result-container" style="border-left: 5px solid {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid #30363D; padding-bottom: 10px;">
                    <h3 style="margin:0; color: {status_text_color} !important; font-size: 1.3rem;">{status_text}</h3>
                    <span style="font-size: 0.9em; color: #8B949E !important;">File: {uploaded_file.name}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Kolom Gambar dan Tabel Info
            col_img, col_info = st.columns([1.2, 1])
            
            with col_img:
                st.image(res_plotted, use_column_width=True, caption="Visualisasi AI")
            
            with col_info:
                st.markdown("<p style='color: #8B949E !important; font-size: 0.9em; margin-bottom: 5px;'>RINCIAN DETEKSI:</p>", unsafe_allow_html=True)
                
                # Render Tabel Custom
                st.markdown(f"""
                <table class="styled-table">
                    <thead>
                        <tr>
                            <th>Kondisi & Objek</th>
                            <th>Akurasi</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
                """, unsafe_allow_html=True)
                
                # Alert Tambahan Jika Busuk
                if rotten_detected:
                    st.markdown("""
                    <div style="margin-top: 15px; padding: 12px; background-color: rgba(127, 29, 29, 0.3); border: 1px solid #7F1D1D; border-radius: 6px;">
                        <strong style="color: #FF7B72 !important;">⚠️ TINDAKAN DIPERLUKAN:</strong><br>
                        <span style="color: #FFA198 !important; font-size: 0.9em;">Objek tidak lolos QC. Segera pisahkan item ini dari batch produksi.</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True) # Penutup div result-container

else:
    # Tampilan Awal (Placeholder)
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px; background-color: #161B22; border-radius: 12px; border: 1px dashed #30363D; margin-top: 20px;">
        <h3 style="color: #58A6FF !important;">Masukan Gambar</h3>
        <p style="color: #8B949E !important;">Silakan upload gambar buah (strawberry, apel, pisang) pada panel kontrol di sebelah kiri.</p>
    </div>
    """, unsafe_allow_html=True)