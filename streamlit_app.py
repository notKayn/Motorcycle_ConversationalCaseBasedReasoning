import streamlit as st
import numpy as np
import pandas as pd
import json
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import pytz
from collections import defaultdict, Counter, deque
import streamlit.components.v1 as components
import os
import pygsheets
import tempfile


st.set_page_config(page_title="Sistem Rekomendasi Motor", layout="centered")
st.title("🏍️ Sistem Rekomendasi Motor")
st.markdown("---")



# =================== Variable Global ===================
@st.cache_data
def load_df():
    df = pd.read_excel("data_motor_excel.xlsx")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_case_vector_df():
    df = pd.read_pickle("case_vector_df.pkl")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_final_df():
    df = pd.read_pickle("final_df.pkl")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

df = load_df()
final_df = load_final_df()
case_vector_df = load_case_vector_df()
case_matrix = case_vector_df.to_numpy()

json_key = dict(st.secrets["gcp_service_account"])
with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
    json.dump(json_key, tmp)
    tmp_path = tmp.name

gc = pygsheets.authorize(service_file=tmp_path)


# =================== Fungsi segmentasi tahap proses ===================
if "step" not in st.session_state:
    st.session_state.step = "intro"
if "user_input" not in st.session_state:
    st.session_state.user_input = {}
if "selected_attrs" not in st.session_state:
    st.session_state.selected_attrs = []
if "prioritas_user" not in st.session_state:
    st.session_state.prioritas_user = {}



def step_intro():
    st.subheader("🧪 Uji Coba Aplikasi Rekomendasi Motor")

    st.markdown("""
    Halo! 👋 Terima kasih sudah bersedia ikut uji coba kecil ini.

    Ini adalah sebuah aplikasi berbentuk web sederhana yang akan menghasilkan rekomendasi motor sesuai dengan kriteria yang kamu inginkan.
    Disini, dilakukan 2 tes aplikasi: 

    - 📋 **Aplikasi 1**: Query-based — Aplikasi ini mecari model motor sangat spesifik dengan atribut/kriteria yang kamu sebut. Semakin spesifik yang kamu sebut, 
    maka semakin sulit ketersediaan model yang kamu cari.

    - 🔍 **Aplikasi 2**: Conversational Case-based Reasoning — Mirip dengan aplikasi 1, tetapi aplikasi ini menggunakan perhitungan khusus untuk
    atribut yang kamu sebut. Aplikasi ini akan memberikan model motor yang paling mirip dengan apa yang kamu sebut di awal, walau diisi dengan atribut yang
    memiliki kemungkinan tidak ditemukan model yang paling cocok sekalipun.
    
    Kamu akan mencoba kedua aplikasi ini secara bergantian, dan memberikan penilaian untuk masing-masing aplikasi berbentuk survei.

    Klik tombol di bawah ini untuk memulai.
    """)

    if st.button("➡️ Mulai"):
        st.session_state.step = "identity"
        st.rerun()

def step_identity():
    st.title("🧍 Identitas Peserta Uji Coba")

    st.markdown("""
    Untuk memastikan hasil survei ini valid dan bisa dipertanggungjawabkan, silakan isi data identitas singkat berikut.
    """)

    nama = st.text_input("📝 Nama Lengkap")
    usia = st.number_input("🎂 Usia", min_value=10, max_value=100, step=1)
    gender = st.radio("⚧️ Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)

    tingkat = st.selectbox("📊 Seberapa paham kamu dengan spesifikasi motor?", [
        "1 - Saya hanya tahu secara umum.",
        "2 - Saya cukup mengerti bagian teknisnya."
    ])

    # Optional (boleh diaktifin kalau perlu)
    email = st.text_input("📧 Email (opsional)", placeholder="Misalnya: kamu@gmail.com")

    if st.button("➡️ Lanjut ke Bagian Aplikasi 1"):
        if nama.strip() and usia:
            st.session_state.user_identity = {
                "nama": nama.strip(),
                "usia": usia,
                "gender": gender,
                "tingkat_pemahaman": tingkat,
                "email": email if email else None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.step = "intro_query_based"
            st.rerun()
        else:
            st.warning("⚠️ Nama dan usia wajib diisi terlebih dahulu.")

def step_intro_query_based():
    st.subheader("🔍 Aplikasi 1: Sistem Rekomendasi Query-Based")

    st.markdown("""
    Seperti yang yang dijelaskan di awal, aplikasi ini hanya memberikan rekomendasi motor sangat spesifik dengan apa yang kamu sebutkan. 
    Cara kerja dari aplikasi 1, simplenya gini:
    - kamu pasang atribut A, sistem akan kasih kamu semua motor yang memiliki atribut A bersama dengan atribut motor lainnya
    - kamu pasang atribut A dan B, sistem akan kasih kamu semua motor yang memilki atribut A dan B bersama dengan atribut motor lainnya juga.
    - kamu pasang atribut A, B, C, D, dan E, sistem akan tetap mencari motor yang memiliki atribut dan menunjukkannya ***bila ditemukan***.
    
    Jadi, kalau kamu menyebutkan:
    - atribut/kriteria motor yang terlalu umum atau sedikit yang digunakan, kemungkinan besar sistem akan memberikan rekomendasi yang terlalu umum juga.
    - atribut/kriteria motor yang terlalu spesifik atau tidak sesuai dengan spesifikasi umum model motor di atas kertas, ada kemungkinan sistem
    tidak bisa memberikan rekomendasi data model motor yang kamu cari.

    Intinya, aplikasi ini memberikan model motor apa adanya sesuai dengan yang kamu sebut di awal.
    """)

    st.markdown("""
    ---
    """)

    st.markdown("""
    Apakah kamu sudah paham konsep cara kerja dari aplikasi 1?
    """)
    

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Saya sudah paham"):
            st.session_state.step = "query_based"
            st.rerun()

    with col2:
        if st.button("Belum paham nih..."):
            st.session_state.step = "intro_query_for_dummies"
            st.rerun()


def step_intro_query_for_dummies():
    st.subheader("🔍 Aplikasi 1: Sistem Rekomendasi Query-Based")

    st.markdown("""
    Jadi gini loh...
    1. Kamu masukan atribut spesifikasi motor yang kamu mau.
    2. Sistem akan cariin motor sesuai atribut spesifikasi motor sesuai yang kamu sebut.

    Inti dari aplikasi 1 sebenarnya hanya sesimple itu, tapi ada hal yang harus kamu tau:
    1. Sistem ini sangat ketat dengan apa yang kamu sebut. Semakin umum maka akan semakin umum atau banyak motor yang muncul, pula sebaliknya.
    2. Kalau sistem tidak bisa menemukan motor sesuai dengan yang kamu sebut, kemungkinannya ada 2:
        - Tidak ada model motor yang sesuai dengan spesifikasi yang kamu sebutkan, atau
        - Spesifikasi motor asli motor tidak sesuai dengan yang kamu sebut, misal kamu sebut 150cc tapi ternyata spesifikasi aslinya 149,7cc atau 151,2cc.
    """)
    
    st.markdown("""
    ---
    """)

    st.markdown("""
    Saya anggap harusnya sudah lebih mudah untuk dimengerti, kita lanjut cobain aplikasinya yuk.
    """)

    if st.button("➡️ Lanjut: cobain aplikasi 1"):
        st.session_state.step = "query_based"
        st.rerun()

def step_query_based():
    st.subheader("🔍 Aplikasi 1: Sistem Rekomendasi Query-Based")

    st.markdown("""
    Masukkan spesifikasi motor yang kamu inginkan, sistem akan mencari motor yang **100% cocok** dengan semua preferensimu.

    🔹 Semakin sedikit atribut yang kamu isi, semakin banyak kemungkinan hasil yang akan ditemukan.  
    🔹 Semakin banyak atribut yang diisi, semakin spesifik model motor yang ditunjukan (kalau ada).  
    ⚠️ Sistem akan mencari **kecocokan yang benar-benar persis**, jadi kalau kamu isi kapasitas mesin 150cc, mungkin tidak ditemukan karena data bisa jadi 149.6 atau 151.2 cc.

    Sistem ini **tidak mentoleransi perbedaan kecil**, jadi cocok digunakan untuk pencarian yang sangat spesifik.
    """)

    st.markdown("---")

    # Label mapping untuk tampilan UI
    label_mapping = {
        "Category": "Kategori",
        "Displacement": "Kapasitas Mesin (cc)",
        "PowerHP": "Tenaga Maksimum (HP)",
        "Brand": "Merek",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (L)",
        "WeightKG": "Berat Motor (kg)",
        "FuelConsumptionKML": "Konsumsi BBM (km/L)",
        "Price": "Harga (Rp)"
    }

    category_label_map = {
        "MaticDaily": "Matic harian",
        "MaticSport": "Matic sport",
        "MaticClassic": "Matic classic",
        "SportNaked": "Sport naked",
        "SportFairing": "Sport fairing",
        "SportAdventure": "Sport adventure", 
        "DualSport/Trail": "Trail / dual sport",
        "Moped": "Bebek",
        "Cruiser": "Cruiser",
        "RetroClassic": "Retro klasik",
        "SportRetro": "Sport retro",
        "SuperSportFairing": "Super sport fairing",
        "SuperSportNaked": "Super sport naked",
        "HyperSportFairing": "Hyper sport fairing",
        "MiniBike": "Motor mini",
        "MiniNaked": "Motor naked mini",
        "Touring": "Touring"
    }

    transmission_label_map = {
        "Automatic": "Otomatis",
        "Manual": "Manual",
        "DCT": "Dual Clutch Transmission"
    }
    
    clutch_label_map = {
        "Wet": "Kopling basah",
        "Dry": "Kopling kering"
    }
    
    engineconfig_label_map = {
        "NearSquare": "Near square (Performa rata diseluruh rentang putaran mesin)",
        "OverBore": "Over bore (Performa di putaran tinggi)",
        "OverStroke": "Over stroke (Performa di putaran rendah)"
    }

    opsi_atribut = list(label_mapping.keys())
    preferensi = {}
    selected_attrs = []

    st.markdown("✅ Checklist atribut yang ingin kamu isi:")

    for attr in opsi_atribut:
        label_id = label_mapping.get(attr, attr)
        if st.checkbox(f"Gunakan {label_id}", key=f"query_use_{attr}"):
            selected_attrs.append(attr)

            if attr in ["Displacement", "PowerHP", "FuelTank", "WeightKG", "FuelConsumptionKML"]:
                preferensi[attr] = st.number_input(f"{label_id}:", step=1, key=f"query_val_{attr}")
            elif attr == "Price":
                preferensi[attr] = st.number_input(f"{label_id}:", min_value=0, step=1_000_000, key=f"query_val_{attr}")
            elif attr in df.columns:
                options = sorted(df[attr].dropna().unique())
            
                if attr == "Category":
                    label_options = [category_label_map.get(o, o) for o in options]
                elif attr == "ClutchType":
                    label_options = [clutch_label_map.get(o, o) for o in options]
                elif attr == "EngineConfig":
                    label_options = [engineconfig_label_map.get(o, o) for o in options]
                elif attr == "Transmission":
                    label_options = [transmission_label_map.get(o, o) for o in options]
                else:
                    label_options = options
            
                pilihan_label = st.selectbox(f"Silakan isi kolom atribut {label_id} di bawah ini.", label_options, key=f"val_{attr}")
                index = label_options.index(pilihan_label)
                val = options[index]
                preferensi[attr] = val 
            else:
                preferensi[attr] = st.text_input(f"{label_id}:", key=f"query_val_{attr}")

    st.markdown("---")

    if st.button("🔎 Cari Motor yang Cocok"):
        hasil = df.copy()
        for attr, val in preferensi.items():
            # Kalau numeric, cocokkan dengan toleransi kecil karena bisa float
            if pd.api.types.is_numeric_dtype(df[attr]):
                hasil = hasil[np.isclose(hasil[attr], float(val), atol=1e-1)]
            else:
                hasil = hasil[hasil[attr] == val]

        if not hasil.empty:
            st.success(f"🎉 Ditemukan {len(hasil)} motor yang cocok dengan preferensimu!")
            for i, row in hasil.iterrows():
                tampilkan_model(row, judul=f"🏍️ {row['Model']}")
        else:
            st.warning("😕 Tidak ada motor yang 100% cocok dengan preferensimu.")

        st.session_state.query_result = hasil.to_dict(orient="records")
        st.session_state.query_input = preferensi
        st.session_state.query_has_run = True  # ✅ Flag bahwa pencarian udah dijalankan

    if st.session_state.get("query_has_run"):
        if st.button("➡️ Lanjut ke Bagian Aplikasi 2"):
            st.session_state.step = "intro_CRSCBR"
            st.rerun()


def step_intro_CRSCBR():
    st.subheader("🤖 Aplikasi 2: Sistem Rekomendasi Conversational Case-Based Reasoning")

    st.markdown("""
    Sistem rekomendasi ini menggunakan pendekatan **Conversational Case-based Reasoning**, 
    dimana user mendapat rekomendasi berdasarkan pengalaman pengguna lain yang memiliki preferensi serupa sebagai solusi rekomendasi utama.
    Namun, bila tidak ditemukan rekam pengalaman yang serupa dengan kriteriamu, sistem akan menyesuaikan agar preferensimu dihitung dan dicocokan
    dengan model motor yang paling mirip. Aplikasi 2 juga menyediakan opsi update atribut/kriteria spesifikasi motor yang kamu cari 
    bila hasil rekomendasi belum memuaskan.

    - Jika tidak ada rekam jejak pengguna lain dengan preferensi yang sama, sistem akan menghitung kemiripan secara otomatis.
    - Bila hasil rekomendasi belum memenuhi preferensi kamu, kamu bisa melakukan *refinement* atau *update* untuk memperbaiki 
    hasil rekomendasi agar lebih sesuai dengan keinginanmu.
    """)

    st.markdown("---")
    
    st.markdown("""
        Gimana? yang ini mudah dipahami kan?
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Oke, paham"):
            st.session_state.step = "input"
            st.rerun()

    with col2:
        if st.button("Masih belum, hehe"):
            st.session_state.step = "intro_CRSCBR_for_dummies"
            st.rerun()


def step_intro_CRSCBR_for_dummies():
    st.subheader("🤖 Aplikasi 2: Sistem Rekomendasi Conversational Case-Based Reasoning")

    st.markdown("""
    Sekali lagi nih aku jelasin:
    1. Sistem bakal cariin kamu motor lewat perpustakaan data pengalaman user lain yang sesuai dengan atribut/kriteria dan 
    urutan prioritas atribut/kriteria yang kamu sebutin.
    2. Fungsi urutan prioritas disini adalah agar hasil rekomendasi motor lebih sesuai dengan ekspektasi prioritas.
        - misal kamu pasang atribut kapasitas mesin 250cc dan kategori motor sport fairing:
            - Kalo kamu pasang **kapasitas mesin** di **prioritas 1**, ada kemungkinan kamu direkomendasikan motor dengan kapasitas yang dekat dari 250cc
    terlebih dahulu tapi tidak selalu sport fairing sebagai diurutan yang paling atas
            - Kalo kamu pasang **kategori motor** di **prioritas 1**, ada kemungkinan kamu direkomendasikan motor dengan kategori motor sport fairing 
    terlebih dahulu tapi tidak selalu memiliki kapasitas 250cc.
            - Hal tersebut mungkin terjadi karena keterbatasan dataset, mohon dimengerti 🙏
    3. Kalo target model motor dengan atribut/kriteria yang kamu sebutin tidak pernah ditemukan oleh user lain, sistem otomatis cariin kamu 
    model motor yang paling mirip dengan cara perhitungan khususnya.

    Ada hal yang harus diketahui, ada satu momen dimana kamu mencari model motor dengan atribut/kriteria spesifikasi namun hasil dari 
    rekomendasinya meleset sedikit atau cukup jauh. Ada beberapa penyebab:
    - Kamu salah memasang urutan prioritas atribut/kriteria yang kamu cari, atau
    - Model motor yang kamu cari memang tidak tersedia di pasaran. 
    """)

    st.markdown("---")
    
    st.markdown("""
        Saya harap lebih mudah untuk dimengerti. Sekarang, kita cobain aplikasi 2
    """)

    if st.button("➡️ Lanjut: Cobain Aplikasi 2"):
        st.session_state.step = "input"
        st.rerun()


def step_input():
    st.subheader("🤖 Aplikasi 2: Sistem Rekomendasi Case-Based")

    st.markdown("""
    Sekarang, coba kamu masukan atribut/kriteria spesifikasi motor yang kamu inginkan di kolom yang tersedia di bawah.
    """)

    st.markdown("---")
    st.subheader("🛠️ Langkah 1: Pilih Atribut dan Isi Preferensi")

    opsi_atribut = [
        "Category", "Displacement", "PowerHP", "Brand", "Transmission",
        "ClutchType", "EngineConfig", "FuelTank", "WeightKG",
        "FuelConsumptionKML", "Price"
    ]

    # Label dalam Bahasa Indonesia
    label_mapping = {
        "Category": "Kategori",
        "Displacement": "Kapasitas Mesin (dalam satuan cc)",
        "PowerHP": "Tenaga Maksimum (*dalam satuan Horsepower*)",
        "Brand": "Merek",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (*dalam satuan Liter*)",
        "WeightKG": "Berat Motor (dalam satuan Kilogram)",
        "FuelConsumptionKML": "Konsumsi BBM (*dalam satuan Kilometer/Liter*)",
        "Price": "Harga (*dalam satuan Rupiah*)"
    }

    numeric_ranges = {
        "Displacement": (50, 1900, 150),
        "PowerHP": (3, 240, 15),
        "FuelTank": (2, 30, 5),
        "WeightKG": (70, 450, 100),
        "FuelConsumptionKML": (10, 100, 40),
        "Price": (10_000_000, 1_450_000_000_000, 25_000_000)
    }

    category_label_map = {
        "MaticDaily": "Matic harian",
        "MaticSport": "Matic sport",
        "MaticClassic": "Matic classic",
        "SportNaked": "Sport naked",
        "SportFairing": "Sport fairing",
        "SportAdventure": "Sport adventure", 
        "DualSport/Trail": "Trail / dual sport",
        "Moped": "Bebek",
        "Cruiser": "Cruiser",
        "RetroClassic": "Retro klasik",
        "SportRetro": "Sport retro",
        "SuperSportFairing": "Super sport fairing",
        "SuperSportNaked": "Super sport naked",
        "HyperSportFairing": "Hyper sport fairing",
        "MiniBike": "Motor mini",
        "MiniNaked": "Motor naked mini",
        "Touring": "Touring"
    }

    transmission_label_map = {
        "Automatic": "Otomatis",
        "DCT": "Dual Clutch Transmission"
    }
    
    clutch_label_map = {
        "Wet": "Kopling basah",
        "Dry": "Kopling kering"
    }
    
    engineconfig_label_map = {
        "NearSquare": "Near square (Performa rata diseluruh rentang putaran mesin)",
        "OverBore": "Over bore (Performa di putaran tinggi)",
        "OverStroke": "Over stroke (Performa di putaran rendah)"
    }

    st.markdown("✅ Checklist atribut yang ingin kamu isi:")

    # Reset setiap kali halaman ini diakses ulang
    st.session_state.selected_attrs = []
    st.session_state.user_input = {}

    for attr in opsi_atribut:
        label = label_mapping.get(attr, attr)
        aktif = st.checkbox(f"Gunakan {label}", key=f"aktif_{attr}")

        if aktif:
            st.session_state.selected_attrs.append(attr)

            # Input nilai berdasarkan tipe
            if attr in numeric_ranges:
                min_val, max_val, default_val = numeric_ranges[attr]
                val = st.number_input(
                    f"{label}:", min_value=min_val, max_value=max_val,
                    value=default_val, step=1, key=f"val_{attr}"
                )

            elif attr == "Price":
                val = st.number_input(
                    f"{label}:", 10_000_000, 1_500_000_000,
                    25_000_000, step=1_000_000, key=f"val_{attr}"
                )
                
            elif attr in df.columns:
                options = sorted(df[attr].dropna().unique())
                
                # Konversi label ke user-friendly jika perlu
                if attr == "Category":
                    label_options = [category_label_map.get(o, o) for o in options]
                elif attr == "ClutchType":
                    label_options = [clutch_label_map.get(o, o) for o in options]
                elif attr == "EngineConfig":
                    label_options = [engineconfig_label_map.get(o, o) for o in options]
                elif attr == "Transmission":
                    label_options = [transmission_label_map.get(o, o) for o in options]
                else:
                    label_options = options  # default, no label mapping
            
                # Tampilkan label tapi simpan value asli
                pilihan_label = st.selectbox(f"Silakan isi kolom atribut {label} di bawah ini.", label_options, key=f"val_{attr}")
                index = label_options.index(pilihan_label)
                val = options[index]

            else:
                val = st.text_input(f"{label}:", key=f"val_{attr}")

            st.session_state.user_input[attr] = val

    if st.button("➡️ Lanjut ke Prioritas") and st.session_state.selected_attrs:
        st.session_state.step = "prioritas"
        st.rerun()

def step_prioritas():
    st.subheader("🎯 Langkah 2: Tentukan Prioritas Atribut")

    st.markdown("""
        Disini, kamu masukin atribut/kriteria yang tadi kamu pilih sesuai dengan kebutuhan prioritas kamu.  
        **Paling atas** itu **paling prioritas** yak.
    """)

    label_mapping = {
        "Brand": "Merek",
        "Category": "Kategori",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "Bore": "Diameter Silinder (mm)",
        "Stroke": "Langkah Piston (mm)",
        "PistonCount": "Jumlah Piston",
        "Displacement": "Kapasitas Mesin (cc)",
        "PowerHP": "Tenaga Maksimum (HP)",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (L)",
        "WeightKG": "Berat Motor (kg)",
        "FuelConsumptionKML": "Konsumsi BBM (km/L)",
        "Price": "Harga (Rp)"
    }

    selected_attrs = st.session_state.selected_attrs
    urutan = len(selected_attrs)
    prioritas = {}
    used = []

    st.markdown("Urutkan atribut yang paling kamu utamakan (1 = paling penting):")

    for i in range(urutan):
        sisa_opsi = [a for a in selected_attrs if a not in used]
        label_opsi = [label_mapping.get(o, o) for o in sisa_opsi]
        pilihan_label = st.selectbox(
            f"Prioritas #{i+1}:", options=[""] + label_opsi, key=f"prioritas_{i}"
        )
        if pilihan_label and pilihan_label != "":
            pilihan = sisa_opsi[label_opsi.index(pilihan_label)]
            prioritas[pilihan] = urutan - i
            used.append(pilihan)

    if len(prioritas) == urutan:
        if st.button("✅ Proses Rekomendasi"):
            st.session_state.prioritas_user = prioritas
            st.session_state.step = "rekomendasi"
            st.rerun()



def step_rekomendasi():
    st.subheader("📈 Langkah 3: Hasil Rekomendasi")

    st.markdown("""
        Disini, kamu akan mendapatkan hasil rekomendasi model motor yang paling mendekati dengan atribut/kriteria yang kamu sebutkan di awal.
        Untuk model motornya, kamu bisa klik/sentuh nama motornya untuk mengetahui spesfikasi motor tersebut.
        Scroll sampai bawah ya, nanti ada umpan balik sederhana yang kamu harus isi.
    """)

    st.subheader("📌 Preferensi kamu:")
    st.json(st.session_state.user_input)

    st.subheader("🎯 Prioritas:")
    st.json(st.session_state.prioritas_user)

     # ⏪ Cek case historis serupa
    populer_dari_case = hitung_model_terpopuler_dari_case_gsheet(
        st.session_state.user_input,
        spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
        sheet_name="case_base"
    )

    if populer_dari_case:
        st.markdown("## 📊 Model yang Sering Dipilih oleh Pengguna Lain")
        for model, jumlah in populer_dari_case:
            st.markdown(f"- **{model}** telah dipilih sebanyak **{jumlah}x** oleh user dengan preferensi yang sama.")

        with st.expander("🛠️ Pilih salah satu model dari data historis?"):
            pilihan = st.selectbox("Pilih model:", [model for model, _ in populer_dari_case])
            if st.button("✅ Gunakan model ini sebagai pilihan akhir"):
                model_final = final_df[final_df["Model"] == pilihan].iloc[0]
                model_final["source"] = "historical_case"
                simpan_case_model_gsheet(
                    user_input=st.session_state.user_input,
                    row_model=model_final,
                    spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
                    sheet_name="case_base",
                    refined=False,
                    refine_log=[],
                    user_ranked=False
                )
                st.session_state.final_chosen_model = model_final
                st.success(f"✅ Model '{pilihan}' disimpan sebagai pilihan akhir.")
                st.session_state.step = "survey_1"
                st.rerun()
    else:
        st.info("📁 Belum ada rekam jejak pengguna lain dengan preferensi ini.")

    st.markdown("---")  # Pisahkan dari hasil perhitungan baru

    user_input = st.session_state.user_input
    prioritas = st.session_state.prioritas_user

    user_vec, weight_vec = buat_user_vector_weighted(user_input, prioritas, case_vector_df, df)
    hasil = rekomendasi_cosine_weighted(user_vec, weight_vec, case_matrix, final_df, user_input, top_n=6)



    st.subheader("🚀 Top-1 Model Paling Mendekati Preferensimu:")
    top1 = hasil.iloc[0]
    tampilkan_model(top1)

    st.markdown("---")

    st.subheader("🔍 5 Model Alternatif Lainnya yang Masih Mirip:")
    for i in range(1, min(6, len(hasil))):
        row = hasil.iloc[i]
        tampilkan_model(row, judul=f"🏍️ Model Alternatif {i}: **{(row.get('Model', f'Model {i+1}')).upper()}**")
        st.markdown("----")



    st.subheader("📝 Feedback Rekomendasi Awal")
    st.markdown("##### Apakah Anda puas dengan hasil Top-1 dari rekomendasi ini?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Ya, Saya Puas"):
            top1_model = hasil.iloc[0].copy()
            top1_model["source"] = "cosine_similarity"
            # simpan_case_model(st.session_state.user_input, top1_model, refined=False, user_ranked=False)
            simpan_case_model_gsheet(
                user_input=st.session_state.user_input,
                row_model=top1_model,
                spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
                sheet_name="case_base",
                refined=False,
                refine_log=[],
                user_ranked=False
            )
            st.session_state.final_chosen_model = top1_model
            st.success("✅ Terima kasih! Rekomendasi telah disimpan.")
            st.session_state.step = "survey_1"
            st.rerun()

    with col2:
        if st.button("❌ Tidak Puas"):
            st.session_state.puas_awal = "tidak"

    if st.session_state.get("puas_awal") == "tidak":
        st.markdown("""
        kamu masih bisa perbaiki hasilnya lho kalo semisal belum memenuhi ekspektasimu 👀, tinggal pilih:
        1. Pilih opsi "Tidak ada"
        2. Klik/sentuh tombol "Mau di-update agar lebih sesuai?"
        """)

        st.markdown("---")
        
        opsi_model = ["Tidak ada"] + list(hasil.iloc[1:6]["Model"]) + ["Saya ingin keluar saja"]
        cocok_lain = st.radio("🎯 Adakah model lain yang mendekati preferensimu?", opsi_model, key="radio_cocok_lain")


        if cocok_lain in hasil.iloc[1:6]["Model"].values:
            if st.button("Aku mau simpan model ini!"):
                idx = hasil[hasil["Model"] == cocok_lain].index[0]
                model_lain = hasil.loc[idx].copy()
                model_lain["source"] = "cosine_similarity"
                # simpan_case_model(st.session_state.user_input, hasil.loc[idx], refined=False, user_ranked=True)
                simpan_case_model_gsheet(
                    user_input=st.session_state.user_input,
                    row_model=model_lain,
                    spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
                    sheet_name="case_base",
                    refined=False,
                    refine_log=[],
                    user_ranked=True
                )
                st.session_state.final_chosen_model = model_lain
                st.success(f"✅ Model '{cocok_lain}' disimpan sebagai pilihan Anda.")
                st.session_state.step = "survey_1"
                st.rerun()

        elif cocok_lain == "Tidak ada":
            if st.button("🔧 Mau di-update agar lebih sesuai?"):
                st.session_state.step = "refinement"
                st.session_state.refine_base_model = hasil.iloc[0].to_dict()
                st.session_state.refine_steps = []
                st.rerun()

        elif cocok_lain == "Saya ingin keluar saja":
            st.warning("🚪 Serius nih? kamu masih bisa refine loh...")
            if st.button("Bodo amat, saya mau keluar!"):
                st.session_state.refine_base_model = hasil.iloc[0].to_dict()
                st.session_state.step = "survey_1"
                st.rerun()


def step_refinement():
    st.subheader("🔧 Langkah 4: Refinement Interaktif")

    iterasi = len(st.session_state.get("refine_steps", [])) + 1
    st.markdown(f"##### 🔁 Refinement Iterasi ke-{iterasi}")

    if "refine_base_model" not in st.session_state:
        st.error("Tidak ada referensi motor untuk refinement.")
        return

    model_awal = st.session_state.refine_base_model
    user_input = st.session_state.user_input.copy()

    st.markdown("##### 📌 Model Referensi (Top-1 Terakhir):")
    tampilkan_model(
        model_awal,
        judul=f"**{(model_awal.get('Model', 'Tidak Diketahui')).upper()}**"
    )

    st.markdown("##### ✍️ Ubah Preferensi yang Ingin Diperbaiki:")

    opsi_atribut = [
        "Category", "Displacement", "PowerHP", "Brand", "Transmission",
        "ClutchType", "EngineConfig", "FuelTank", "WeightKG",
        "FuelConsumptionKML", "Price"
    ]

    label_mapping = {
        "Category": "Kategori",
        "Displacement": "Kapasitas Mesin (cc)",
        "PowerHP": "Tenaga Maksimum (HP)",
        "Brand": "Merek",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (L)",
        "WeightKG": "Berat Motor (kg)",
        "FuelConsumptionKML": "Konsumsi BBM (km/L)",
        "Price": "Harga (Rp)"
    }

    numeric_ranges = {
        "Displacement": (50, 1900, 150),
        "PowerHP": (3, 240, 15),
        "FuelTank": (2, 30, 5),
        "WeightKG": (70, 450, 100),
        "FuelConsumptionKML": (10, 100, 40),
        "Price": (10_000_000, 1_450_000_000, 25_000_000)
    }

    category_label_map = {
        "MaticDaily": "Matic harian",
        "MaticSport": "Matic sport",
        "MaticClassic": "Matic classic",
        "SportNaked": "Sport naked",
        "SportFairing": "Sport fairing",
        "SportAdventure": "Sport adventure", 
        "DualSport/Trail": "Trail / dual sport",
        "Moped": "Bebek",
        "Cruiser": "Cruiser",
        "RetroClassic": "Retro klasik",
        "SportRetro": "Sport retro",
        "SuperSportFairing": "Super sport fairing",
        "SuperSportNaked": "Super sport naked",
        "HyperSportFairing": "Hyper sport fairing",
        "MiniBike": "Motor mini",
        "MiniNaked": "Motor naked mini",
        "Touring": "Touring"
    }

    transmission_label_map = {
        "Automatic": "Otomatis",
        "DCT": "Dual Clutch Transmission"
    }
    
    clutch_label_map = {
        "Wet": "Kopling basah",
        "Dry": "Kopling kering"
    }
    
    engineconfig_label_map = {
        "NearSquare": "Near square (Performa rata diseluruh rentang putaran mesin)",
        "OverBore": "Over bore (Performa di putaran tinggi)",
        "OverStroke": "Over stroke (Performa di putaran rendah)"
    }

    st.markdown("###### *Checklist atribut yang ingin kamu ubah atau tambahkan")

    refine_selected_attrs = []
    perubahan = {}

    for attr in opsi_atribut:
        val_lama = user_input.get(attr, None)
        label_id = label_mapping.get(attr, attr)
        label = f"{label_id} (saat ini: `{val_lama}`)" if val_lama else f"{label_id} (tidak diubah)"
        aktif = st.checkbox(label, key=f"aktif_refine_{attr}")

        if aktif:
            refine_selected_attrs.append(attr)

            if attr in df.columns and df[attr].dtype == object:
                opsi = sorted(df[attr].dropna().unique())
            
                # Konversi label untuk tampil ke user
                if attr == "Category":
                    label_opsi = [category_label_map.get(o, o) for o in opsi]
                elif attr == "ClutchType":
                    label_opsi = [clutch_label_map.get(o, o) for o in opsi]
                elif attr == "Transmission":
                    label_opsi = [transmission_label_map.get(o, o) for o in opsi]
                elif attr == "EngineConfig":
                    label_opsi = [engineconfig_label_map.get(o, o) for o in opsi]
                else:
                    label_opsi = opsi
            
                # Default index untuk pilihan
                if val_lama in opsi:
                    index_default = opsi.index(val_lama)
                else:
                    index_default = 0
            
                pilihan_label = st.selectbox(
                    f"Update {label_id}:", label_opsi, index=index_default,
                    key=f"refine_val_{attr}"
                )
                val_baru = opsi[label_opsi.index(pilihan_label)]

            # Numerik
            elif attr in numeric_ranges:
                min_val, max_val, default_val = numeric_ranges[attr]
                val_baru = st.number_input(
                    f"Update {label_id}:", min_value=min_val, max_value=max_val,
                    value=int(val_lama) if val_lama else default_val,
                    step=1, key=f"refine_val_{attr}"
                )

            # Fallback
            else:
                val_baru = st.text_input(f"Update {label_id}:", val_lama or "", key=f"refine_val_{attr}")

            if val_baru != val_lama:
                perubahan[attr] = (val_lama, val_baru)
                user_input[attr] = val_baru

    if st.button("✅ Simpan & Hitung Ulang"):
        if perubahan:
            st.session_state.refine_steps.append(perubahan)
            st.session_state.user_input = user_input
            st.session_state.active_attrs_after_refine = sorted(set(user_input.keys()))
            st.session_state.step = "refine_prioritas"
            st.rerun()
        else:
            st.warning("⚠️ Tidak ada perubahan yang dilakukan.")

    if st.button("❌ Cancel dan keluar dari app"):
        st.success("Sesi refinement selesai. Menyimpan hasil final.")
        st.session_state.step = "survey_1"
        st.rerun()

def step_refine_prioritas():
    st.subheader("🎯 Langkah 5: Mengurutkan Prioritas Atribut Refinement")

    label_mapping = {
        "Brand": "Merek",
        "Category": "Kategori",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "Bore": "Diameter Silinder (mm)",
        "Stroke": "Langkah Piston (mm)",
        "PistonCount": "Jumlah Piston",
        "Displacement": "Kapasitas Mesin (cc)",
        "PowerHP": "Tenaga Maksimum (HP)",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (L)",
        "WeightKG": "Berat Motor (kg)",
        "FuelConsumptionKML": "Konsumsi BBM (km/L)",
        "Price": "Harga (Rp)"
    }

    selected_attrs = st.session_state.get("active_attrs_after_refine", [])
    urutan = len(selected_attrs)
    prioritas = {}
    used = []

    st.markdown("Urutkan kembali atribut yang kamu anggap paling penting:")

    for i in range(urutan):
        sisa_opsi = [a for a in selected_attrs if a not in used]
        label_opsi = [label_mapping.get(o, o) for o in sisa_opsi]
        pilihan_label = st.selectbox(
            f"Prioritas #{i+1}:", options=[""] + label_opsi, key=f"refine_prioritas_final_{i}"
        )
        if pilihan_label and pilihan_label != "":
            pilihan = sisa_opsi[label_opsi.index(pilihan_label)]
            prioritas[pilihan] = urutan - i
            used.append(pilihan)

    if len(prioritas) == urutan:
        if st.button("🚀 Hitung Ulang Rekomendasi"):
            st.session_state.prioritas_user = prioritas

            user_input = st.session_state.user_input
            user_vec, weight_vec = buat_user_vector_weighted(user_input, prioritas, case_vector_df, df)
            hasil_refined = rekomendasi_cosine_weighted(user_vec, weight_vec, case_matrix, final_df, user_input, top_n=6)

            st.session_state.refine_base_model = hasil_refined.iloc[0].to_dict()
            st.session_state.last_refined_result = hasil_refined
            st.success("✅ Rekomendasi diperbarui!")
            st.session_state.step = "refinement_result"
            st.rerun()

def step_refinement_result():
    
    st.subheader("✅ Langkah 6: Hasil Rekomendasi Setelah Refinement")

    iterasi = len(st.session_state.get("refine_steps", []))
    st.markdown(f"##### 📊 Total Iterasi Refinement: {iterasi}")

    st.subheader("📌 Preferensi kamu:")
    st.json(st.session_state.user_input)

    st.subheader("🎯 Prioritas:")
    st.json(st.session_state.prioritas_user)

    hasil = st.session_state.get("last_refined_result", None)

    if hasil is not None:
        st.markdown("#### 🚀 Top-1 Refined Model Paling Mendekati Preferensi baru kamu:")
        top1_refined = hasil.iloc[0]
        tampilkan_model(top1_refined)

        st.markdown("---")

        st.markdown("#### 🔍 5 Refined Model Alternatif Lainnya yang Masih Mirip:")
        for i in range(1, min(6, len(hasil))):
            row_refined = hasil.iloc[i]
            tampilkan_model(row_refined, judul=f"🏍️ Model Alternatif {i}: **{(row_refined.get('Model', f'Model {i+1}')).upper()}**")
            st.markdown("----")
    else:
        st.warning("Belum ada hasil terbaru dari refinement.")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Saya puas dengan refined model motor Top-1"):
            top1_refinedmodel = hasil.iloc[0].copy()
            top1_refinedmodel["source"] = "cosine_similarity"
            simpan_case_model_gsheet(
                user_input=st.session_state.user_input,
                row_model=top1_refinedmodel,
                spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
                sheet_name="case_base",
                refined=True,
                refine_log=st.session_state.get("refine_steps", []),
                user_ranked=False
            )
            st.session_state.final_chosen_model = top1_refinedmodel
            st.success("✅ Terima kasih! Rekomendasi telah disimpan.")
            st.session_state.step = "survey_1"
            st.rerun()

    with col2:
        if st.button("❎ Hasil belum puas"):
            st.session_state.show_refine_options = True
            st.rerun()

    if st.session_state.get("show_refine_options", False):
        opsi_model_refine = list(hasil.iloc[1:6]["Model"])
        opsi_pilihan = ["Tidak ada"] + opsi_model_refine + ["Saya ingin keluar saja"]
        st.markdown("#### 🎯 Apakah ada model lain dari hasil refinement yang lebih cocok?")
        pilih_lain = st.radio("Pilihan kandidat motor:", opsi_pilihan, key="radio_refine_pilihan")

        if pilih_lain in opsi_model_refine:
            if st.button("✅ Simpan model ini"):
                idx = hasil[hasil["Model"] == pilih_lain].index[0]
                simpan_case_model_gsheet(
                    user_input=st.session_state.user_input,
                    row_model=hasil.loc[idx].copy(),
                    spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM',
                    sheet_name="case_base",
                    refined=True,
                    refine_log=st.session_state.get("refine_steps", []),
                    user_ranked=True
                )
                st.session_state.final_chosen_model = hasil.iloc[0]  # atau hasil.iloc[0]
                st.success(f"✅ Model '{pilih_lain}' disimpan sebagai pilihan Anda.")
                st.session_state.step = "survey_1"
                st.rerun()

        elif pilih_lain == "Tidak ada":
            if st.button("🔁 Refine Lagi"):
                st.session_state.refine_iteration_count = len(st.session_state.get("refine_steps", [])) + 1
                st.session_state.refine_base_model = hasil.iloc[0].to_dict()
                st.session_state.step = "refinement"
                st.session_state.pop("show_refine_options", None)
                st.rerun()

        elif pilih_lain == "Saya ingin keluar saja":
            st.warning("🚪 Proses dihentikan akan dihentikan dan rekomendasi tidak disimpan ke kamus. Yakin?")
            if st.button("Keluar & Akhiri"):
                st.session_state.step = "survey_1"
                st.rerun()

# def step_survey_1():
#     st.title("📝 Survei Pengalaman - Aplikasi 2 (Case-Based Recommender)")

#     st.markdown("""
#     Terima kasih telah mencoba aplikasi kedua. Kami ingin tahu pendapat kamu terhadap sistem rekomendasi ini.
#     """)

#     rating_akurasi = st.radio("🎯 Menurut kamu, seberapa akurat sistem ini dalam merekomendasikan motor yang sesuai?", [
#         "Sangat akurat", "Cukup akurat", "Kurang akurat", "Tidak akurat"
#     ], key="cb_akurasi")

#     rating_puasan = st.radio("😊 Apakah kamu puas dengan hasil rekomendasi yang diberikan oleh sistem ini?", [
#         "Sangat puas", "Puas", "Kurang puas", "Tidak puas"
#     ], key="cb_puasan")

#     rating_pengalaman = st.radio("🧭 Bagaimana pengalaman kamu saat menggunakan aplikasi ini?", [
#         "Sangat nyaman", "Cukup nyaman", "Sedikit membingungkan", "Tidak nyaman"
#     ], key="cb_pengalaman")

#     saran = st.text_area("💬 Ada saran, kritik, atau komentar lainnya?", placeholder="Tulis pendapat kamu di sini...")

#     if st.button("➡️ Lanjut ke Survei Perbandingan"):
#         st.session_state.survey_1_feedback = {
#             "akurasi": rating_akurasi,
#             "puasan": rating_puasan,
#             "pengalaman": rating_pengalaman,
#             "saran": saran
#         }
#         st.session_state.step = "survey_2"
#         st.rerun()

def step_survey_1():
    st.subheader("📝 Survei Pengalaman dengan Aplikasi 2 (Case-Based)")
    st.markdown("Berikan penilaianmu terhadap sistem rekomendasi Case-Based berdasarkan pengalaman kamu.")
    st.markdown("Skala penilaian:")
    st.markdown("- 1 = Sangat Tidak Setuju")
    st.markdown("- 5 = Sangat Setuju")

    skala = [1, 2, 3, 4, 5]
    survey_1_feedback = {}

    st.markdown("### 🎯 Kualitas Rekomendasi (PRQ)")
    survey_1_feedback["prq_like_product"] = st.selectbox("Saya menyukai model motor yang saya pilih", skala)
    survey_1_feedback["prq_dislike_interaction"] = st.selectbox("Saya tidak suka interaksi dengan sistem ini", skala)

    st.markdown("### ⚡ Efisiensi (PE)")
    survey_1_feedback["pe_fast_find"] = st.selectbox("Saya bisa menemukan motor yang saya suka dengan cepat", skala)

    st.markdown("### 🤝 Kepercayaan (TR)")
    survey_1_feedback["tr_would_buy"] = st.selectbox("Saya akan membeli motor ini suatu hari nanti", skala)
    survey_1_feedback["tr_use_again"] = st.selectbox("Saya ingin menggunakan sistem ini lagi di masa depan", skala)

    st.markdown("### 📖 Informasi (INF)")
    survey_1_feedback["inf_easy_info"] = st.selectbox("Saya mudah mendapatkan informasi tentang motor", skala)

    st.markdown("### 🧭 Kemudahan Penggunaan (ETU)")
    survey_1_feedback["etu_difficult_find"] = st.selectbox("Saya merasa sulit menemukan motor yang sesuai keinginan", skala)
    survey_1_feedback["etu_no_difficulty"] = st.selectbox("Saya tidak mengalami kesulitan saat menggunakan sistem", skala)

    st.markdown("### 🧠 Kemudahan Memahami (EOU)")
    survey_1_feedback["eou_easy_options"] = st.selectbox("Pertanyaan dan pilihan mudah dipahami", skala)
    survey_1_feedback["eou_understood_all"] = st.selectbox("Saya paham semua yang ditampilkan dalam sistem", skala)

    st.markdown("---")
    if st.button("➡️ Lanjut ke Survei Perbandingan"):
        st.session_state.survey_1_feedback = survey_1_feedback
        st.session_state.step = "survey_2"
        st.rerun()

# def step_survey_2():
#     st.title("⚖️ Survei Perbandingan Dua Sistem")

#     st.markdown("""
#     Sekarang setelah kamu mencoba **dua jenis sistem rekomendasi**, kami ingin tahu pendapat akhirmu dalam membandingkan keduanya.
#     """)

#     sistem_terfavorit = st.radio("💡 Dari dua sistem yang kamu coba, mana yang lebih kamu sukai?", [
#         "Aplikasi 1 - Query-Based", 
#         "Aplikasi 2 - Case-Based"
#     ], key="survey2_favorit")

#     alasan = st.text_area("🧠 Jelaskan kenapa kamu memilih sistem tersebut:", placeholder="Tulis alasannya di sini...")

#     efektifitas = st.radio("📈 Menurutmu, sistem mana yang lebih efektif dalam membantumu menemukan motor yang kamu cari?", [
#         "Aplikasi 1 - Query-Based", 
#         "Aplikasi 2 - Case-Based", 
#         "Sama-sama efektif", 
#         "Keduanya kurang efektif"
#     ], key="survey2_efektivitas")

#     if st.button("✅ Selesai & Simpan Jawaban"):
#         st.session_state.survey_2_feedback = {
#             "favorit": sistem_terfavorit,
#             "alasan": alasan,
#             "efektivitas": efektifitas
#         }
#         st.session_state.step = "finish"
#         st.rerun()

def step_survey_2():
    st.subheader("⚖️ Survei Perbandingan Dua Sistem")

    st.markdown("Bandingkan pengalaman kamu antara Aplikasi 1 (Query-Based) dan Aplikasi 2 (Case-Based).")

    survey_2_feedback = {}

    st.markdown("### 🌟 Kenyamanan Penggunaan")
    survey_2_feedback["prefer_interface"] = st.radio(
        "Saya lebih nyaman menggunakan:",
        ["Aplikasi 1 (Query-Based)", "Aplikasi 2 (Case-Based)"],
        key="prefer_interface"
    )

    st.markdown("### 🎯 Akurasi Rekomendasi")
    survey_2_feedback["prefer_accuracy"] = st.radio(
        "Menurut saya, hasil rekomendasinya lebih akurat di:",
        ["Aplikasi 1 (Query-Based)", "Aplikasi 2 (Case-Based)"],
        key="prefer_accuracy"
    )

    st.markdown("### 🔁 Niat Menggunakan Kembali")
    survey_2_feedback["reuse"] = st.radio(
        "Jika di masa depan saya ingin mencari motor lagi, saya akan menggunakan kembali:",
        ["Aplikasi 1 (Query-Based)", "Aplikasi 2 (Case-Based)"],
        key="reuse"
    )

    st.markdown("### ✍️ Kritik / Saran (Opsional)")
    survey_2_feedback["free_feedback"] = st.text_area("Silakan isi jika ada kritik atau saran", "")

    if st.button("✅ Selesai dan Tampilkan Ringkasan Evaluasi"):
        st.session_state.survey_2_feedback = survey_2_feedback
        st.session_state.step = "finish"
        st.rerun()


def step_finish_evaluation():
    st.title("🎉 Evaluasi Selesai")

    st.markdown("""
    Terima kasih telah mengikuti sesi uji coba sistem rekomendasi kami.  
    Berikut ini adalah rangkuman jawaban dan preferensi kamu:
    """)

    # Identitas
    st.subheader("👤 Identitas Pengguna")
    st.json(st.session_state.get("user_identity", {}))

    # Input & hasil query-based
    st.subheader("🔍 Aplikasi 1 - Query-Based")
    
    st.markdown("**Preferensi yang dimasukkan:**")
    st.json(st.session_state.get("query_input", {}))
    
    st.markdown("**Hasil Rekomendasi:**")
    query_result = st.session_state.get("query_result", [])
    if query_result:
        for i, row in enumerate(query_result[:3]):
            tampilkan_model(row, judul=f"📌 Hasil {i+1}: " + row["Model"])
    else:
        st.info("Tidak ada hasil yang cocok.")



    # Input & hasil case-based
    st.subheader("🤖 Aplikasi 2 - Case-Based")

    st.markdown("**Preferensi terakhir yang dimasukkan:**")
    st.json(st.session_state.get("user_input", {}))

    st.markdown("**Prioritas Atribut terakhir:**")
    st.json(st.session_state.get("prioritas_user", {}))

    st.markdown("**Hasil Rekomendasi terakhir:**")
    if "final_chosen_model" in st.session_state: # historical/cosine top1/cosine top2-6
        st.success("Model yang dipilih olehmu sebagai rekomendasi akhir:")
        tampilkan_model(st.session_state.final_chosen_model)
        st.session_state.final_CRSCBR_answer = st.session_state.final_chosen_model.copy().to_dict()

    elif "refine_base_model" in st.session_state: # keluar dari app
        st.info("Model rekomendasi terakhir dari sistem:") 
        tampilkan_model(st.session_state.refine_base_model)
        st.session_state.final_CRSCBR_answer = st.session_state.refine_base_model.copy()

    elif "hasil" in st.session_state: # where the fvck is this came from?
        st.warning("Model rekomendasi awal:")
        tampilkan_model(st.session_state.hasil.iloc[0])
        st.session_state.final_CRSCBR_answer = st.session_state.hasil.iloc[0].to_dict()

    else:
        st.warning("Belum ada rekomendasi yang berhasil ditentukan.")
        st.session_state.final_CRSCBR_answer = {"status": "no result"}

    # Log refinement (jika ada)
    if "refine_steps" in st.session_state and st.session_state["refine_steps"]:
        st.subheader("📝 Log Refinement:")
        with st.expander("Lihat log refinement yang telah dilakukan:"):
            for i, step in enumerate(st.session_state["refine_steps"], 1):
                st.markdown(f"**Iterasi {i}:**")
                st.write(step)


    # Survei 1
    st.subheader("📝 Feedback untuk Aplikasi Case-Based")
    with st.expander("Lihat jawaban survei 1:"):
        st.json(st.session_state.get("survey_1_feedback", {}))

    # Survei 2
    st.subheader("⚖️ Perbandingan Dua Sistem")
    with st.expander("Lihat jawaban survei 2:"):
        st.json(st.session_state.get("survey_2_feedback", {}))

    st.session_state.user_has_saved = False
    
    # Simpan hasil akhir
    if st.button("💾 Simpan hasil jawabanmu"):
        final_data = {
            "identity": st.session_state.get("user_identity"),
            "query_input": st.session_state.get("query_input"),
            "query_result": st.session_state.get("query_result"),
            "user_input": st.session_state.get("user_input"),
            "prioritas_user": st.session_state.get("prioritas_user"),
            "final_CRSCBR_answer": st.session_state.get("final_CRSCBR_answer"),
            "refine_steps": st.session_state.get("refine_steps", []),
            "survey_1_feedback": st.session_state.get("survey_1_feedback"),
            "survey_2_feedback": st.session_state.get("survey_2_feedback"),
            "timestamp": timestamp_WIB(),
            "case_id": generate_case_id()
        }

        data_untuk_gsheet = format_data_for_gsheet(final_data)
        success, msg = kirim_data_ke_gsheet(data_untuk_gsheet, spreadsheet_id='193gZBpZUWYv1GJxvgibbf04uR_txgJPiFoEIGTuVPSM', sheet_name="hasil_user_testing")

        # filepath = simpan_ke_file_json_agregat(final_data)
        st.success("✅ Hasil berhasil disimpan!")
        st.session_state.user_has_saved = True

    # Tombol reset setelah simpan
    if st.session_state.get("user_has_saved") == True:
        if st.button("🔄 Balik lagi ke page paling awal?"):
            # Simpan step target sebelum menghapus session
            st.session_state["__next_step"] = "intro"
    
            # Hapus semua session kecuali __next_step
            keys_to_keep = ["__next_step"]
            for key in list(st.session_state.keys()):
                if key not in keys_to_keep:
                    del st.session_state[key]
    
            st.rerun()




# =================== FUNGSI ASLI ===================


# ==========================
# Buat user_vector dan weight_vector berdasarkan preferensi user
# ==========================
def buat_user_vector_weighted(user_input, prioritas_user, final_df, df_mentah):
    """
    Buat user_vector dan weight_vector berdasarkan preferensi user,
    dengan bobot eksplisit dari prioritas_user.
    """
    user_vector = np.zeros(len(final_df.columns), dtype=float)
    weight_vector = np.zeros(len(final_df.columns), dtype=float)
    col_index = {col: i for i, col in enumerate(final_df.columns)}

    for attr, val in user_input.items():
        weight = prioritas_user.get(attr, 1.0)  # default ke 1.0 kalau tidak ditemukan

        # One-hot encoding
        if attr in ["Brand", "Category", "Transmission", "ClutchType", "EngineConfig"]:
            prefix = attr.replace(" ", "")
            col_name = f"{prefix}_{val}".lower()
            match = [col for col in final_df.columns if col.lower() == col_name]
            if match:
                idx = col_index[match[0]]
                user_vector[idx] = 1.0
                weight_vector[idx] = weight

        # Numerikal
        else:
            norm_col = f"{attr}_normalized"
            if norm_col in final_df.columns and attr in df_mentah.columns:
                idx = col_index[norm_col]
                max_val = df_mentah[attr].max()
                norm_val = float(val) / max_val
                user_vector[idx] = norm_val
                weight_vector[idx] = weight

    return user_vector, weight_vector


# ==========================
# Rekomendasi Cosine Similarity Berbobot
# ==========================
def rekomendasi_cosine_weighted(user_vec, weight_vec, case_matrix, final_df, user_input, top_n=6):
    """
    Menghitung cosine similarity berbobot + penalti selisih nilai numerik dari preferensi user.
    """
    # Konversi preferensi user ke dict
    user_pref = dict(user_input)

    # Siapkan target numerik dari preferensi user (jika ada)
    target_power = float(user_pref.get("PowerHP", 0))
    target_cc = float(user_pref.get("Displacement", 0))
    target_price = float(user_pref.get("Price", 0))
    target_weight = float(user_pref.get("WeightKG", 0))
    target_fuel = float(user_pref.get("FuelTank", 0))

    # Hitung cosine similarity (berbobot)
    weighted_user_vector = user_vec * weight_vec
    weighted_case_matrix = case_matrix * weight_vec
    similarity_scores = cosine_similarity([weighted_user_vector], weighted_case_matrix)[0]

    # Tempel ke final_df
    final_df_with_score = final_df.copy()
    final_df_with_score["Similarity"] = similarity_scores

    # Hitung penalti
    final_df_with_score["PowerPenalty"] = abs(final_df_with_score["PowerHP"] - target_power) if target_power else 0
    final_df_with_score["CCPenalty"] = abs(final_df_with_score["Displacement"] - target_cc) if target_cc else 0
    final_df_with_score["PricePenalty"] = abs(final_df_with_score["Price"] - target_price) / 1_000_000 if target_price else 0
    final_df_with_score["WeightPenalty"] = abs(final_df_with_score["WeightKG"] - target_weight) if target_weight else 0
    final_df_with_score["FuelPenalty"] = abs(final_df_with_score["FuelTank"] - target_fuel) if target_fuel else 0

    # Final score: cosine - penalti (atur skala sesuai preferensi) <<<<<------ buat atur skala prioritas numerikal
    final_df_with_score["FinalScore"] = (
        final_df_with_score["Similarity"]
        - 0.04 * final_df_with_score["PowerPenalty"]
        - 0.02 * final_df_with_score["CCPenalty"]
        - 0.01 * final_df_with_score["PricePenalty"]
        - 0.01 * final_df_with_score["WeightPenalty"]
        - 0.01 * final_df_with_score["FuelPenalty"]
    )

    # Urutkan dan ambil top-N
    sorted_df = final_df_with_score.sort_values(by="FinalScore", ascending=False)
    return sorted_df.head(top_n)


# ==========================
# Timing dan ID Case
# ==========================
def timestamp_WIB():
    return datetime.now(pytz.timezone('Asia/Jakarta')).strftime("%Y-%m-%d %H:%M:%S")

def generate_case_id():
    return f"case_{datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

# ==========================
# Format data user untuk Google Sheets
# ==========================
def format_data_for_gsheet(data_dict):
    formatted = {}
    for k, v in data_dict.items():
        if v is None:
            formatted[k] = "N/A"
        elif isinstance(v, (dict, list)):
            try:
                formatted[k] = json.dumps(v, ensure_ascii=False)
            except:
                formatted[k] = str(v)
        elif isinstance(v, (pd.Series, pd.DataFrame)):
            formatted[k] = str(v.to_dict())
        else:
            formatted[k] = str(v)
    return formatted


def kirim_data_ke_gsheet(data_dict, spreadsheet_id, sheet_name="hasil_user_testing"):
    try:
        json_key = dict(st.secrets["gcp_service_account"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
            json.dump(json_key, tmp)
            tmp_path = tmp.name

        gc = pygsheets.authorize(service_file=tmp_path)
        sh = gc.open_by_key(spreadsheet_id)
        wks = sh.worksheet_by_title(sheet_name)

        # FORMAT DULU
        formatted_data = format_data_for_gsheet(data_dict)

        wks.append_table(list(formatted_data.values()), dimension='ROWS')
        return True, "✅ Data berhasil dikirim ke Google Sheets."
    except Exception as e:
        return False, f"❌ Gagal mengirim data ke Google Sheets: {e}"



# ==========================
# Load case base dari Google Sheets (digunakan oleh hitung_model_terpopuler_dari_case_gsheet)
# ==========================
def load_case_base_from_gsheet(spreadsheet_id, sheet_name="CaseBase"):
    json_key = dict(st.secrets["gcp_service_account"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(json_key, tmp)
        tmp_path = tmp.name
    
    gc = pygsheets.authorize(service_file=tmp_path)
    sh = gc.open_by_key(spreadsheet_id)
    wks = sh.worksheet_by_title(sheet_name)
    records = wks.get_all_records()

    # Parse ulang field JSON
    for r in records:
        r["user_input"] = json.loads(r["user_input"])
        r["refine_steps"] = json.loads(r["refine_steps"])
        r["chosen_models"] = json.loads(r["chosen_models"])
        r["is_refined"] = r["is_refined"].lower() == "true"
        r["user_ranked"] = r["user_ranked"].lower() == "true"
        r["refine_iteration_count"] = int(r["refine_iteration_count"])
    return records


# ==========================
# Hitung model terpopuler dari case base GSheets (showed at step_rekomendasi)
# ==========================
def hitung_model_terpopuler_dari_case_gsheet(user_input, spreadsheet_id, sheet_name="CaseBase"):
    records = load_case_base_from_gsheet(spreadsheet_id, sheet_name)
    
    # Normalisasi preferensi user
    user_set = set((k.lower(), str(v).lower()) for k, v in user_input.items())

    hitung_model = {}
    for case in records:
        case_input = case.get("user_input", {})
        
        # ⛔ Skip kalau user_input-nya bukan dict
        if not isinstance(case_input, dict):
            continue

        case_set = set((k.lower(), str(v).lower()) for k, v in case_input.items())

        if case_set == user_set:
            for model_info in case.get("chosen_models", []):
                model = model_info.get("model")
                if model:
                    hitung_model[model] = hitung_model.get(model, 0) + 1

    hasil = sorted(hitung_model.items(), key=lambda x: x[1], reverse=True)
    return hasil  # list of (model, jumlah)


# ==========================
# Simpan case model ke Google Sheets
# ==========================
def simpan_case_model_gsheet(user_input, row_model, spreadsheet_id, sheet_name="case_base", refined=False, refine_log=None, user_ranked=False):

    # Siapkan data case
    case_data = {
        "case_id": generate_case_id(),
        "user_input": json.dumps(user_input, ensure_ascii=False),
        "is_refined": refined,
        "refine_steps": json.dumps(refine_log if refine_log else [], ensure_ascii=False),
        "refine_iteration_count": len(refine_log) if refine_log else 0,
        "chosen_models": json.dumps([{
            "model": row_model["Model"],
            "similarity_score": float(row_model["Similarity"]) if "Similarity" in row_model else None,
            "source": row_model.get("source", "cosine_similarity")
        }], ensure_ascii=False),
        "user_ranked": user_ranked,
        "timestamp": datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    }

    # Connect ke GSheet
    json_key = dict(st.secrets["gcp_service_account"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp:
        json.dump(json_key, tmp)
        tmp_path = tmp.name
    
    gc = pygsheets.authorize(service_file=tmp_path)
    sh = gc.open_by_key(spreadsheet_id)
    wks = sh.worksheet_by_title(sheet_name)

    # Append baris baru
    wks.append_table(list(case_data.values()), dimension='ROWS', overwrite=False)


# ==========================
# Logging Refinement Steps
# ==========================
def write_log_refine_iteration(refine_queue, user_input_before):
    step = []
    for attr, new_val in refine_queue:
        old_val = next((val for a, val in user_input_before if a == attr), None)
        step.append({
            "attribute": attr,
            "from": old_val,
            "to": new_val
        })
    return step


# ==========================
# Tampilkan model dengan atribut terstruktur
# ==========================
def tampilkan_model(row, judul=None):
    atribut_kelompok = {
        "🔧 Spesifikasi Mesin": [
            "Displacement", "PowerHP", "Bore", "Stroke", "PistonCount", "EngineConfig"
        ],
        "⚙️ Transmisi & Struktur": [
            "Transmission", "ClutchType", "WeightKG", "Brand", "Category"
        ],
        "⛽ Konsumsi & Kapasitas": [
            "FuelTank", "FuelConsumptionKML"
        ],
        "💰 Harga": [
            "Price"
        ],
        "📊 Skor Kemiripan": [
            "Similarity"
        ]
    }

    label_mapping = {
        "Brand": "Merek",
        "Category": "Kategori",
        "Transmission": "Transmisi",
        "ClutchType": "Jenis Kopling",
        "Bore": "Diameter Silinder (mm)",
        "Stroke": "Langkah Piston (mm)",
        "PistonCount": "Jumlah Piston",
        "Displacement": "Kapasitas Mesin (cc)",
        "PowerHP": "Tenaga Maksimum (HP)",
        "EngineConfig": "Konfigurasi Mesin",
        "FuelTank": "Kapasitas Tangki (L)",
        "WeightKG": "Berat Motor (kg)",
        "FuelConsumptionKML": "Konsumsi BBM (km/L)",
        "Similarity": "Skor Kemiripan",
        "Price": "Harga (Rp)"
    }

    nama_model = row.get("Model", "Model X")
    model_judul = judul if judul else f"🏍️ **{nama_model.upper()}**"

    with st.expander(model_judul):
        for kategori, atribut_list in atribut_kelompok.items():
            st.markdown(f"**{kategori}**")
            for attr in atribut_list:
                if attr in row:
                    label = label_mapping.get(attr, attr)
                    value = row[attr]
                    if attr == "Price":
                        try:
                            formatted = f"{int(value):,}".replace(",", ".")
                        except:
                            formatted = value
                    elif attr == "Similarity":
                        try:
                            formatted = f"{float(value)*100:.4f}%"
                        except:
                            formatted = value
                    elif isinstance(value, float):
                        formatted = f"{value:.2f}"
                    else:
                        formatted = value
                    st.markdown(f"- **{label}**: {formatted}")
            st.markdown("")


# =================== STREAMLIT APP ===================

# Cek apakah ada instruksi untuk pindah ke step baru setelah reset
if "__next_step" in st.session_state:
    st.session_state.step = st.session_state["__next_step"]
    del st.session_state["__next_step"]


if st.session_state.step == "intro":
    step_intro()
elif st.session_state.step == "identity":
    step_identity()
elif st.session_state.step == "intro_query_based":
    step_intro_query_based()
elif st.session_state.step == "intro_query_for_dummies":
    step_intro_query_for_dummies()
elif st.session_state.step == "query_based":
    step_query_based()
elif st.session_state.step == "intro_CRSCBR":
    step_intro_CRSCBR()
elif st.session_state.step == "intro_CRSCBR_for_dummies":
    step_intro_CRSCBR_for_dummies()
elif st.session_state.step == "input":
    step_input()
elif st.session_state.step == "prioritas":
    step_prioritas()
elif st.session_state.step == "rekomendasi":
    step_rekomendasi()
elif st.session_state.step == "refinement":
    step_refinement()
elif st.session_state.step == "refine_prioritas":
    step_refine_prioritas()
elif st.session_state.step == "refinement_result":
    step_refinement_result()
elif st.session_state.step == "survey_1":
    step_survey_1()
elif st.session_state.step == "survey_2":
    step_survey_2()
elif st.session_state.step == "finish":
    step_finish_evaluation()
