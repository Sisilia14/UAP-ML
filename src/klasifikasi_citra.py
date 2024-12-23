import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
import base64

# Judul aplikasi
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px; font-size: 50px; color: #8BD9CA;">
        KLASIFIKAI MAKANAN INDONESIA
    </div>
    """,
    unsafe_allow_html=True,
)


# Fungsi untuk mengonversi gambar ke Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path gambar
image_path = r"C:\Users\azizah sisilia\Downloads\UAP\src\BG.jpeg"
base64_image = get_base64_image(image_path)

# Tambahkan CSS untuk latar belakang dengan gambar Base64
st.markdown(
    f"""
    <style>
        body {{
            background-image: url('data:image/jpeg;base64,{base64_image}'); /* Menggunakan Base64 */
            background-size: cover; /* Menyesuaikan ukuran gambar */
            background-repeat: no-repeat; /* Menghindari pengulangan gambar */
            background-position: center; /* Memusatkan gambar */
            font-family: Arial, sans-serif; /* Ganti dengan font yang diinginkan */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# CSS untuk deskripsi web
st.markdown(
    """
    <div style="text-align: justify; margin-bottom: 20px;">
            Makanan Indonesia merupakan bagian penting dari budaya dan tradisi yang kaya di tanah air. 
        Setiap daerah memiliki ciri khas dan keunikan dalam hidangannya, yang mencerminkan keberagaman budaya, rasa, dan bahan baku lokal.
    </div>

    <div style="text-align: justify; margin-bottom: 20px;">
            Sebagaimana diatur dalam berbagai kebijakan kuliner dan promosi pariwisata, 
        makanan Indonesia diharapkan dapat memperkenalkan kekayaan budaya kita kepada dunia.
    </div>

    <div style="text-align: justify; margin-bottom: 20px;">
            Oleh karena itu, aplikasi ini dibuat untuk membantu masyarakat membedakan makanan yang ada,
            seperti berasal dari mana, bahannya apa, dan apakah benar ini makanan dari Indonesia atau negara lain.
    </div>
    """,
    unsafe_allow_html=True,
)


# Fungsi prediksi
def predict(uploaded_image, model_path):
    # Daftar kelas
    class_names = [
        "Ayam Goreng",
        "Burger",
        "French Fries",
        "Gado-Gado",
        "Ikan Goreng",
        "Mie Goreng",
        "Nasi Goreng",
        "Nasi Padang",
        "Pizza",
        "Rawon",
        "Rendang",
        "Sate",
        "Soto",
    ]

    class_descriptions = [
        "Ayam goreng adalah hidangan yang terbuat dari potongan ayam yang dibumbui dan kkemudian digoreng hingga matang, ayam goreng memiiki berbagai variasi cara pengolahannyaseperti yang khas dari Indonesia yaitu ayam goreng Penyet, bumbu rujak, betawi, rempah",
        "Burger adalah makanan roti dan olahan daging cincang yang diolah dengan di panggang, makanan ini berasal dari Jerman",
        "French fries adalah makanan yang terbuat dari kentang yang dipotong menjadi batang panjang dan digoreng hingga renyah. Malanan ini berasal dari belgia / Prancis",
        "Gado-gado adalah salad khas Indonesia yang terdiri dari campuran sayuran rebus, tahu, tempe, dan telur, disajikan dengan saus kacang yang kental. Gado-Gado berasal dari Indonesia dan populer di Jawa",
        "Ikan goreng adalah hidangan yang terbuat dari ikan yang dibumbui dan kemudian digoreng hingga matang. Asal ikhan goreng tergantung dengan variasi seperti ikan goreng sambal matah khas Bali, bumbu rujak, ikan goreng rempah",
        "Mie goreng berasal dari tradisi kuliner Asia, khususnya dari negara-negara seperti Indonesia. Olahan mie goreng seperti Mie goreng jawa",
        "Nasi goreng adalah hidangan yang terbuat dari nasi yang digoreng dengan bumbu dan bahan tambahan, seperti sayuran, daging, dan telur. Berbagai cara pegolahan nasi goreng berdasarkan khas daerah seperti nasgor Aceh, Jawa, nasgor Bakar, nasgor Kampung",
        "Nasi Padang adalah hidangan khas Minangkabau dari Sumatera Barat, Indonesia, yang terdiri dari nasi putih disajikan dengan berbagai macam lauk-pauk dan sambal. Terkenal dengan cita rasa yang kaya, pedas, dan bumbu rempah yang kuat",
        "Pizza adalah hidangan yang terbuat dari adonan tepung yang dibentuk menjadi bulatan, kemudian diberi saus tomat, keju, dan berbagai topping sebelum dipanggang. Makanan ini berasal dari Italia",
        "Rawon adalah hidangan sup daging sapi khas Indonesia yang terkenal dengan kuahnya yang gelap dan kaya rempah. Kuah rawon biasanya terbuat dari kluwek (buah kepayang) yang memberikan warna hitam dan rasa yang khas. Ini merupakan makanan khas Jawa Timur",
        "Rendang adalah hidangan daging yang dimasak dengan santan dan bumbu rempah hingga empuk dan kaya rasa. Makanan ini berasal dari Minangkabau, Sumatera Barat",
        "Sate adalah hidangan daging yang ditusuk pada batang bambu dan dibakar, biasanya disajikan dengan saus kacang atau sambal. Makanan ini memiliki khas berdaarkan daerah masing-masing seperti madura, sate lilit khas Bali, sate klatak Jogja",
        "Soto adalah hidangan sup berkuah yang menggunakan daging, sayuran, dan bumbu rempah. Makanan ini berasal dari berbagai daerah di Indonesia seperti Betawi, Lamongan, Madura, dan Coto Makassar",
    ]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(
        uploaded_image, target_size=(224, 224)
    )  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Muat model
    model = tf.keras.models.load_model(model_path)

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])  # Hitung probabilitas
    return (
        class_names[np.argmax(score)],
        class_descriptions[np.argmax(score)],
        100 * np.max(score),
    )  # Prediksi label dan confidence


# Daftar model yang tersedia
models = ["VGG19", "MobileNetV2"]

# Membuat kolom untuk menampung tombol-tombol
cols = st.columns(len(models))

# Menampilkan tombol untuk setiap model
selected_model = None
for i, model in enumerate(models):
    if cols[i].button(model):
        selected_model = model

# Tentukan path model berdasarkan pilihan
if selected_model == "VGG19":
    model_path = Path(__file__).parent / "Model/Image/VGG19/model.h5"
elif selected_model == "MobileNetV2":
    model_path = Path(__file__).parent / "Model/Image/MobileNetV2/model.h5"
else:
    st.warning("Silakan pilih model untuk prediksi.")

# Komponen file uploader untuk banyak file
uploads = st.file_uploader(
    "Unggah citra untuk mendapatkan hasil prediksi",
    type=["png", "jpg"],
    accept_multiple_files=True,
)

# Tombol prediksi
if st.button("Predict", type="primary"):
    if uploads and selected_model:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(
                upload,
                caption=f"Citra yang diunggah: {upload.name}",
                use_container_width=True,
            )

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, label_description, confidence = predict(upload, model_path)
                    st.write(f"Image: **{upload.name}**")
                    st.write(f"Label : **{label}**")
                    st.write(f"Confidence: **{confidence:.5f}%**")
                    st.write(f"Keterangan Makanan: **{label_description}**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        if not uploads:
            st.error("Unggah setidaknya satu citra terlebih dahulu!")
        if not selected_model:
            st.error("Silakan pilih model terlebih dahulu!")
