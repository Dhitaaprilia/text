from streamlit_option_menu import option_menu
import streamlit as st
import joblib
import pandas as pd

st.header("Klasifikasi Artikel Berita", divider='rainbow')
st.subheader("Oleh Anggota Kelompok 4 :")
st.write("1. Silvia Rahmawati Avrelia ( 20-062 )")
st.write("2. Dhita Aprilia Dhamayanti ( 20-102 )")
st.write("3. Annisa Putri Pawestri ( 20-110 )")
st.write("4. Citra Indah Lestari ( 20-202 )")
st.write("-------------------------------------------------------------------------------------------------------------------------")

text = st.text_area("Masukkan Artikel Berita")

button = st.button("Submit")

if "nb_reduksi" not in st.session_state:
    st.session_state.nb_reduksi = []
    st.session_state.nb_asli = []

if button:
    vectorizer = joblib.load("vectorizer.pkl")
    tfidf_matrics = vectorizer.transform([text]).toarray()
    
    # Predict Model Naive Bayes Reduksi
    #model_reduksi = joblib.load("nb (1).pkl")
    #lda = joblib.load("lda (1).pkl")
    #lda_transform = lda.transform(tfidf_matrics)
    #prediction_reduksi = model_reduksi.predict(lda_transform)
    #st.session_state.nb_reduksi = prediction_reduksi[0]
    
    # Predict Model Naive Bayes Tanpa Reduksi
    model_asli = joblib.load("NaiveBayes.pkl")
    prediction_asli = model_asli.predict(tfidf_matrics)
    st.session_state.nb_asli = prediction_asli[0]

selected = option_menu(
  menu_title="",
  options=["Dataset Information","Klasifikasi"],
  icons=["data", "Process", "model", "implemen", "Test", "sa"],
  orientation="horizontal"
  )

if selected == "Dataset Information":
    #st.write("Dataset Asli")
    st.dataframe(pd.read_csv('dataset_detik_combine.csv'), use_container_width=True)
    #st.write("Dataset Hasil Reduksi Dimensi")
    #st.dataframe(pd.read_csv('reduksi_dimensi.csv'), use_container_width=True)


elif selected == "Klasifikasi":
  st.write(f"Prediction Category : {st.session_state.nb_asli}")
        
#elif selected == "History Uji Coba":
    #st.write("Hasil Uji Coba")
    #st.dataframe(pd.read_csv('history_data.csv'), use_container_width=True)
