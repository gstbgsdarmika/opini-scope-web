import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from models.machine.preprocessing.preprocessing import TextPreprocessor
from flask import Flask, request, jsonify, render_template, flash, send_file, redirect, session, url_for, send_from_directory

app = Flask(__name__, static_url_path='/static')

# Route untuk halaman index / beranda
@app.route("/")
def index():
    active_page = 'index'
    return render_template("index.html", active_page=active_page)

# Route untuk halaman analisis
@app.route("/analisis")
def analysis():
    active_page = 'analysis'
    return render_template("analysis.html", active_page=active_page)

# Route untuk halaman analisis file
@app.route("/analisis-file", methods=['GET', 'POST'])
def analysisInputFile():
    active_page = 'analysis'
    page = request.args.get('page', 1, type=int) 
    
    if request.method == 'POST':
        if 'file-input' not in request.files:
            return render_template('analysisInputFile.html', active_page=active_page, error='File tidak ditemukan')
        
        file = request.files['file-input']
        if file.filename == '':
            return render_template('analysisInputFile.html', active_page=active_page, error='File tidak ditemukan')
        
        try:
            # Memanggil model yang digunakan
            def load_model(file_path):
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            
            tfidf_vectorizer = load_model('models/machine/tfidf_vectorizer.pkl')
            feature_selector = load_model('models/machine/feature_selector.pkl')
            svm_classifier = load_model('models/machine/svm_classifier.pkl')
            
            # Baca file CSV menggunakan Pandas
            filename = file
            df = pd.read_csv(filename, encoding='latin-1')

            # Periksa keberadaan kolom 'tweet'
            if 'komentar' not in df.columns:
                return render_template('analysisInputFile.html', active_page=active_page, error='Kolom komentar tidak ditemukan dalam file CSV')

            # Mengambil hanya kolom 'tweet'
            df = df[['komentar']]

            # Proses preprocessing data
            preprocessor = TextPreprocessor()
            df['clean_comment'] = df['komentar'].apply(preprocessor.preprocess_text)

            # Ubah tweet menjadi vektor fitur TF-IDF
            tfidf_features = tfidf_vectorizer.transform(df['clean_comment'])

            # Lakukan pemilihan fitur
            selected_features = feature_selector.transform(tfidf_features)

            # Buat prediksi menggunakan model SVM
            predictions = svm_classifier.predict(selected_features)

            # Tambahkan prediksi ke DataFrame
            df['prediction'] = predictions

            # Menghitung jumlah data yang dilabelkan positif dan negatif
            positive_data = sum(predictions == 1)
            negative_data = sum(predictions == 0)

            # Menghitung persentase data positif dan negatif dari hasil prediksi
            total_data = len(df)
            positive_percentage = (positive_data / total_data) * 100
            negative_percentage = (negative_data / total_data) * 100

            # Logika untuk paginasi
            per_page = 10 
            pages = total_data // per_page + (1 if total_data % per_page > 0 else 0) 

            # Batasi data yang ditampilkan berdasarkan nomor halaman
            start_index = (page - 1) * per_page
            end_index = min(start_index + per_page, total_data)
            sliced_df = df.iloc[start_index:end_index]
            
            # Path untuk menyimpan file CSV sementara
            temporary_csv_path = 'database/hasil analisis.csv'
            
            # Membuat file CSV
            csv_columns = ['Komentar', 'Preprocessing', 'Sentimen']
            with open(temporary_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for index, row in df.iterrows():
                    writer.writerow({
                            'Komentar': row['komentar'],
                            'Preprocessing': row['clean_comment'], 
                            'Sentimen': 'positif' if row['prediction'] == 1 else 'negatif'})
        
            return render_template('resultInputFile.html', active_page=active_page, 
                    df=df, 
                    positive_data=positive_data, negative_data=negative_data,
                    positive_percentage=positive_percentage, negative_percentage=negative_percentage,
                    pages=pages, page=page, start_index=start_index, end_index=end_index, total_data=total_data)

        except pd.errors.EmptyDataError:
            return render_template('analysisInputFile.html',active_page=active_page, error='Data tidak ditemukan')
        except pd.errors.ParserError:
            return render_template('analysisInputFile.html', active_page=active_page, error='Data tidak ditemukan')
        except IOError as e:
            print("I/O error:", e)
            return render_template('analysisInputFile.html', active_page=active_page, error='Gagal menyimpan file CSV')
    return render_template("analysisInputFile.html", active_page=active_page)

# Route untuk halaman analisis teks
@app.route("/analisis-teks", methods=['GET', 'POST'])
def analysisInputText():
    active_page = 'analysis'
    if request.method == 'POST':
        text = request.form.get('text-input')
        
        # Periksa apakah teks tidak kosong
        if not text:
            error_message = "Tidak boleh kosong. Silakan masukkan teks!"
            return render_template("analysisInputText.html", active_page=active_page, error_message=error_message)

        # Memanggil model yang digunakan
        def load_model(file_path):
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
            
        tfidf_vectorizer = load_model('models/machine/tfidf_vectorizer.pkl')
        feature_selector = load_model('models/machine/feature_selector.pkl')
        svm_classifier = load_model('models/machine/svm_classifier.pkl')

        # Proses preprocessing data
        preprocessor = TextPreprocessor()  
        clean_sentence = preprocessor.preprocess_text(text)
        
        # Ubah tweet menjadi vektor fitur TF-IDF
        new_sentence_tfidf = tfidf_vectorizer.transform([clean_sentence])
        
        # Lakukan pemilihan fitur
        new_sentence_feature_selector = feature_selector.transform(new_sentence_tfidf)
        
        # Buat prediksi menggunakan model SVM
        prediction = svm_classifier.predict(new_sentence_feature_selector)
        
        # Mengubah nilai prediksi jadi positif dan negatif
        prediction_label = "positif" if prediction == 1 else "negatif"

        return render_template("resultInputText.html", active_page=active_page, text=text, clean_sentence=clean_sentence, prediction_label=prediction_label)

    return render_template("analysisInputText.html", active_page=active_page)

# Route untuk halaman hasil analisis file
@app.route("/hasil-analisis-file")
def resultInputFile():
    active_page = 'analysis'
    return render_template("resultInputFile.html", active_page=active_page)

# Route untuk halaman hasil analisis teks
@app.route("/hasil-analisis-teks")
def resultInputText():
    active_page = 'analysis'
    return render_template("resultInputText.html", active_page=active_page)

# Route untuk mendownload file hasil analisis
@app.route('/download-csv')
def download_csv():
    filename = 'database/hasil analisis.csv'
    return send_file(filename, as_attachment=True)

# Route untuk halaman 404
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@app.route("/preline.js")
def serve_preline_js():
    return send_from_directory("node_modules/preline/dist", "preline.js")

@app.route("/apexcharts.min.js")
def serve_apexcharts_js():
    return send_from_directory("node_modules/apexcharts/dist", "apexcharts.min.js")

@app.route("/lodash.min.js")
def serve_lodash_js():
    return send_from_directory("node_modules/lodash", "lodash.min.js")

@app.route("/apexcharts.min.js")
def serve_apexcharts_css():
    return send_from_directory("node_modules/apexcharts/dist", "apexcharts.css")

if __name__ == '__main__':
    app.run(debug=True)