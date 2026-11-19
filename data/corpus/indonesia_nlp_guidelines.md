# Panduan Preprocessing & Evaluasi NLP Bahasa Indonesia

## Karakteristik Khusus Teks Bahasa Indonesia
Pemrosesan Bahasa Alami (Natural Language Processing / NLP) untuk teks Bahasa Indonesia menghadapi beberapa tantangan khas dibanding Bahasa Inggris:
- **Morfologi Aglutinatif**: Satu kata dasar dapat membentuk banyak variasi melalui imbuhan awalan, sisipan, akhiran, dan konfiks (contoh: *ajar* -> *belajar*, *mengajar*, *pelajaran*, *pengajaran*; *lindung* -> *melindungi*, *pelindungan*, *perlindungan*). Penggunaan char n-gram (3–5 karakter) atau subword tokenizer sangat membantu menjaga kemiripan semantik antar variasi morfologi ini.
- **Ragam Bahasa Tidak Baku & Slang (Alay/Colloquial)**: Data media sosial, tiket dukungan pelanggan, dan ulasan e-commerce penuh dengan singkatan seperti *yg* (yang), *gk/ga/nggak* (tidak), *bgt* (banget), *udh* (sudah), *dgn* (dengan), *blm* (belum), serta campur kode (*code-mixing*) Indonesia-Inggris-daerah (Jawa, Sunda, Betawi).
- **Reduplikasi Kata**: Kata ulang seperti *undang-undang*, *masing-masing*, dan *hati-hati* memiliki makna semantik utuh yang tidak boleh dipotong sembarangan oleh regex pembersih tanda baca.

## Pipeline Normalisasi Teks dan Penanganan Slang
Urutan preprocessing yang direkomendasikan untuk produksi NLP Bahasa Indonesia:
1. **Unicode Normalization (NFKC) & Pembersihan Noise**: Hapus karakter kontrol tak terlihat, normalisasi tanda kutip pintar, dan pertahankan angka penting seperti pasal, persentase, serta nominal Rupiah (`Rp5 miliar`, `3x24 jam`).
2. **Normalisasi Slang Berbasis Kamus Terkurasi**: Gunakan kamus slang (*colloquial Indonesian lexicon*) hanya untuk tugas klasifikasi teks atau pencarian non-formal; hindari normalisasi agresif pada dokumen hukum atau medis karena dapat mengubah akronim teknis (*UU*, *RS*, *KMS*, *DPO*).
3. **Penanganan Stopwords Secara Selektif**: Untuk tugas analisis sentimen dan deteksi negasi, **jangan menghapus kata negasi** seperti *tidak*, *bukan*, *belum*, *jangan*, dan *tanpa* karena menghapusnya akan membalikkan polaritas makna kalimat secara total.
4. **Stemming vs Subword**: Stemmer berbasis aturan (seperti Sastrawi / Nazief-Adriani) efektif untuk indeks pencarian BM25 skala kecil, namun berisiko *over-stemming* pada kata serapan modern. Untuk sistem hybrid modern, kombinasikan token utuh pada BM25 dengan subword/character n-gram pada retrieval vektor.

## Deteksi dan Masking PII (Personally Identifiable Information)
Sebelum teks Bahasa Indonesia digunakan untuk pelatihan model, evaluasi, atau dikirim ke API LLM, pipeline wajib menjalankan **PII Masking** otomatis untuk mematuhi UU PDP:
- **NIK (Nomor Induk Kependudukan)**: pola 16 digit numerik (`\b\d{16}\b`) diganti token `[NIK_REDACTED]`.
- **Nomor Telepon Indonesia**: awalan `+62`, `62`, atau `08` dengan panjang 10–13 digit diganti `[PHONE_REDACTED]`.
- **NPWP dan Nomor Rekening**: pola 15–16 digit pajak atau nomor rekening bank diganti `[FINANCIAL_ID_REDACTED]`.
- **Email dan Alamat Rumah**: deteksi pola email standar serta Named Entity Recognition (NER) untuk nama orang dan alamat jalan sebelum penyimpanan log evaluasi.

## Metrik Evaluasi NLP untuk Data Imbalanced
Dataset produksi di Indonesia (seperti deteksi penipuan transaksi, klasifikasi tiket keluhan prioritas tinggi, atau moderasi konten berbahaya) hampir selalu mengalami ketimpangan kelas (**class imbalance**, misalnya kelas positif hanya 1%–5%).
- **Hindari Accuracy sebagai Metrik Utama**: Pada dataset dengan 98% kelas negatif, model dummy yang selalu memprediksi negatif tetap mencapai Accuracy 98% namun gagal total mendeteksi kasus kritis.
- **Gunakan Macro-F1 dan PR-AUC**: **Macro-F1** memberikan bobot setara bagi setiap kelas sehingga kegagalan pada kelas minoritas langsung terlihat tajam. Untuk klasifikasi biner yang sangat timpang, laporkan **PR-AUC (Precision-Recall Area Under Curve)** bersama Recall pada batas False Positive Rate tertentu.
- **Evaluasi Retrieval & QA**: Gunakan **Recall@k**, **MRR**, dan **NDCG@k** pada query campuran Bahasa Indonesia baku, semi-formal, dan istilah teknis Inggris, serta lakukan error analysis per kategori panjang dokumen dan bahasa.
