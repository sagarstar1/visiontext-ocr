# 🔍 VisionText OCR

> A production-grade Optical Character Recognition web app built with Python, OpenCV, and Tesseract — deployable in one click.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?style=flat-square)
![Tesseract](https://img.shields.io/badge/Tesseract-5.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼 Image Upload | PNG, JPG, BMP, TIFF, WebP support |
| ⚙ Preprocessing Pipeline | Adaptive thresholding, Otsu, denoising, auto-deskew |
| 🔍 OCR Extraction | Tesseract with configurable PSM modes |
| 📊 Confidence Scoring | Per-word accuracy scores with visual indicators |
| 🌐 Multi-language | English + Hindi (extensible) |
| ⬇ Export | TXT, CSV, JSON download |
| 🎨 Dark UI | Professional dark-themed interface |

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- Tesseract installed on your system

**Install Tesseract:**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/visiontext-ocr.git
cd visiontext-ocr

pip install -r requirements.txt

streamlit run app.py
```

App runs at `http://localhost:8501`

---

## ☁ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as main file
4. Click **Deploy** — done! ✅

Streamlit Cloud auto-installs `packages.txt` (Tesseract) and `requirements.txt`.

---

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Image Processing:** OpenCV (thresholding, denoising, deskew)
- **OCR Engine:** Tesseract 5.x via pytesseract
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit Cloud

---

## 📁 Project Structure

```
visiontext-ocr/
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (Tesseract)
├── .streamlit/
│   └── config.toml         # Theme & server config
└── README.md
```

---

## 📸 How It Works

1. **Upload** any image containing text
2. **Preprocess** — OpenCV cleans, denoises, and thresholds the image
3. **Extract** — Tesseract engine reads text from the processed image
4. **Review** — Confidence scores highlight reliability per word
5. **Export** — Download results as TXT, CSV, or JSON

---

## 🔮 Roadmap

- [ ] PDF batch processing
- [ ] REST API (FastAPI wrapper)
- [ ] Receipt/Invoice field parser
- [ ] Table extraction → Excel export
- [ ] Handwriting recognition (TrOCR)

---

## 👤 Author

Built by **[Your Name]** · [LinkedIn](#) · [Portfolio](#)

---

## 📄 License

MIT License — free to use, modify, and deploy.
