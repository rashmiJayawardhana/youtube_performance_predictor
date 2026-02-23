# YouTube Performance Predictor
**Predicting YouTube Video Upload Success for Sri Lankan Creators**  
*A Pre-Upload Feature Classification Study Using XGBoost*  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://youtube-performance-predictor.streamlit.app/)

## 🌐 Live Demo

> **Try the app online — no installation required:**  
> 👉 **[https://youtube-performance-predictor.streamlit.app/](https://youtube-performance-predictor.streamlit.app/)**

Enter a video title, set your channel details, choose a publish date, and get an instant prediction on whether your video will perform above your channel's median — along with feature-level explanations.

---

## Project Structure

```
youtube_performance_predictor/
├── main.py                   ← Main ML script (run this)
├── requirements.txt          ← Python dependencies
├── data/                     ← Place your Excel files here (see below)
│   ├── Content_Excel_file_-_Rasmi_Vibes.xlsx
│   ├── Content_Excel_file_-_Hey_Lee.xlsx
│   └── Sorted_by_Content_-_Timeline_of_Nuraj.xlsx
└── outputs/                  ← Auto-generated plots & CSV (created on first run)
    ├── 01_eda.png
    ├── 02_evaluation.png
    ├── 03_importance.png
    ├── 04_pdp.png
    ├── 05_shap.png
    └── dataset_combined.csv
```

---

## Setup & Run

### 1. Create and activate a virtual environment

```powershell
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Place your data files

Copy the three Excel files exported from YouTube Studio into the `data/` folder:

| File | Channel |
|---|---|
| `Content_Excel_file_-_Rasmi_Vibes.xlsx` | Rasmi Vibes (Channel A) |
| `Content_Excel_file_-_Hey_Lee.xlsx` | Hey Lee (Channel B) |
| `Sorted_by_Content_-_Timeline_of_Nuraj.xlsx` | Timeline of Nuraj (Channel C) |

> Each file must contain a sheet named **`Table data`** (YouTube Studio's default export format).

### 4. Run the script

```powershell
python main.py
```

Generated plots and the combined dataset CSV will appear in the `outputs/` folder.

---

## Algorithm
- **XGBoost** (eXtreme Gradient Boosting)
- **XAI methods**: Built-in Feature Importance, Permutation Importance, Partial Dependence Plots, SHAP-like Attribution

## Dataset
- 275 YouTube videos · 3 Sri Lankan channels · 20 pre-upload features
- No data leakage — only features available before/at upload time are used
- Channel-relative median labelling (neutralises scale differences across channels)
