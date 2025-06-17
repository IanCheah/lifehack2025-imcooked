I'm cooked!

Dataset used for training CNN model are taken from kaggle.
https://www.kaggle.com/datasets/sakshivyavahare20/color-blindness-simulation-and-correction?resource=download


### 🔁 Prerequisites

- Python 3.10+  
- Node.js 16+ & npm/yarn  
- `poppler-utils` (for `pdf2image`)  
  - Ubuntu/debian: `sudo apt install poppler-utils`  
  - macOS (Homebrew): `brew install poppler`

# 🚀 Setup

## 1. Clone repository
```bash
git clone https://github.com/IanCheah/lifehack2025-imcooked.git
cd lifehack2025-imcooked
```

## 2. Backend setup
```bash
 # Setup virtual environment
python3 -m venv .venv
# Activate virtual environment
source .venv/bin/activate # For MacOS
.venv\Scripts\activate # For Windows
# Install dependencies
pip install requirements.txt
```

## 3. Frontend setup
Install dependencies:
```bash
cd lifehack2025
npm install
```

## 4. Start backend server
```bash
cd backend
uvicorn main:app --reload
```

## 5. Start frontend server
```bash
cd lifehack2025
npm run dev
```
