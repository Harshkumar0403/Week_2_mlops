Perfect ✅ — here’s a professional, ready-to-use **`README.md`** for your Git + DVC + Iris ML project.
It explains every step clearly, from setup to training, DVC tracking, and inference with versioning.

---

## 🧠 **README.md**

```markdown
# 🌸 Week 2 MLOps — DVC + Git + Decision Tree Classifier on Iris Dataset

This project demonstrates **end-to-end machine learning version control** using  
**Git** + **DVC (Data Version Control)** integrated with **Google Cloud Storage (GCS)** for data and model tracking.

---

## 🚀 Project Overview

We use the **Iris dataset** to:
1. Train a **Decision Tree Classifier**
2. Version control the **data**, **model**, and **results** using **DVC**
3. Use **Google Cloud Storage (GCS)** as the DVC remote
4. Run inference on two dataset versions (`v1` and `v2`)
5. Effortlessly switch between dataset versions using `git checkout` + `dvc checkout`

---

## 🧩 Project Structure

```

Week_2_mlops/
├── data/
│   ├── raw/iris.csv
│   ├── v1/data.csv
│   └── v2/data.csv
├── models/
│   └── model.pkl
├── train.py
├── inference.py
├── result.txt
├── inference_v1_results.txt
├── inference_v2_results.txt
├── dvc.yaml / .dvc files
└── README.md

````

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Harshkumar0403/Week_2_mlops.git
cd Week_2_mlops
````

### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install dvc dvc-gs scikit-learn pandas joblib
```

### 3️⃣ Initialize DVC

```bash
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

### 4️⃣ Configure GCS Remote

```bash
dvc remote add -d gcsremote gs://mlops-course-phonic-axle-473506-u8-unique/dvcstore
git commit -am "Add GCS remote for DVC"
```

---

## 📊 Data Setup

### Fetch and organize data

```bash
mkdir -p data/raw data/v1 data/v2

curl -L -o data/raw/iris.csv "https://raw.githubusercontent.com/IITMBSMLOps/ga_resources/week_1/data/raw/iris.csv"
curl -L -o data/v1/data.csv "https://raw.githubusercontent.com/IITMBSMLOps/ga_resources/week_1/data/v1/data.csv"
curl -L -o data/v2/data.csv "https://raw.githubusercontent.com/IITMBSMLOps/ga_resources/week_1/data/v2/data.csv"
```

### Track data with DVC

```bash
dvc add data/raw/iris.csv data/v1/data.csv data/v2/data.csv
git add data/.gitignore data/**/*.dvc
git commit -m "Track raw and versioned datasets with DVC"
dvc push
```

---

## 🧠 Model Training

Run the training script:

```bash
python3 train.py
```

This will:

* Train a `DecisionTreeClassifier` on `data/raw/iris.csv`
* Save the trained model to `models/model.pkl`
* Save the classification report to `result.txt`

Track model & result with DVC:

```bash
dvc add models/model.pkl result.txt
git add models/model.pkl.dvc result.txt.dvc
git commit -m "Track trained model and results"
dvc push
```

---

## 🧮 Inference

Run inference for specific datasets:

### For **v1**:

```bash
python3 inference.py --data data/v1/data.csv --out inference_v1_results.txt
```

### For **v2**:

```bash
python3 inference.py --data data/v2/data.csv --out inference_v2_results.txt
```

Track inference results with DVC:

```bash
dvc add inference_v1_results.txt inference_v2_results.txt
git add *.dvc
git commit -m "Track inference results for v1 and v2"
dvc push
```

---

## 🔁 Versioning with Git + DVC

Tag dataset versions:

```bash
git tag data-v1
git tag data-v2
```

Switch between versions:

```bash
git checkout data-v1
dvc checkout
# -> Workspace synced with version 1 data/model

git checkout data-v2
dvc checkout
# -> Workspace synced with version 2 data/model
```

---

## ☁️ Remote Storage Details

All data, models, and results are pushed to:

```
gs://mlops-course-phonic-axle-473506-u8-unique/dvcstore
```

To clean the remote (if re-running):

```bash
gsutil -m rm -r gs://mlops-course-phonic-axle-473506-u8-unique/dvcstore/**
```

---

## 🧹 Clean Workspace

To remove DVC caches locally:

```bash
dvc gc -w -f
```

---

