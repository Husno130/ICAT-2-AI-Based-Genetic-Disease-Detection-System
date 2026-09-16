# 🧬 GenoSight — AI-Based Genetic Variant Pathogenicity Prediction System

GenoSight is an AI-based system for predicting the **pathogenicity of individual genetic variants** using genomic and ClinVar-derived features. It uses **XGBoost** for binary classification and provides an interactive **Streamlit** interface for single-variant analysis.

> **Research & educational use only. This project is not a clinical diagnostic tool.**

## ✨ Features

- Predicts variants as **Pathogenic** or **Benign**
- Uses genomic and annotation-derived features
- Provides prediction confidence and class probabilities
- Retrieves related gene and disease information
- Interactive Streamlit web interface
- Includes the trained model and preprocessing artifacts

## 🔬 Workflow

```text
Variant Input
     ↓
Feature Encoding
     ↓
Gene & Disease Lookup
     ↓
Feature Scaling
     ↓
XGBoost Classifier
     ↓
Pathogenic / Benign Prediction
     ↓
Confidence & Related Information
```

The training data is derived from a ClinVar VCF file. The pipeline extracts:

- `CLNSIG` → Clinical significance
- `GENEINFO` → Gene
- `CLNDN` → Disease name

The final model uses **seven engineered features**.

## 🤖 Model

**Extreme Gradient Boosting (XGBoost)** was selected as the final classifier.

```text
n_estimators = 890
random_state = 42
```

In the notebook experiments, XGBoost achieved the highest observed accuracy among the tested model families, at approximately **76.9%** on the selected test split.

## 🗂️ Project Structure

```text
ICAT-2-AI-Based-Genetic-Disease-Detection-System/
│
├── app.py
├── temporary/
│   └── main.ipynb
├── clinvar.vcf
├── disease_lookup.csv
├── xgboost_dna_model.pkl
├── scaler.pkl
├── nuc_map.pkl
├── chrom_order.pkl
├── requirements.txt
└── README.md
└── ICAT 2 Documentation 
```

## ⚙️ Installation

```bash
git clone https://github.com/Husno130/ICAT-2-AI-Based-Genetic-Disease-Detection-System
cd ICAT-2-AI-Based-Genetic-Disease-Detection-System
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run

```bash
streamlit run app.py
```

Enter the **chromosome, REF allele, genomic position, and ALT allele**, then select **Analyse Variant** to generate a prediction.

## 📊 Evaluation

The project compares Random Forest, Gradient Boosting, and XGBoost. XGBoost was selected based on the highest observed accuracy in the notebook experiments.

The current evaluation uses a **1% row-level test split** and does not include an independent external validation dataset. Therefore, the reported performance should **not be interpreted as clinical accuracy**.

## 🚀 Future Improvements

- VCF upload and batch variant analysis
- Richer sequence, conservation, allele-frequency, and functional features
- Improved gene representation
- Cross-validation and external validation
- Variant-aware data splitting to reduce possible leakage
- Additional metrics such as ROC-AUC, PR-AUC, sensitivity, specificity, and calibration
- SHAP-based prediction explanations

---

**GenoSight** · AI-based genetic variant analysis and pathogenicity prediction
