import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GenoSight · Pathogenicity Predictor",
    page_icon="🧬",
    layout="centered"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark clinical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e14;
    color: #e2e8f0;
}

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    letter-spacing: -0.5px;
    color: #7ee8a2;
    margin-bottom: 0.25rem;
}
.hero p {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
}

.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1.25rem;
}

.result-pathogenic {
    background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-benign {
    background: linear-gradient(135deg, #061a10, #0d2a1a);
    border: 1px solid #22c55e;
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-sub {
    font-size: 0.9rem;
    color: #94a3b8;
}
.result-disease {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    background: rgba(255,255,255,0.05);
    border-radius: 6px;
    font-size: 0.9rem;
    line-height: 1.6;
}
.disease-tag {
    display: inline-block;
    background: rgba(126, 232, 162, 0.12);
    border: 1px solid rgba(126, 232, 162, 0.3);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    margin: 0.15rem 0.2rem;
    font-size: 0.82rem;
    color: #7ee8a2;
    font-family: 'Space Mono', monospace;
}
.confidence-bar-wrap {
    margin-top: 1rem;
}
.confidence-label {
    font-size: 0.78rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.3rem;
}
.gene-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 4px;
    padding: 0.15rem 0.5rem;
    margin-top: 0.5rem;
    font-size: 0.82rem;
    color: #a5b4fc;
    font-family: 'Space Mono', monospace;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #065f46, #047857);
    color: #ecfdf5;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #047857, #059669);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.25);
}

hr { border-color: #1e293b; }
.stAlert { border-radius: 8px; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model      = joblib.load("xgboost_dna_model.pkl")
    nuc_map    = joblib.load("nuc_map.pkl")
    lookup     = pd.read_csv("disease_lookup.csv")

    # ── FIX 1: load chrom_order (saved by the notebook) ──────────────────
    # The notebook encodes chromosomes as integers (1-25) via chrom_order,
    # NOT via nuc_map. Without this, CHROM_encoded is always 0 and the
    # lookup table never finds a matching row → disease is always missing.
    chrom_order = joblib.load("chrom_order.pkl")

    # ── FIX 2: load the scaler saved by the notebook ──────────────────────
    # The model was trained on StandardScaler-transformed features.
    # Sending raw (unscaled) values at inference time silently corrupts
    # every prediction.
    scaler = joblib.load("scaler.pkl")

    return model, nuc_map, chrom_order, scaler, lookup

try:
    model, nuc_map, chrom_order, scaler, lookup_df = load_assets()
    assets_ok = True
except FileNotFoundError as e:
    assets_ok = False
    missing = str(e)

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
NUCLEOTIDES = ["A", "T", "G", "C"]

def encode_chrom(chrom_str, chrom_order):
    """
    Use chrom_order (the same dict saved by the notebook) so that
    chromosome encoding at inference matches training exactly.
    """
    return chrom_order.get(str(chrom_str), 0)

def encode_nuc(val, nuc_map):
    return nuc_map.get(str(val).upper(), 0)

def get_gene_features(chrom_enc, pos, lookup):
    """
    Find gene/disease info from the lookup table.
    Returns all disease names as a list so the UI can display them properly.
    """
    row = lookup[
        (lookup["CHROM_encoded"] == chrom_enc) &
        (lookup["POS"] == pos)
    ]
    if not row.empty:
        r            = row.iloc[0]
        gene_enc     = int(r.get("Gene_encoded",    0))
        num_diseases = int(r.get("Num_Diseases",    1))
        dis_unknown  = int(r.get("Disease_Unknown", 0))
        gene_name    = r.get("Gene",          None)
        raw_disease  = r.get("Disease_Name",  None)

        # Split pipe-separated disease names into a clean list
        diseases = []
        if raw_disease and str(raw_disease).lower() not in ("nan", ""):
            diseases = [
                d.strip().replace("_", " ")
                for d in str(raw_disease).split("|")
                if d.strip().lower() not in ("not_provided", "not_specified", "nan", "")
            ]
    else:
        gene_enc, num_diseases, dis_unknown = 0, 1, 0
        gene_name, diseases = None, []

    return gene_enc, num_diseases, dis_unknown, gene_name, diseases


def nearest_lookup(chrom_enc, pos, lookup, window=50_000):
    """
    If no exact position match, find the nearest variant on the same
    chromosome within `window` bases and return its disease info.
    This dramatically increases how often the app can show a disease name.
    """
    chrom_rows = lookup[lookup["CHROM_encoded"] == chrom_enc].copy()
    if chrom_rows.empty:
        return None, None, []
    chrom_rows["dist"] = (chrom_rows["POS"] - pos).abs()
    nearest = chrom_rows.loc[chrom_rows["dist"].idxmin()]
    if nearest["dist"] > window:
        return None, None, []
    gene_name = nearest.get("Gene", None)
    raw_disease = nearest.get("Disease_Name", None)
    diseases = []
    if raw_disease and str(raw_disease).lower() not in ("nan", ""):
        diseases = [
            d.strip().replace("_", " ")
            for d in str(raw_disease).split("|")
            if d.strip().lower() not in ("not_provided", "not_specified", "nan", "")
        ]
    return nearest.get("Gene_encoded", 0), gene_name, diseases

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧬 GenoSight</h1>
    <p>AI-powered DNA variant pathogenicity predictor</p>
</div>
""", unsafe_allow_html=True)

if not assets_ok:
    st.error(
        f"**Could not load model files.** Make sure all five files are in the same folder as app.py:\n\n"
        f"- `xgboost_dna_model.pkl`\n- `nuc_map.pkl`\n- `chrom_order.pkl`\n"
        f"- `scaler.pkl`\n- `disease_lookup.csv`\n\n`{missing}`"
    )
    st.stop()

# ── Input card ──
st.markdown('<div class="card"><div class="card-title">Variant Input</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    chrom = st.selectbox("Chromosome", CHROMOSOMES, index=0)
    ref   = st.selectbox("REF Allele", NUCLEOTIDES, index=0)
with col2:
    pos   = st.number_input("Position (POS)", min_value=1, value=1000000, step=1)
    alt   = st.selectbox("ALT Allele", NUCLEOTIDES, index=1)

st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("⚡ ANALYSE VARIANT")

# ── Prediction ──
if predict_btn:
    if ref == alt:
        st.warning("REF and ALT alleles are identical — this is not a variant.")
    else:
        # ── Encode inputs ──────────────────────────────────────────────────
        ref_enc   = encode_nuc(ref, nuc_map)
        alt_enc   = encode_nuc(alt, nuc_map)
        # FIX 1: use chrom_order, not nuc_map, for chromosome encoding
        chrom_enc = encode_chrom(chrom, chrom_order)

        # ── Lookup: exact match first, then nearest neighbour ──────────────
        gene_enc, num_dis, dis_unk, gene_name, diseases = get_gene_features(
            chrom_enc, int(pos), lookup_df
        )

        # If exact match had no diseases, try nearest variant on same chrom
        if not diseases:
            near_gene_enc, near_gene_name, near_diseases = nearest_lookup(
                chrom_enc, int(pos), lookup_df
            )
            if near_diseases:
                diseases  = near_diseases
                gene_name = near_gene_name or gene_name
                if near_gene_enc:
                    gene_enc = near_gene_enc

        # ── Build feature vector (must match training column order) ────────
        X_raw = np.array([[ref_enc, alt_enc, chrom_enc, int(pos),
                           gene_enc, num_dis, dis_unk]], dtype=float)

        # FIX 2: apply the same StandardScaler used during training
        X_input = scaler.transform(X_raw)

        pred       = model.predict(X_input)[0]
        proba      = model.predict_proba(X_input)[0]
        confidence = float(max(proba)) * 100

        is_pathogenic = int(pred) == 1

        # ── Build disease HTML block ───────────────────────────────────────
        if diseases:
            tags = "".join(f'<span class="disease-tag">{d}</span>' for d in diseases)
            disease_block = f'<div class="result-disease">🔬 Associated condition(s):<br>{tags}</div>'
            if gene_name:
                disease_block += f'<div class="gene-badge">Gene: {gene_name}</div>'
        else:
            disease_block = (
                '<div class="result-disease" style="color:#64748b;">'
                '⚠️ No disease association found in the lookup table for this position.'
                '</div>'
            )
            if gene_name:
                disease_block += f'<div class="gene-badge">Gene: {gene_name}</div>'

        # ── Result banner ──────────────────────────────────────────────────
        if is_pathogenic:
            st.markdown(f"""
            <div class="result-pathogenic">
                <div class="result-label" style="color:#ef4444;">⚠ PATHOGENIC</div>
                <div class="result-sub">This variant is predicted to be disease-causing.</div>
                {disease_block}
                <div class="confidence-bar-wrap">
                    <div class="confidence-label">CONFIDENCE · {confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-benign">
                <div class="result-label" style="color:#22c55e;">✓ BENIGN</div>
                <div class="result-sub">This variant is predicted to be non-pathogenic.</div>
                {disease_block}
                <div class="confidence-bar-wrap">
                    <div class="confidence-label">CONFIDENCE · {confidence:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability breakdown ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Probability breakdown"):
            class_names = {0: "Benign", 1: "Pathogenic"}
            for label, prob in zip(model.classes_, proba):
                name = class_names.get(int(label), str(label))
                st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

        with st.expander("🔩 Encoded feature vector (debug)"):
            feat_df = pd.DataFrame(
                [[ref_enc, alt_enc, chrom_enc, int(pos), gene_enc, num_dis, dis_unk]],
                columns=["REF_enc", "ALT_enc", "CHROM_enc", "POS",
                         "Gene_enc", "Num_Diseases", "Disease_Unknown"]
            )
            st.dataframe(feat_df, use_container_width=True)
            st.caption(
                f"chrom_order key used: '{chrom}' → {chrom_enc} | "
                f"Diseases found: {len(diseases)}"
            )

# ── Footer ──
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.78rem;"
    "font-family:Space Mono,monospace;'>GenoSight · for research use only</p>",
    unsafe_allow_html=True
)