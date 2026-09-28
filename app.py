"""
app.py - MedScript web app (Streamlit).

    streamlit run app.py

Opens in your browser at http://localhost:8501
"""
import pandas as pd
import streamlit as st

from src import config
from src.correction import analyse_prediction, find_indian_equivalents, fuzzy_search
from src.predictor import Predictor, available_models
from src.preprocessing import load_gray, preprocess_steps

st.set_page_config(page_title="MedScript", page_icon="💊", layout="wide")

MODEL_LABELS = {"cnn": "ResNet-18 CNN (transfer learning)", "baseline": "HOG + Logistic Regression (baseline)"}


# ---------------------------------------------------------------------------
# Cached loaders: Streamlit re-runs this whole script on every click, so we
# cache anything slow (models, the 254k-row medicine database).
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model(key):
    return Predictor(prefer=key)


@st.cache_data
def get_equivalents(generic):
    return find_indian_equivalents(generic, max_rows=10)


@st.cache_data
def get_test_samples():
    """A few test images so you can demo without uploading anything."""
    try:
        from src.data_loader import load_split
        test = load_split("test")
        return test.groupby("medicine").head(1).reset_index(drop=True)
    except FileNotFoundError:
        return None


def show_equivalents(generic):
    table, summary = get_equivalents(generic)
    st.markdown(f"#### Indian equivalents for **{generic}**")
    if summary["n_products"] == 0:
        st.info("No single-ingredient match in the Indian medicine database "
                "(it mainly covers allopathic drugs, not supplements).")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Products with same ingredient", f"{summary['n_products']:,}")
    c2.metric("Manufacturers", f"{summary['n_manufacturers']:,}")
    c3.metric(f"Price range ({summary['price_basis']})",
              f"₹{summary['min_price']:.2f} – ₹{summary['max_price']:.2f}",
              help=f"Median ₹{summary['median_price']:.2f}")
    st.dataframe(table, hide_index=True)
    st.caption("Cheapest first. Strength (mg) is not visible in a single word, "
               "so the pharmacist must match the strength on the prescription.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("💊 MedScript")
st.sidebar.write("Reads a handwritten medicine name, checks how sure it is, "
                 "and finds Indian equivalents.")
models = available_models()
if not models:
    st.error("No trained model found. Run `python -m src.train_baseline` first.")
    st.stop()
default = models.index("cnn") if "cnn" in models else 0
model_key = st.sidebar.radio("Model", models, index=default, format_func=MODEL_LABELS.get)
threshold = st.sidebar.slider("Confidence threshold", 0.3, 0.95, float(config.CONFIDENCE_THRESHOLD), 0.05,
                              help="Below this, the word is sent to a pharmacist to verify.")
st.sidebar.caption("⚠️ Assistive tool only - a pharmacist makes the final decision.")
model = get_model(model_key)

tab_read, tab_search, tab_perf = st.tabs(["📷 Read handwriting", "🔎 Search by text", "📊 Model performance"])

# ---------------------------------------------------------------------------
# Tab 1: read an image
# ---------------------------------------------------------------------------
with tab_read:
    st.header("Read a handwritten medicine name")
    uploaded = st.file_uploader("Upload a cropped image of ONE handwritten word",
                                type=["png", "jpg", "jpeg"])
    gray, true_label = None, None
    if uploaded is not None:
        gray = load_gray(uploaded.getvalue())
    else:
        samples = get_test_samples()
        if samples is not None:
            choice = st.selectbox("...or try an image from the TEST set (never seen in training)",
                                  samples["medicine"].tolist())
            row = samples[samples["medicine"] == choice].iloc[0]
            gray, true_label = load_gray(row["image_path"]), choice

    if gray is not None:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Image")
            st.image(gray, caption="Input", clamp=True)
            with st.expander("See the image-processing steps"):
                for title, img in preprocess_steps(gray, config.BASELINE_IMG_SIZE).items():
                    st.image(img, caption=title, clamp=True)
        with right:
            st.subheader("Prediction")
            probs = model.predict_proba([gray])[0]
            result = analyse_prediction(probs, model.classes, threshold=threshold)
            box = {"ACCEPT": st.success, "INGREDIENT_ONLY": st.warning, "VERIFY": st.error}[result["status"]]
            box(result["message"])
            top = pd.DataFrame(result["top"]).rename(
                columns={"brand": "Brand", "generic": "Active ingredient", "prob": "Confidence"})
            st.dataframe(top, hide_index=True,
                         column_config={"Confidence": st.column_config.ProgressColumn(
                             format="%.2f", min_value=0.0, max_value=1.0)})
            if true_label is not None:
                ok = true_label == result["top"][0]["brand"]
                st.caption(f"Correct answer: **{true_label}** {'✅' if ok else '❌'}")
        show_equivalents(result["generic"])

# ---------------------------------------------------------------------------
# Tab 2: fuzzy text search
# ---------------------------------------------------------------------------
with tab_search:
    st.header("Search by text (typos are fine)")
    st.write("Type what you *think* the prescription says. Fuzzy matching finds the closest real names.")
    query = st.text_input("Medicine name", placeholder="e.g. Montiar, crocn, Azithrosin")
    if query:
        with st.spinner("Searching 135,000+ brand names..."):
            dataset_matches, indian_matches = fuzzy_search(query)
        c1, c2 = st.columns(2)
        c1.markdown("**In the prescription dataset**")
        c1.dataframe(pd.DataFrame(dataset_matches), hide_index=True)
        c2.markdown("**In the Indian medicine database**")
        c2.dataframe(pd.DataFrame(indian_matches), hide_index=True)
        if dataset_matches:
            show_equivalents(dataset_matches[0]["active ingredient"])

# ---------------------------------------------------------------------------
# Tab 3: model performance (reads the files saved by evaluate.py)
# ---------------------------------------------------------------------------
with tab_perf:
    st.header("How good is the model? (test set)")
    comparison = config.REPORTS_DIR / "model_comparison.csv"
    if comparison.exists():
        st.dataframe(pd.read_csv(comparison), hide_index=True)
    else:
        st.info("Run `python -m src.evaluate` to generate the results.")
    ablation = config.REPORTS_DIR / "ablation_preprocessing.csv"
    if ablation.exists():
        st.subheader("Does the image processing help? (validation accuracy)")
        st.dataframe(pd.read_csv(ablation), hide_index=True)
    figures = [
        ("model_comparison.png", "Model comparison"),
        ("cnn_confidence_tradeoff.png", "Automation vs. accuracy (CNN)"),
        ("baseline_confidence_tradeoff.png", "Automation vs. accuracy (baseline)"),
        ("cnn_training_curves.png", "CNN training curves"),
        ("cnn_failures.png", "Where the CNN fails"),
        ("baseline_failures.png", "Where the baseline fails"),
    ]
    for name, title in figures:
        path = config.FIGURES_DIR / name
        if path.exists():
            st.subheader(title)
            st.image(str(path))
