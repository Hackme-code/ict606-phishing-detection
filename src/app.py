import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "final_model.joblib")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    return model, meta


model, meta = load()
features = meta["features"]
examples = meta["examples"]
phish_idx = list(model.classes_).index(-1)

st.set_page_config(page_title="Phishing Website Detector", layout="wide")
st.title("Phishing Website Detector")
st.caption(
    "ICT606 Assessment 3 demo. Tuned Gradient Boosting trained on the UCI Phishing "
    "Websites dataset. Inputs are the dataset's 30 pre-extracted features "
    "(-1 = phishing-like, 0 = suspicious, 1 = legitimate-like)."
)

# Default every feature to 1, then let a preset overwrite them
for f in features:
    st.session_state.setdefault(f"f_{f}", 1)


def load_preset():
    name = st.session_state["preset"]
    if name in examples:
        for f, v in examples[name].items():
            st.session_state[f"f_{f}"] = v


st.sidebar.header("Settings")
st.sidebar.selectbox("Load an example", ["Custom"] + list(examples),
                     key="preset", on_change=load_preset)
threshold = st.sidebar.slider(
    "Decision threshold (flag as phishing if probability is at least this)",
    0.05, 0.95, float(meta["threshold"]), 0.05)
st.sidebar.write(f"Chosen threshold from analysis: **{meta['threshold']:.2f}** "
                 "(97% phishing recall target)")

left, right = st.columns([2, 1])

with left:
    st.subheader("Website features")
    cols = st.columns(3)
    for i, f in enumerate(features):
        cols[i % 3].selectbox(f, [-1, 0, 1], key=f"f_{f}")

row = pd.DataFrame([{f: st.session_state[f"f_{f}"] for f in features}])[features]
p_phish = float(model.predict_proba(row)[0, phish_idx])

with right:
    st.subheader("Result")
    st.metric("Probability of phishing", f"{p_phish:.1%}")
    st.progress(min(max(p_phish, 0.0), 1.0))
    if p_phish >= threshold:
        st.error(f"FLAGGED AS PHISHING (threshold {threshold:.2f})")
    else:
        st.success(f"Looks legitimate (threshold {threshold:.2f})")

    st.subheader("Most influential features")
    st.caption("Ranked by permutation importance on the test set, with the values you entered.")
    top = pd.DataFrame({"Feature": meta["top_features"],
                        "Your value": [row.iloc[0][f] for f in meta["top_features"]]})
    st.dataframe(top, hide_index=True, use_container_width=True)

st.info(
    "Limitations: the model relies heavily on `URL_of_Anchor` and `SSLfinal_State`. "
    "Setting both to 1 can hide a phishing site (see the third example). "
    "A real tool would also need to extract these features from a URL and page, "
    "and would be one layer among other defences."
)