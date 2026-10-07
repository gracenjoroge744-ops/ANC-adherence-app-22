import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ANC Adherence Predictor", page_icon="🤰")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_village_map():
    df = pd.read_csv("village_grp.csv")
    df = df.drop_duplicates(subset="Village").sort_values("Village")
    return dict(zip(df["Village"], df["Village_grp"]))


model = load_model()
village_to_grp = load_village_map()

MARITAL = ["single", "married"]
CLASS_1_LABEL = "Adhered to ANC"
CLASS_0_LABEL = "Did not adhere to ANC"

st.title("ANC Adherence Prediction")
st.caption("Prototype for illustration only. Not for clinical decision-making.")

age = st.number_input("Age", min_value=10, max_value=60, value=20, step=1)
village = st.selectbox("Village", list(village_to_grp.keys()))
distance = st.number_input(
    "Distance to facility (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
)
marital = st.selectbox("Marital status", MARITAL)

if st.button("Predict"):
    row = pd.DataFrame(
        [
            {
                "Age": age,
                "Village": village,
                "Distance_km": distance,
                "Marital status": marital,
                "Village_grp": village_to_grp[village],
            }
        ]
    )
    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    p1 = float(proba[list(model.classes_).index(1)])

    label = CLASS_1_LABEL if pred == 1 else CLASS_0_LABEL
    st.subheader(f"Prediction: {label}")
    st.write(f"Probability of adherence: *{p1:.1%}*")
    st.progress(min(max(p1, 0.0), 1.0))
