import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ANC Adherence Predictor", page_icon="🤰", layout="wide")

# ----------------------------- settings -----------------------------
WHO_CONTACT_WEEKS = [12, 20, 26, 30, 34, 36, 38, 40]  # WHO 8-contact ANC schedule
MARITAL = ["single", "married"]
FAR_KM = 8          # distance (km) at or above which "far from facility" advice shows
HIGH_CUTOFF = 0.40  # probability of adherence below this = high risk of non-adherence
MOD_CUTOFF = 0.60   # below this = moderate risk


# ----------------------------- loading ------------------------------
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


# ----------------------------- helpers ------------------------------
def expected_visits(ga_weeks):
    return sum(1 for w in WHO_CONTACT_WEEKS if ga_weeks >= w)


def risk_tier(p_adhere):
    if p_adhere < HIGH_CUTOFF:
        return "High"
    if p_adhere < MOD_CUTOFF:
        return "Moderate"
    return "Low"


def recommendations(p_adhere, age, distance, ga, attended):
    tier = risk_tier(p_adhere)
    recs = []
    if tier == "High":
        recs.append(
            "Priority follow-up: phone call or SMS reminder before the next due visit; "
            "community health volunteer (CHV) home visit if she cannot be reached."
        )
    elif tier == "Moderate":
        recs.append("Send a reminder for the next visit and confirm the date with the client.")
    else:
        recs.append("Routine reminders; reinforce the importance of completing all ANC contacts.")

    if pd.notna(ga) and pd.notna(attended):
        gap = expected_visits(ga) - int(attended)
        if int(attended) == 0 and ga >= 12:
            recs.append("No ANC contact recorded yet: arrange the first visit urgently.")
        elif gap > 0:
            recs.append(
                f"{gap} contact(s) behind the WHO schedule: trace and offer a catch-up visit soon."
            )
        if int(attended) < len(WHO_CONTACT_WEEKS):
            recs.append(
                f"Next WHO contact is due at about {WHO_CONTACT_WEEKS[int(attended)]} weeks of gestation."
            )

    if distance >= FAR_KM:
        recs.append(
            f"Lives {distance:g} km away: consider outreach clinics, CHV linkage, "
            "transport support, or bundling services into fewer trips."
        )
    if age < 20:
        recs.append("Adolescent (under 20): offer youth-friendly counselling and, if she agrees, involve a support person.")
    return recs


def predict_proba_adherence(df):
    X = pd.DataFrame(
        {
            "Age": df["Age"],
            "Village": df["Village"],
            "Distance_km": df["Distance_km"],
            "Marital status": df["Marital status"],
            "Village_grp": df["Village"].map(village_to_grp).fillna("Other"),
        }
    )
    idx = list(model.classes_).index(1)
    return model.predict_proba(X)[:, idx]


# ------------------------------- UI ---------------------------------
st.title("ANC Adherence Prediction")
st.caption(
    "Prototype for illustration only. The model gives a rough estimate and must not "
    "replace clinical judgement."
)

tab_single, tab_dash = st.tabs(["Single client", "Clinician dashboard"])

# ------------------------- single client tab ------------------------
with tab_single:
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20, step=1)
        village = st.selectbox("Village", list(village_to_grp.keys()))
        distance = st.number_input(
            "Distance to facility (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
        )
        marital = st.selectbox("Marital status", MARITAL)
    with c2:
        ga = st.number_input("Gestational age (weeks)", min_value=1, max_value=42, value=20, step=1)
        attended = st.number_input("ANC visits attended so far", min_value=0, max_value=12, value=1, step=1)
        st.caption(
            "Gestational age and visits attended are not inputs to the prediction model. "
            "They are used for the schedule check and recommendations."
        )

    if st.button("Predict", type="primary"):
        row = pd.DataFrame(
            [{"Age": age, "Village": village, "Distance_km": distance, "Marital status": marital}]
        )
        p = float(predict_proba_adherence(row)[0])
        tier = risk_tier(p)

        st.subheader("Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of adherence", f"{p:.1%}")
        m2.metric("Risk of non-adherence", tier)
        m3.metric("Prediction", "Adhered" if p >= 0.5 else "Did not adhere")
        st.progress(min(max(p, 0.0), 1.0))

        exp = expected_visits(ga)
        gap = exp - int(attended)
        s1, s2, s3 = st.columns(3)
        s1.metric("Expected visits by now (WHO)", exp)
        s2.metric("Visits attended", int(attended))
        s3.metric("Visits behind schedule", max(gap, 0))

        st.subheader("Recommendations")
        for r in recommendations(p, age, distance, ga, attended):
            st.markdown(f"- {r}")

# --------------------------- dashboard tab --------------------------
with tab_dash:
    pin_required = st.secrets.get("CLINICIAN_PIN", None) if hasattr(st, "secrets") else None
    allowed = True
    if pin_required:
        pin = st.text_input("Clinician PIN", type="password")
        allowed = pin == str(pin_required)
        if pin and not allowed:
            st.error("Incorrect PIN.")
    else:
        st.info(
            "No PIN is set, so anyone with the link can open this tab. "
            "Set CLINICIAN_PIN in the app's Secrets to protect it."
        )

    if allowed:
        st.write(
            "Upload a CSV of clients to get risk levels, schedule gaps and recommendations "
            "for the whole clinic. Data is processed in your browser session and is not stored."
        )
        template = pd.DataFrame(
            {
                "Patient_ID": ["P001", "P002"],
                "Age": [17, 24],
                "Village": ["ahero", "bahati"],
                "Distance_km": [11.5, 3.0],
                "Marital status": ["single", "married"],
                "Gestational_weeks": [22, 30],
                "Visits_attended": [1, 3],
            }
        )
        st.download_button(
            "Download CSV template",
            template.to_csv(index=False),
            file_name="anc_template.csv",
            mime="text/csv",
        )
        up = st.file_uploader("Upload client CSV", type="csv")

        if up is not None:
            # sep=None sniffs commas/semicolons; utf-8-sig removes a hidden BOM
            data = pd.read_csv(up, sep=None, engine="python", encoding="utf-8-sig")
            original_cols = list(data.columns)

            def _key(name):
                return "".join(ch for ch in str(name).lower() if ch.isalnum())

            ALIASES = {
                "age": "Age",
                "village": "Village",
                "distancekm": "Distance_km",
                "distance": "Distance_km",
                "maritalstatus": "Marital status",
                "marital": "Marital status",
                "gestationalweeks": "Gestational_weeks",
                "gestationalage": "Gestational_weeks",
                "ga": "Gestational_weeks",
                "visitsattended": "Visits_attended",
                "numberofvisits": "Visits_attended",
                "patientid": "Patient_ID",
                "id": "Patient_ID",
            }
            data = data.rename(columns={c: ALIASES[_key(c)] for c in data.columns if _key(c) in ALIASES})

            required = ["Age", "Village", "Distance_km", "Marital status"]
            missing = [c for c in required if c not in data.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.write("Columns found in your file:", original_cols)
            else:
                data["Village"] = data["Village"].astype(str).str.strip().str.lower()
                data["Marital status"] = data["Marital status"].astype(str).str.strip().str.lower()
                if "Gestational_weeks" not in data.columns:
                    data["Gestational_weeks"] = float("nan")
                if "Visits_attended" not in data.columns:
                    data["Visits_attended"] = float("nan")
                if "Patient_ID" not in data.columns:
                    data["Patient_ID"] = range(1, len(data) + 1)

                try:
                    data["P_adherence"] = predict_proba_adherence(data)
                except Exception as e:
                    st.error(
                        "Could not score this file. Check that Village and Marital status "
                        f"use the same spellings as the training data.\n\nDetails: {e}"
                    )
                    st.stop()

                data["Risk"] = data["P_adherence"].apply(risk_tier)
                data["Expected_visits"] = data["Gestational_weeks"].apply(
                    lambda g: expected_visits(g) if pd.notna(g) else float("nan")
                )
                data["Visits_behind"] = (data["Expected_visits"] - data["Visits_attended"]).clip(lower=0)
                data["Recommendations"] = data.apply(
                    lambda r: " | ".join(
                        recommendations(
                            r["P_adherence"], r["Age"], r["Distance_km"],
                            r["Gestational_weeks"], r["Visits_attended"],
                        )
                    ),
                    axis=1,
                )

                f1, f2 = st.columns(2)
                villages = sorted(data["Village"].unique())
                sel_v = f1.multiselect("Filter by village", villages, default=villages)
                sel_r = f2.multiselect(
                    "Filter by risk", ["High", "Moderate", "Low"], default=["High", "Moderate", "Low"]
                )
                view = data[data["Village"].isin(sel_v) & data["Risk"].isin(sel_r)]

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Clients", len(view))
                k2.metric("High risk", int((view["Risk"] == "High").sum()))
                k3.metric("Behind schedule", int((view["Visits_behind"] > 0).sum()))
                k4.metric(
                    "Avg. probability of adherence",
                    f"{view['P_adherence'].mean():.0%}" if len(view) else "-",
                )

                g1, g2 = st.columns(2)
                with g1:
                    st.caption("Clients by risk level")
                    st.bar_chart(
                        view["Risk"].value_counts().reindex(["High", "Moderate", "Low"]).fillna(0)
                    )
                with g2:
                    st.caption("High-risk clients by village")
                    hr = view[view["Risk"] == "High"]["Village"].value_counts()
                    if len(hr):
                        st.bar_chart(hr)
                    else:
                        st.write("None in the current selection.")

                st.subheader("Client list (lowest likelihood of adherence first)")
                show = view.sort_values("P_adherence")[
                    [
                        "Patient_ID", "Age", "Village", "Distance_km", "Gestational_weeks",
                        "Visits_attended", "Expected_visits", "Visits_behind",
                        "P_adherence", "Risk", "Recommendations",
                    ]
                ].copy()
                show["P_adherence"] = show["P_adherence"].round(3)
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download results (CSV)",
                    show.to_csv(index=False),
                    file_name="anc_results.csv",
                    mime="text/csv",
                )
