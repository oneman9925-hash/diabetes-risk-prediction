import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide"
)

st.markdown("""
<style>
body {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1,h2,h3 {
    color: #00ffd5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("<h1 style='text-align:center;'>🧠 AI Diabetes Risk Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Random Forest | Excel Dataset</h4>", unsafe_allow_html=True)

# ---------------- Load Excel/CSV ----------------
df = pd.read_csv(
    "DiaBD Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
)

# ---------------- Data Cleaning ----------------
df = df[['bmi', 'glucose', 'systolic_bp', 'diabetic']]
df.dropna(inplace=True)

# ---------------- Model ----------------
X = df[['bmi', 'glucose', 'systolic_bp']]
y = df['diabetic']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- Sidebar Input ----------------
st.sidebar.header("🧪 Patient Input Panel")

bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
glucose = st.sidebar.slider("Glucose (mg/dL)", 70, 300, 120)
bp = st.sidebar.slider("Systolic BP", 90, 220, 120)

# ---------------- Prediction Logic ----------------
ml_pred = model.predict([[bmi, glucose, bp]])[0]

rule_based = (
    glucose >= 126 or
    (bmi >= 26 and glucose >= 115) or
    (bp >= 170 and glucose >= 110)
)

final_result = "YES" if ml_pred == 1 or rule_based else "NO"

# ---------------- Result ----------------
st.markdown("## 🔍 Prediction Result")

if final_result == "YES":
    st.error("⚠️ Diabetes Risk: POSITIVE")
else:
    st.success("✅ Diabetes Risk: NEGATIVE")

st.metric(
    label="🎯 Model Accuracy",
    value=f"{accuracy*100:.2f}%",
    delta="Random Forest Model"
)

# ---------------- Unique Graph ----------------
st.markdown("## 📊 Unique Health Pattern Visualization")

fig = px.parallel_coordinates(
    df,
    dimensions=['bmi', 'glucose', 'systolic_bp'],
    color='diabetic',
    color_continuous_scale=px.colors.sequential.Plasma
)

st.plotly_chart(fig, use_container_width=True)
