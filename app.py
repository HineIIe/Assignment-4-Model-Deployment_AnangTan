import streamlit as st
import pandas as pd
import pickle

from src.preprocess import preprocess_data




st.set_page_config(
    page_title="Spaceship Titanic Prediction",
    layout="wide"
)


# =========================
# Load Model (Cached)
# =========================

@st.cache_resource
def load_model():

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_model()


# =========================
# Title
# =========================

st.title("🚀 Spaceship Titanic Transport Prediction")
st.write("ASG 04 MD - **Anang Tantowi**")


# =========================
# Sidebar Inputs
# =========================

st.sidebar.header("Passenger Information")

HomePlanet = st.sidebar.selectbox(
    "Home Planet",
    ["Earth", "Europa", "Mars"]
)

CryoSleep = st.sidebar.selectbox(
    "CryoSleep",
    [True, False]
)

Destination = st.sidebar.selectbox(
    "Destination",
    ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]
)

Age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

VIP = st.sidebar.selectbox(
    "VIP",
    [True, False]
)


st.sidebar.header("Cabin Information")

Cabin = st.sidebar.text_input(
    "Cabin (example: B/123/P)",
    "B/0/P"
)

PassengerGroup = st.sidebar.number_input(
    "Passenger Group",
    min_value=1,
    value=1
)

CabinSide = st.sidebar.selectbox(
    "Cabin Side",
    ["P", "S"]
)


st.sidebar.header("Spending")

RoomService = st.sidebar.number_input(
    "Room Service",
    min_value=0.0,
    value=0.0
)

FoodCourt = st.sidebar.number_input(
    "Food Court",
    min_value=0.0,
    value=0.0
)

ShoppingMall = st.sidebar.number_input(
    "Shopping Mall",
    min_value=0.0,
    value=0.0
)

Spa = st.sidebar.number_input(
    "Spa",
    min_value=0.0,
    value=0.0
)

VRDeck = st.sidebar.number_input(
    "VR Deck",
    min_value=0.0,
    value=0.0
)


# =========================
# Prediction Button
# =========================

if st.sidebar.button("Predict"):

    input_data = pd.DataFrame({
        "PassengerId": [f"{PassengerGroup}_01"],
        "HomePlanet": [HomePlanet],
        "CryoSleep": [CryoSleep],
        "Cabin": [Cabin],
        "Destination": [Destination],
        "Age": [Age],
        "VIP": [VIP],
        "RoomService": [RoomService],
        "FoodCourt": [FoodCourt],
        "ShoppingMall": [ShoppingMall],
        "Spa": [Spa],
        "VRDeck": [VRDeck],
        "Name": ["Streamlit Passenger"]
    })


    st.subheader("Input Data")
    st.dataframe(input_data)


    # =========================
    # Prediction
    # =========================

    with st.spinner("Running model..."):

        X = preprocess_data(input_data, is_train=False)

        processed = preprocessor.transform(X)

        prediction = model.predict(processed)[0]

        probability = model.predict_proba(processed)[0][1]


    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ Passenger was Transported")
    else:
        st.error("❌ Passenger was NOT Transported")


    st.metric(
        label="Transport Probability",
        value=f"{probability:.2%}"
    )






st.markdown("---")

st.write("Model: Logistic Regression")
st.write("Course: Model Deployment")