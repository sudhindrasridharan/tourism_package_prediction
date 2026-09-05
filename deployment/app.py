
import streamlit as st
import pandas as pd
import pickle
import os


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_PATH,
    "deployment",
    "best_model.pkl"
)


@st.cache_resource
def load_model():

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    return model


model = load_model()


# ============================================================
# 3. APPLICATION TITLE
# ============================================================

st.title("✈️ Tourism Package Purchase Prediction")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)


# ============================================================
# 4. GET CUSTOMER INPUTS
# ============================================================

st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    type_of_contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    number_of_person_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )


with col2:

    preferred_property_star = st.selectbox(
        "Preferred Property Star",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced"]
    )

    number_of_trips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=50,
        value=3
    )

    passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    own_car = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    number_of_children = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )


with col3:

    designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=25000
    )

    pitch_satisfaction_score = st.slider(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3
    )

    product_pitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Deluxe",
            "Standard",
            "Super Deluxe",
            "King"
        ]
    )

    number_of_followups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=20,
        value=3
    )

    duration_of_pitch = st.number_input(
        "Duration of Pitch",
        min_value=0,
        max_value=60,
        value=10
    )


# ============================================================
# 5. CREATE DATAFRAME FROM USER INPUT
# ============================================================

input_data = {
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch
}


input_df = pd.DataFrame(
    [input_data]
)


# ============================================================
# 6. DISPLAY INPUT DATA
# ============================================================

st.header("Customer Input")

st.dataframe(
    input_df,
    use_container_width=True
)


# ============================================================
# 7. MAKE PREDICTION
# ============================================================

if st.button(
    "Predict Package Purchase",
    type="primary"
):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(
        input_df
    )[0][1]


    # ========================================================
    # 8. DISPLAY RESULT
    # ========================================================

    st.header("Prediction Result")

    if prediction == 1:

        st.success(
            "The customer is likely to purchase "
            "the Wellness Tourism Package."
        )

    else:

        st.warning(
            "The customer is unlikely to purchase "
            "the Wellness Tourism Package."
        )


    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )
