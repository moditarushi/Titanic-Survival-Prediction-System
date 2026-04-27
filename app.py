import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
model = joblib.load("titanic_model.pkl")

# UI
st.markdown("<h1 style='text-align:center;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
fare = st.slider("Fare", 0, 500, 50)

sex = 0 if sex == "Male" else 1

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex, age, fare]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0]

    st.write("Model used: Logistic Regression")

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Not Survived")

    survival_prob = prob[1]

    # Show Inputs
    st.subheader("Your Inputs")
    st.write({
       "Pclass": pclass,
       "Sex": "Male" if sex == 0 else "Female",
       "Age": age,
       "Fare": fare
  })

    # Prediction Message
    if survival_prob > 0.7:
        st.success(f"🟢 High chance of survival ({prob[1]*100:.2f}%)")
    elif survival_prob > 0.4:
        st.warning(f"🟡 Moderate chance of survival ({prob[1]*100:.2f}%)")
    else:
        st.error(f"🔴 Low chance of survival ({prob[1]*100:.2f}%)")

    st.caption("Prediction based on Logistic Regression trained on Titanic dataset")

    # Probability Graph
    st.subheader("📊 Survival Probability")

    fig, ax = plt.subplots()
    labels = ["Not Survived", "Survived"]
    values = [prob[0], prob[1]]

    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Survival Probability")
    ax.grid(True)

    st.pyplot(fig)

    # Insights
    st.subheader("📌 Insights")

    st.write("• Female passengers had higher survival chances")
    st.write("• Higher class passengers survived more")
    st.write("• Higher fare often increases survival probability")

    st.markdown("---")
    st.write("Made by Tarushi Modi | Machine Learning Project")

  
    st.markdown("""
    ### 📌 About Project
    This model predicts survival using Logistic Regression.

    - Features: Pclass, Sex, Age, Fare  
    - Dataset: Titanic  
    - Accuracy: {:.2f}%
    """)
