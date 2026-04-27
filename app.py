import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("titanic.csv")

# Data Preprocessing
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert sex to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Features and Targets
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)
model = LogisticRegression()
model.fit(X_train, y_train)

model1 = LogisticRegression()
model2 = DecisionTreeClassifier()

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

acc1 = model1.score(X_test, y_test)
acc2 = model2.score(X_test, y_test)

accuracy = model.score(X_test, y_test)

# UI
st.markdown("<h1 style='text-align:center;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)

st.write(f"### Model Accuracy: {accuracy*100:.2f}%")

# Model comparison
st.subheader("📊 Model Comparison")
st.write(f"Logistic Regression: {acc1*100:.2f}%")
st.write(f"Decision Tree: {acc2*100:.2f}%")

st.bar_chart({
    "Logistic Regression": acc1,
    "Decision Tree": acc2
})

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

    # Confusion Matrix

    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y_test, model.predict(X_test))

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax2)

    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    st.pyplot(fig2)

    # Feature Importance
    st.subheader("📊 Feature Importance")

    importance = model.coef_[0]
    features = X.columns

    fig3, ax3 = plt.subplots(figsize =(6,4))
    ax3.barh(features, importance, color = "skyblue")
    ax3.set_title("Feature Importance")

    st.pyplot(fig3)
    

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
    """.format(accuracy*100))
