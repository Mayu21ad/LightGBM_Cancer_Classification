import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def preprocess_data(data):
    data = data.drop('Patient Id', axis=1)
    le = LabelEncoder()
    encoded_level = le.fit_transform(data['Level'])
    data = data.drop('Level', axis=1)
    data['encoded_level'] = encoded_level
    return data

st.title("A LightGBM Classification Approach")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = preprocess_data(data)
    X = data[['Air Pollution', 'Gender', 'chronic Lung Disease', 'Smoking', 'Dry Cough']]
    y = data['encoded_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.write(f"Model accuracy score: {accuracy * 100:.2f}%")

    st.subheader("Testing with user input")

    air_pollution = st.number_input("Air Pollution", min_value=0, max_value=500)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x:'Male' if x == 1 else 'Female')
    chronic_lung_disease = st.number_input("Chronic Lung Disease", min_value=0, max_value=10)
    smoking = st.number_input("Smoking", min_value=0, max_value=10)
    dry_cough = st.number_input("Dry Cough", min_value=0, max_value=10)

    user_input = pd.DataFrame({
        'Air Pollution': [air_pollution],
        'Gender': [gender],
        'chronic Lung Disease': [chronic_lung_disease],
        'Smoking': [smoking],
        'Dry Cough': [dry_cough]
    })

    # Mapping function for prediction results
    def map_prediction(value):
        if value == 1:
            return "Low"
        elif value == 2:
            return "Medium"
        elif value == 0:
            return "High"

# When user clicks 'Predict' button
    if st.button("Predict"):
        prediction = clf.predict(user_input)
        prediction_label = map_prediction(prediction[0])  # Map the predicted value to a label
        st.write(f"Predicted level: {prediction_label}")


else:
    st.write("Please upload a CSV file to proceed.")
