# import streamlit as st
# import pandas as pd
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# def preprocess_data(data):
#     data = data.drop('Patient Id', axis=1)
#     le = LabelEncoder()
#     encoded_level = le.fit_transform(data['Level'])
#     data = data.drop('Level', axis=1)
#     data['encoded_level'] = encoded_level
#     return data

# st.title("A LightGBM Classification Approach")
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     data = preprocess_data(data)
#     X = data[['Air Pollution', 'Gender', 'chronic Lung Disease', 'Smoking', 'Dry Cough']]
#     y = data['encoded_level']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#     clf = lgb.LGBMClassifier()
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     st.write(f"Model accuracy score: {accuracy * 100:.2f}%")

#     st.subheader("Testing with user input")

#     air_pollution = st.number_input("Air Pollution", min_value=0, max_value=500)
#     gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x:'Male' if x == 1 else 'Female')
#     chronic_lung_disease = st.number_input("Chronic Lung Disease", min_value=0, max_value=10)
#     smoking = st.number_input("Smoking", min_value=0, max_value=10)
#     dry_cough = st.number_input("Dry Cough", min_value=0, max_value=10)

#     user_input = pd.DataFrame({
#         'Air Pollution': [air_pollution],
#         'Gender': [gender],
#         'chronic Lung Disease': [chronic_lung_disease],
#         'Smoking': [smoking],
#         'Dry Cough': [dry_cough]
#     })

#     # Mapping function for prediction results
#     def map_prediction(value):
#         if value == 1:
#             return "Low"
#         elif value == 2:
#             return "Medium"
#         elif value == 0:
#             return "High"

# # When user clicks 'Predict' button
#     if st.button("Predict"):
#         prediction = clf.predict(user_input)
#         prediction_label = map_prediction(prediction[0])  # Map the predicted value to a label
#         st.write(f"Predicted level: {prediction_label}")


# else:
#     st.write("Please upload a CSV file to proceed.")

import streamlit as st
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Create tabs for visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Air Pollution", "Gender", "Chronic Lung Disease", "Smoking", "Dry Cough"])

    # Tab for Air Pollution
    with tab1:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['encoded_level'], y=data['Air Pollution'], ax=ax)
        ax.set_title('Air Pollution vs Encoded Level')
        ax.set_xlabel('Encoded Level')
        ax.set_ylabel('Air Pollution')
        st.pyplot(fig)
        st.write("This boxplot shows the distribution of air pollution levels across different encoded levels of the target variable. It helps to understand if higher pollution is associated with a certain level.")

    # Tab for Gender
    with tab2:
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', hue='encoded_level', data=data, ax=ax)
        ax.set_title('Gender vs Encoded Level')
        ax.set_xlabel('Gender (0: Female, 1: Male)')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        st.write("This countplot shows the number of male and female patients for each encoded level of the target variable. It helps to assess the gender distribution in different levels.")

    # Tab for Chronic Lung Disease
    with tab3:
        fig, ax = plt.subplots()
        sns.violinplot(x=data['encoded_level'], y=data['chronic Lung Disease'], ax=ax)
        ax.set_title('Chronic Lung Disease vs Encoded Level')
        ax.set_xlabel('Encoded Level')
        ax.set_ylabel('Chronic Lung Disease')
        st.pyplot(fig)
        st.write("This violin plot shows the density distribution of chronic lung disease severity across different levels of the target variable. It provides insight into how chronic lung disease relates to the target.")

    # Tab for Smoking
    with tab4:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['encoded_level'], y=data['Smoking'], ax=ax)
        ax.set_title('Smoking vs Encoded Level')
        ax.set_xlabel('Encoded Level')
        ax.set_ylabel('Smoking')
        st.pyplot(fig)
        st.write("This boxplot visualizes the relationship between smoking habits and the encoded level. It helps to examine if smoking is more prevalent in certain levels of the target.")

    # Tab for Dry Cough
    with tab5:
        fig, ax = plt.subplots()
        sns.boxplot(x=data['encoded_level'], y=data['Dry Cough'], ax=ax)
        ax.set_title('Dry Cough vs Encoded Level')
        ax.set_xlabel('Encoded Level')
        ax.set_ylabel('Dry Cough')
        st.pyplot(fig)
        st.write("This boxplot illustrates how dry cough symptoms vary across the different encoded levels of the target variable. It can indicate if dry cough is a distinguishing factor for the severity level.")

    st.subheader("Testing with user input")

    air_pollution = st.number_input("Air Pollution", min_value=0, max_value=500)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
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

    def map_prediction(value):
        if value == 1:
            return "Low"
        elif value == 2:
            return "Medium"
        elif value == 0:
            return "High"

    if st.button("Predict"):
        prediction = clf.predict(user_input)
        prediction_label = map_prediction(prediction[0])  
        st.write(f"Predicted level: {prediction_label}")

else:
    st.write("Please upload a CSV file to proceed.")

