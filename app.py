import streamlit as st
import pandas as pd
import sqlite3
import bcrypt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Database functions
def create_database():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        st.success("User  registered successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    conn.close()

def check_user(username, password):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode('utf-8'), result[0]):
        return True
    return False

# Load and preprocess data
def load_and_preprocess_data(filepath):
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file {filepath} was not found.")
        return None, None

    selected_columns = ['Price range', 'Aggregate rating', 'Votes',
                        'Has Table booking', 'Has Online delivery', 'Rating text']
    data = data[selected_columns]

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Encode categorical columns
    label_encoders = {}
    categorical_columns = ['Has Table booking', 'Has Online delivery', 'Rating text']
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders

# Train the model
def train_model(data):
    X = data[['Price range', 'Aggregate rating', 'Votes', 'Has Table booking', 'Has Online delivery']]
    y = data['Rating text']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    return model, accuracy, conf_matrix

# Create visualizations
def create_visualizations(data, model, conf_matrix):
    visualization_option = st.selectbox("Select Visualization", ["Feature Importance", "Class Distribution", "Confusion Matrix Heatmap", "Correlation HeatMap", "Pair Plot", "Feature Pair Relation", "Prediction Distribution"])

    if visualization_option == "Feature Importance":
        st.subheader('Feature Importance')
        plt.figure(figsize=(10, 6))
        feature_importance = model.feature_importances_
        features = ['Price range', 'Aggregate rating', 'Votes', 'Has Table booking', 'Has Online delivery']
        sns.barplot(x=feature_importance, y=features, palette='viridis')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.grid(axis='x')
        st.pyplot(plt)

    elif visualization_option == "Class Distribution":
        st.subheader('Class Distribution')
        plt.figure(figsize=(10, 6))
        class_counts = data['Rating text'].value_counts()
        class_counts.plot(kind='bar', color='skyblue')
        for i, count in enumerate(class_counts):
            plt.text(i, count + 5, f'{count} ({count / class_counts.sum() * 100:.1f}%)', ha='center')
        plt.title('Class Distribution')
        plt.xlabel('Rating Text')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    elif visualization_option == "Confusion Matrix Heatmap":
        st.subheader('Confusion Matrix Heatmap')
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Poor', 'Average', 'Good', 'Very Good', 'Excellent'], yticklabels=['Poor', 'Average', 'Good', 'Very Good', 'Excellent'])
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)

    elif visualization_option == "Pair Plot":
        sns.pairplot(data[['Price range', 'Aggregate rating', 'Votes', 'Rating text']], hue='Rating text', palette='viridis')
        plt.title('Pair Plot')
        st.pyplot(plt)

    elif visualization_option == "Correlation HeatMap":
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)

    elif visualization_option == "Feature Pair Relation":
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data, x='Aggregate rating', y='Votes', hue='Rating text', palette='viridis')
        plt.title('Feature Pair Relationship')
        plt.xlabel('Aggregate Rating')
        plt.ylabel('Votes')
        plt.legend(title='Rating Text')
        st.pyplot(plt)

    elif visualization_option == "Prediction Distribution":
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Rating text'], bins=5, kde=True, color='lightgreen')
        plt.title('Prediction Distribution')
        plt.xlabel('Rating Text')
        plt.ylabel('Frequency')
        st.pyplot(plt)

# Get user input for prediction
def get_user_input(label_encoders):
    price_range = st.number_input("Price range (1-4):", min_value=1, max_value=4)
    aggregate_rating = st.number_input("Aggregate rating (e.g., 4.5):", format="%.1f")
    votes = st.number_input("Number of votes (e.g., 150):", min_value=0)
    has_table_booking = st.selectbox("Has table booking:", ['Yes', 'No'])
    has_online_delivery = st.selectbox("Has online delivery:", ['Yes', 'No'])

    # Encode categorical inputs
    has_table_booking = label_encoders['Has Table booking'].transform([has_table_booking])[0]
    has_online_delivery = label_encoders['Has Online delivery'].transform([has_online_delivery])[0]

    return [[price_range, aggregate_rating, votes, has_table_booking, has_online_delivery]]

# Predict user input
def predict_user_input(model, label_encoders):
    input_data = get_user_input(label_encoders)
    if st.button("Predict"):
        prediction = model.predict(input_data)
        rating_text = label_encoders['Rating text'].inverse_transform(prediction)[0]
        st.success(f"Predicted Rating: {rating_text}")

# Main function
def main():
    create_database()  # Create the database if it doesn't exist

    st.title("Restaurant Rating Prediction App")

    # User authentication
    st.sidebar.title("User  Authentication")
    option = st.sidebar.selectbox("Select an option", ["Login", "Register"])

    if option == "Register":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.button("Register"):
            create_user(username, password)

    elif option == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.button("Login"):
            if check_user(username, password):
                st.session_state['logged_in'] = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password.")

    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        st.warning("Please log in to access the app.")
        return

    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "Visualizations", "Prediction", "Model Performance", "Source Code", "Logout"])

    # Load and preprocess data
    filepath = "Dataset.csv"  # Replace with the correct path to your dataset
    data, label_encoders = load_and_preprocess_data(filepath)
    if data is None:
        return

    model, accuracy, conf_matrix = train_model(data)

    if page == "Home":
        st.write("Welcome to the Restaurant Rating Prediction App!")

    elif page == "Visualizations":
        create_visualizations(data, model, conf_matrix)

    elif page == "Prediction":
        st.subheader("Make a Prediction")
        predict_user_input(model, label_encoders)

    elif page == "Model Performance":
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

    elif page == "Source Code":
        st.write("You can find the source code for this application on GitHub.")
        st.markdown("[View Source Code](https://github.com/Ronak1231/cognifyz-Technologies_Task-3.git)", unsafe_allow_html=True)

    elif page == "Logout":
        st.session_state['logged_in'] = False
        st.success("Logged out successfully!")

if __name__ == "__main__":
    main()
