# 🍽️ Restaurant Rating Prediction - End-to-End ML App

This project implements an **end-to-end machine learning system** for restaurant rating predictions using **Streamlit** for the user interface and **Scikit-Learn** for predictive modeling. It helps analyze restaurant data and make insightful predictions.

---

## ✅ Features

- **User Authentication**: Register and login functionality using **SQLite** and **bcrypt**.
- **Data Preprocessing**: Cleans missing values and selects key features.
- **Machine Learning Models**: Uses **RandomForestClassifier** for predictions.
- **Web Application**: Built using **Streamlit** for user interaction.
- **Performance Evaluation**: Displays **accuracy and confusion matrix**.
- **Data Visualization**: Generates multiple plots using **Matplotlib & Seaborn**.

---

## 📜 Prerequisites

Ensure the following are installed:

1. **Python 3.8 or above**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Required Libraries** (Install from `requirements.txt`):
   ```sh
   pip install -r requirements.txt
   ```
3. **Dataset**: Ensure `Dataset.csv` is in the project directory.

---

## 🛠 Setup Instructions

### 1. Clone the Repository

```sh
git clone https://github.com/Ronak1231/Machine_Learning_Restaurant_Task-3.git
cd Machine_Learning_Restaurant_Task-3
```

### 2. Install Dependencies

```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```sh
streamlit run app.py
```

---

## 🗃️ File Structure

```
Machine_Learning_Restaurant_Task-2/

├── Dataset.csv                     # Dataset file
|
├── Trail
│   ├── Restaurant_Task-3
|
├── app.py
├── requirements.txt
├── LICENSE
├── README.md                                 # Project documentation
```

---

## 🤷 How It Works

### Backend Flow

1. **User Authentication**: Users can register and log in.
2. **Data Loading**: Reads and processes dataset, handling missing values.
3. **Feature Engineering**: Identifies key attributes relevant for prediction.
4. **Model Training**: Implements **Random Forest Classifier**.
5. **Model Evaluation**: Uses accuracy and confusion matrix for performance measurement.
6. **Prediction**: Users input restaurant details, and the model predicts a rating.

### Frontend Flow

- Users **input restaurant details** via **Streamlit UI**.
- The system processes the input and **visualizes data trends**.
- The model predicts and returns the result.

---

## 📊 Visualizations

The app provides various visualizations:

- **Feature Importance**: Displays key features affecting predictions.
- **Class Distribution**: Shows distribution of rating categories.
- **Confusion Matrix Heatmap**: Evaluates model performance.
- **Correlation Heatmap**: Displays relationships between features.
- **Pair Plot**: Shows scatterplots for different features.

---

## 🤖 Technologies Used

- **Python**: Programming language for data processing and ML.
- **Streamlit**: Web-based application interface.
- **Scikit-Learn**: Machine learning models and preprocessing.
- **Matplotlib & Seaborn**: Data visualization tools.
- **SQLite & bcrypt**: User authentication system.

---

## 🔜 Future Improvements

1. Add real-time restaurant data integration.
2. Implement additional machine learning models for better accuracy.
3. Deploy the app on **AWS, Google Cloud, or Heroku**.

---

## 🤝 Acknowledgments

- [Scikit-Learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Matplotlib](https://matplotlib.org/)

---

## ✍️ Author  
[Ronak Bansal](https://github.com/Ronak1231)

---

## 🙌 Contributing  
Feel free to fork this repository, improve it, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🐛 Troubleshooting  
If you encounter issues, create an issue in this repository.

---

## 📧 Contact  
For inquiries or support, contact [ronakbansal12345@gmail.com].
