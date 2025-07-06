# job-domain-project
# Job Category Predictor Web App

This project uses machine learning to predict a person's likely job domain based on their resume or list of skills. It uses a **TF-IDF + Logistic Regression pipeline**, trained on a labeled dataset of resumes.

---

## 🚀 Features
- Input skills or resume text
- Predicts job domain (e.g., Data Science, HR, Testing, etc.)
- Shows confidence scores for each category
- Clean dark-themed Streamlit UI

---

## 📁 Project Structure
```
├── app.py                # Streamlit web app
├── train_model.py        # Script to train and save model
├── UpdatedResumeDataSet.csv  # Input dataset
├── model.pkl             # Trained ML model (saved after training)
├── label_encoder.pkl     # Encoded job categories
├── requirements.txt      # Python dependencies
```

---

## ▶️ How to Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/job-category-predictor.git
cd job-category-predictor
```

### 2. Set Up Environment
```bash
pip install -r requirements.txt
```

### 3. Train the Model (optional)
If you haven’t yet:
```bash
python train_model.py
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud
1. Push all files to a **public GitHub repo**
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and select the repo
4. Choose `app.py` as the main file
5. Hit **Deploy** 🎉

---

## ✍️ Example Input
**Skills Input:**
```
Python, Machine Learning, Pandas, SQL, NLP
```

---

## 📜 License
MIT License

---

## 🤖 Built With
- Python
- Scikit-learn
- Streamlit
- Pandas
- Joblib
