📊 Customer Churn Prediction Dashboard
🚀 AI-Powered Insights using Machine Learning + DeepSeek LLM

This project is an interactive churn prediction dashboard built with Streamlit that predicts whether a telecom customer is likely to churn and provides AI-generated business explanations and recommendations.

It integrates a trained XGBoost model for predictions, SHAP for explainability, and DeepSeek AI for natural-language insights — combining data science with modern LLM-driven interpretation.

🧠 Key Features

✅ Customer Churn Prediction — Predicts whether a customer is likely to churn based on telecom usage data.

🤖 AI-Powered Explanations — Uses DeepSeek’s LLM API to generate clear, business-friendly explanations for every prediction.

🧩 Interactive Streamlit UI — Real-time predictions through an intuitive dashboard.

📈 Explainable AI (SHAP) — Highlights which factors most influence churn risk.

🔁 Reusable ML Pipeline — Includes model training, preprocessing, and serialization with joblib.

🔐 Secure API Integration — Loads your DeepSeek API key safely from a .env file.

🧰 Tech Stack
Layer	Technology
Frontend / UI	Streamlit
Model	XGBoost
Explainability	SHAP
Data Handling	Pandas, NumPy
AI Integration	DeepSeek LLM (OpenAI-compatible)
Serialization	Joblib
Environment Management	python-dotenv




customer_churn_xai/
│

├── app.py                # 🎯 Main Streamlit application
│

├── churn_model.pkl       # 💾 Trained XGBoost model
├── encoder.pkl           # 🧩 Saved LabelEncoders for categorical features
│

├── telco_churn.csv       # 📊 Input telecom dataset
├── model_training.ipynb  # 📘 Notebook for model training & SHAP exploration
│

├── requirements.txt      # 📦 Python dependencies
└── .env                  # 🔐 Contains DeepSeek API key (not committed)


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/your-username/customer-churn-xai.git
cd customer-churn-xai

2️⃣ Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your DeepSeek API key

Create a .env file in the root folder:

DEEPSEEK_API_KEY=sk-your_api_key_here


(Do not include quotes or spaces)

5️⃣ Run the Streamlit app
streamlit run app.py


Then open your browser at http://localhost:8501
.

🧩 How It Works

Data Preprocessing
Cleans and encodes telecom data using LabelEncoders saved as encoder.pkl.

Model Training
Uses XGBoost to predict churn and saves the model as churn_model.pkl.

Dashboard Interaction
Users enter customer details on the sidebar; the model predicts churn probability.

Explainability
SHAP identifies top risk and retention factors affecting the prediction.

AI Explanation
DeepSeek LLM generates a natural-language summary and actionable recommendations.

💬 Example Output

Prediction: Customer is likely to churn (Probability: 0.81)

🤖 AI Explanation:
The customer shows a high churn risk due to short tenure and high monthly charges.
Automatic payment and fiber-optic connection slightly reduce risk.
Offering a loyalty reward or a contract upgrade may help retain this customer.

🧾 Environment Variables
Variable	Description
DEEPSEEK_API_KEY	Your DeepSeek API key for AI explanations
🧪 Requirements
streamlit
pandas
numpy
matplotlib
shap
joblib
xgboost
scikit-learn
openai
python-dotenv


Install with:

pip install -r requirements.txt

🔒 Security Notes

Never commit your .env file or API keys.

Add .env to .gitignore to keep secrets private.

For Streamlit Cloud or deployment, set API keys as environment variables in your hosting settings.

🌟 Future Enhancements

📄 PDF export of prediction + AI explanation

📊 Batch churn prediction via CSV upload

🧠 Offline LLM integration (local inference)

🌐 Multi-language AI summaries

🤝 Contributing

Pull requests are welcome!
To propose major changes, open an issue first to discuss what you’d like to modify.
