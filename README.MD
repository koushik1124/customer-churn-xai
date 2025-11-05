📊 Customer Churn Prediction Dashboard
🚀 AI-Powered Insights using Machine Learning + DeepSeek LLM

This project is an interactive churn prediction dashboard built with Streamlit that predicts whether a telecom customer is likely to churn and provides AI-generated business explanations and recommendations.

It integrates a trained XGBoost model for predictions, SHAP for model explainability, and DeepSeek AI for human-readable insights — combining data science with modern LLM-driven interpretation.

🧠 Key Features

✅ Customer Churn Prediction — Predicts whether a customer is likely to churn based on key telecom features.
✅ AI-Powered Explanations — Uses DeepSeek’s LLM API to generate clear, business-friendly explanations for every prediction.
✅ Interactive Streamlit UI — User-friendly dashboard for real-time predictions and insights.
✅ Explainable AI (SHAP) — Highlights which factors most impact churn risk.
✅ Reusable ML Pipeline — Includes model training, preprocessing, and saving with joblib.
✅ Environment-Safe API Integration — Loads your API key securely from a .env file.

🏗️ Tech Stack
Layer	Technology
Frontend / UI	Streamlit
Model	XGBoost
Explainability	SHAP
Data Handling	Pandas, NumPy
AI Integration	DeepSeek LLM (OpenAI-compatible)
Serialization	Joblib
Environment Management	python-dotenv
🗂️ Project Structure
customer_churn_xai/
│
├── app.py                  # Main Streamlit application
├── churn_model.pkl         # Trained XGBoost model
├── encoder.pkl             # Saved LabelEncoders (for categorical features)
├── telco_churn.csv         # Input dataset
├── model_training.ipynb    # Notebook for training & SHAP exploration
├── requirements.txt        # Python dependencies
└── .env                    # Contains your DeepSeek API key (not committed)

⚙️ Installation & Setup
1️⃣ Clone this repository
git clone https://github.com/your-username/customer-churn-xai.git
cd customer-churn-xai

2️⃣ Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # On macOS/Linux
.venv\Scripts\activate         # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your DeepSeek API key

Create a .env file in the root folder:

DEEPSEEK_API_KEY=sk-your_api_key_here


⚠️ Do not commit this file to GitHub. Add it to .gitignore.

5️⃣ Run the app
streamlit run app.py


The dashboard will launch in your browser (usually at http://localhost:8501).

🧩 How It Works
🔹 Step 1: Data Processing

The raw telco_churn.csv dataset is cleaned and preprocessed.

Categorical features are label-encoded and saved using encoder.pkl.

🔹 Step 2: Model Training

A churn classifier is trained using XGBoost in model_training.ipynb.

The trained model is serialized as churn_model.pkl.

🔹 Step 3: Dashboard Interaction

User enters customer details in the Streamlit sidebar.

The model predicts churn probability in real-time.

SHAP values explain feature importance.

🔹 Step 4: AI Explanation

DeepSeek API reads the SHAP output.

Generates a natural, human-readable summary explaining why the customer may churn and what actions to take.

🧾 Example AI Output

Prediction: Customer is likely to churn (Probability: 0.82)

🤖 AI Explanation:
The customer has a high churn risk due to a short tenure and high monthly charges.
Automatic payments and fiber-optic service slightly reduce the risk.
Offering a loyalty discount or a contract upgrade may improve retention.

🧰 Environment Variables
Variable	Description
DEEPSEEK_API_KEY	Your DeepSeek API key for AI-generated explanations
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


Install via:

pip install -r requirements.txt

🔒 Security Notes

Never commit .env files or API keys to public repositories.

Add .env to your .gitignore:

.env
__pycache__/
.venv/


If deploying publicly (e.g., Streamlit Cloud), set the API key as a secret environment variable in your dashboard settings.

📈 Future Enhancements

📄 Downloadable PDF Reports (prediction + AI summary)

📊 Batch prediction for multiple customers (CSV upload)

🧩 Model retraining with new datasets

🌐 Multi-language AI explanations

🧠 Integration with OpenAI GPT-4 or local Llama models (offline option)

🤝 Contributing

Pull requests are welcome!
If you’d like to add features or improve explainability, please fork the repo and open a PR.
