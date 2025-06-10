# AIB25-Fake-Profile-Account-Detection

├── Datasets/
│   ├── Instagram_fake_profile_dataset.csv # Tabular dataset from kaggle
│   ├── Sample_from_real_ig_accounts.csv # manually collected dataset from real IG account
│
├── Notebooks/
│   ├── Baseline_for_fake_account.ipynb
│   ├── Fake_X_profile_detection.ipynb
│   ├── Fake_bot_account_detection.ipynb
│
├── model/
│   └── fake_ig_model.pkl              # LightGBM model for tabular Instagram detection
│
├── sample_images/                     # Contains sample screenshots for testing the image classifier
├── app.py                             # Streamlit app for model deployment
├── requirements.txt                   # Python dependencies
└── README.md

🧠 Model Files
Model	Type	Location
fake_ig_model.pkl	LightGBM	Stored in /model/
best_resnet18_model.pth	ResNet-18	Google Drive 🔗
best_resnet18_model_ig.pth	ResNet-18	Google Drive 🔗

⚠️ The two .pth files are too large to be stored directly in the repository. They are automatically downloaded when you run the app.

📦 Dataset Info
The dataset for X (Twitter) profile screenshot classification is too large for this repository.

🔗 Dataset: x_fake_profile_detection on HuggingFace

🔗 Baseline Model: SigLip2 on HuggingFace
