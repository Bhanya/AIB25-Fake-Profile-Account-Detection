# AIB25-Fake-Profile-Account-Detection

├── Datasets/
│   ├── Instagram_fake_profile_dataset.csv
│   ├── Sample_from_real_ig_accounts.csv
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
