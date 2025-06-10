# AIB25-Fake-Profile-Account-Detection

```
├── Datasets/
│   ├── Instagram_fake_profile_dataset.csv # Tabular Dataset used in Fake/bot Instagram account detection
│   ├── 
│   └── Sample_from_real_ig_accounts.csv # IG account sample collected manually (100 sample)
│
├── Notebooks/
│   ├── Baseline_for_fake_account.ipynb
│   ├── Fake_X_profile_detection.ipynb
│   └── Fake_bot_account_detection.ipynb
│
├── model/
│   └── fake_ig_model.pkl        # LightGBM model for tabular IG detection
│                                # Resnet18 model link down below
├── sample_images/               # Sample screenshots for the image classifier
│   ├── ...
│
├── app.py                       # Streamlit app for deployment
├── requirements.txt             # Python dependencies
└── README.md
```

Note:
- โมเดลที่ใช่ในโปรเจ็กต์ Fake X/IG profile detection มีขนาดไฟล์ที่ใหญ่เกินไปทำให้ต้อง upload ไว้ใน google drive แทน
  - Model trained on X screenshot: https://drive.google.com/file/d/1fJoH544qIINdmrO3KPTUb9TJQKntVZGE/view?usp=sharing
  - Model fine tuned on IG screenshot: https://drive.google.com/file/d/15jb5frBeVffjp0NNCdHg2NXWvTTSDqpV/view?usp=sharing

- Dataset for Fake X/IG profile detection project: https://huggingface.co/datasets/drveronika/x_fake_profile_detection
- Baseline for Fake X/IG profile detection project: https://huggingface.co/prithivMLmods/x-bot-profile-detection

Blog: https://medium.com/@bhanyawong/ai-builders-fake-social-media-account-detection-on-x-and-instagram-666390015a5b

Deployment: https://fake-profile-account-detection.streamlit.app/#ai-builders-2025-fake-social-media-account-profile-detection
