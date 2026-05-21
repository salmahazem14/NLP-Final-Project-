from language_detector import LanguageIdentifier

# 1. Initialize the module
detector = LanguageIdentifier()

# 2. Prepare data SPECIFICALLY for Arabic and English
X_train, y_train, X_valid, y_valid, X_test, y_test = detector.prepare_data(
    languages=["ar", "en"]
)

# 3. Train the dedicated model
detector.train(X_train, y_train)

# 4. Evaluate it
detector.evaluate(X_test, y_test, split_name="Test Set")

# 5. Save the artifact for downstream deployment
detector.save_model("ar_en_language_classifier.pkl")