# Train - Test Split
df_train, df_test = train_test_split(
    df_hf_processed,
    test_size=0.2,
    random_state=42,
    stratify=df_hf_processed["target"]
)
 
y_train_before = df_train["target"].value_counts().sort_index()
min_binary_size = df_train["target"].value_counts().min()
 
# Undersampling
df_train_balanced = (
    df_train
    .groupby("target", group_keys=False)
    .apply(lambda x: x.sample(min_binary_size, random_state=42))
    .reset_index(drop=True)
)
 
y_train_after = df_train_balanced["target"].value_counts().sort_index()
 
# Feature Engineering
df_train_feat = engineer_features(df_train_balanced)
df_test_feat = engineer_features(df_test)
 
y_train = df_train_feat["target"].values
y_test = df_test_feat["target"].values
 
absa_score_cols = ["absa_skin_aspect", "absa_hair_aspect", "absa_product_aspect"]
absa_flag_cols  = ["absa_skin_available", "absa_hair_available", "absa_product_available"]
other_numeric   = [
    "review_length", "price_usd",
    "is_vegan", "is_cruelty_free", "is_clean_beauty",
    "has_hyaluronic", "has_retinol", "has_niacinamide",
    "has_acids", "has_fragrance", "user_product_alignment"
]
categorical_cols = ["skin_type", "skin_tone"]
text_col = "review_text"
 
preprocessor = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=1500, stop_words="english", dtype=np.float32), text_col),
        ("cat", Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))
        ]), categorical_cols),
      
        ("absa_scores", Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value=0.0)), 
            ('scaler', StandardScaler())
        ]), absa_score_cols),
        ("absa_flags", SimpleImputer(strategy="constant", fill_value=0), absa_flag_cols), 
        ("num", Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('scaler', StandardScaler())
        ]), other_numeric)
    ],
    remainder="drop"
)
 
print("\n--- Model Training Process Started (GridSearchCV & Baseline Integrated) ---")
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
 
model_results = {}
 
# Baseline Model
print(">> Training Majority Baseline Model...")
dummy_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", DummyClassifier(strategy="most_frequent"))
])
dummy_cv_scores = cross_val_score(dummy_pipe, df_train_feat, y_train, cv=cv_strategy, scoring="f1_macro", n_jobs=-1)
 
dummy_pipe.fit(df_train_feat, y_train)
dummy_preds = dummy_pipe.predict(df_test_feat)
dummy_probs = dummy_pipe.predict_proba(df_test_feat)[:, 1]
 
try:
    dummy_auc = roc_auc_score(y_test, dummy_probs)
except ValueError:
    dummy_auc = 0.5
 
model_results["Majority Baseline"] = {
    "b_acc": balanced_accuracy_score(y_test, dummy_preds),
    "f1": f1_score(y_test, dummy_preds, average="macro", zero_division=0),
    "roc_auc": dummy_auc,
    "cv_mean": dummy_cv_scores.mean(),
    "cv_std": dummy_cv_scores.std(),
    "preds": dummy_preds,
    "probs": dummy_probs
}
 
param_grids = {
    "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
    "Random Forest": {"model__max_depth": [5, 10, 15], "model__n_estimators": [50, 100]},
    "SVM (Linear)": {"model__estimator__C": [0.1, 0.5, 1.0]}
}
 
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "SVM (Linear)": CalibratedClassifierCV(LinearSVC(dual=False, random_state=42), cv=3)
}
 
for name, model in base_models.items():
    print(f">> Running hyperparameter tuning for {name}...")
 
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
 
    grid_search = GridSearchCV(
        pipe,
        param_grids[name],
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1
    )
 
    grid_search.fit(df_train_feat, y_train)
    best_model = grid_search.best_estimator_
 
   
    cv_mean = grid_search.best_score_
    cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
 
    preds = best_model.predict(df_test_feat)
    probs = best_model.predict_proba(df_test_feat)[:, 1]
 
    model_results[name] = {
        "b_acc": balanced_accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, average="macro"),
        "roc_auc": roc_auc_score(y_test, probs),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "preds": preds,
        "probs": probs
    }
    print(f"   Best Params: {grid_search.best_params_} | Unbiased CV F1: {cv_mean:.4f}")
 
# Joblib Serialization
joblib.dump({
    "y_train_before": y_train_before,
    "y_train_after": y_train_after,
    "model_results": model_results,
    "y_test": y_test,
    "absa_missing_series": absa_missing_series,
    "absa_flag_distributions": absa_flag_distributions
}, "data/model_outputs.joblib")
 
print("\n✅ CELL 2 SUCCESSFULLY COMPLETED AND STORED VIA JOBLIB!")
