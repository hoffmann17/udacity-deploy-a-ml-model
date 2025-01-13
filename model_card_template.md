# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier designed to predict whether an individual’s income exceeds $50,000 based on census data. The model includes hyperparameter tuning via GridSearchCV, which optimizes for the F1 score using a 5-fold cross-validation process.

### Key Components:
- **Algorithm:** Random Forest Classifier
- **Hyperparameters Tuned:**
  - Number of estimators (`n_estimators`): [50, 100, 200]
  - Maximum depth (`max_depth`): [10, 20, None]
  - Minimum samples split (`min_samples_split`): [2, 5, 10]
- **Random State:** 42 (for reproducibility)
- **Frameworks Used:**
  - `scikit-learn` for model training and evaluation
  - `pandas` and `numpy` for data manipulation
  - `joblib` for saving trained models

## Intended Use

This model is intended for educational and research purposes to demonstrate the end-to-end process of building, training, and evaluating a machine learning model. It is also useful for practical applications in income classification tasks based on demographic and occupational features.

### Notable Use Cases:
- Predicting income level for socio-economic studies
- Generating insights on feature importance related to income

### Limitations:
The model is not suitable for:
- High-stakes decisions affecting individuals’ lives, such as loan approvals or hiring.
- Situations requiring explainability beyond feature importance scores.

## Training Data

The training dataset is derived from the "Adult Census Income" dataset, containing demographic, educational, and occupational data. The dataset includes categorical and numerical features, such as:
- **Categorical Features:** workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Label:** Binary classification of income level (`salary` > $50,000 or <= $50,000)

The dataset was split into training (80%) and test (20%) sets.

## Evaluation Data

The evaluation dataset is the test split from the original dataset. It follows the same preprocessing steps as the training data, ensuring compatibility with the trained model. Performance metrics are computed on this test data.

## Metrics

The model’s performance is evaluated using the following metrics:
- **Precision:** Measures the proportion of true positive predictions among all positive predictions.
- **Recall:** Measures the proportion of true positives identified among all actual positives.
- **F1 Score (F-beta):** A harmonic mean of precision and recall.

### Model Performance:
- **Precision:** Achieved on the test set.
- **Recall:** Achieved on the test set.
- **F1 Score:** Achieved on the test set.

Additionally, performance is computed on slices of the data based on unique values of categorical features. The results are saved in a CSV file (`slice_output.csv`) for further analysis.

## Ethical Considerations

- **Bias in Data:** The model inherits any biases present in the training data. For example, socio-economic and demographic features may reflect systemic biases.
- **Fairness:** The performance across different demographic slices should be carefully evaluated to ensure fairness.
- **Privacy:** Data used for training and evaluation should comply with privacy regulations such as GDPR and CCPA. Personally identifiable information (PII) is excluded from the dataset.

## Caveats and Recommendations

- This model is a baseline implementation. For production use, additional steps like model interpretability, fairness audits, and robustness testing should be conducted.
- Data quality significantly impacts model performance; hence, ensure the dataset is up-to-date and accurately labeled.
- Monitor performance on real-world data and retrain periodically to account for changes in underlying data distributions.

