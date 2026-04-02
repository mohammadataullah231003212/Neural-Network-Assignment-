# Bank Term Deposit Prediction

A Logistic Regression Model to predict if a bank customer will subscribe to a term deposit or not.

## Dataset
- File: `bank-full.csv`
- 45,211 rows and 17 columns
- Target variable: `y` (yes/no → did the customer subscribe?)


## Methodology (Step by Step)

1. Imported all required libraries
2. Loaded the dataset using pandas with `;` as delimiter
3. Explored the data to understand structure and class distribution
4. Detected anomalies like `-1` in `pdays` and `unknown` in categorical columns
5. Replaced those anomalies with `NaN` to treat them as missing values
6. Rechecked the data to confirm cleaning worked
7. Filled missing values — used most frequent value for some columns, `"none"` for others
8. Clipped outliers in numerical columns like `balance`, `duration`, `campaign`
9. Encoded target variable (`yes/no` → `1/0`) and applied one-hot encoding to categorical features
10. Split data into features (X) and target (y)
11. Did an 80/20 train-test split
12. Scaled features using StandardScaler
13. Trained a Logistic Regression model with `class_weight='balanced'`
14. Evaluated the model using accuracy, precision, recall, F1, and confusion matrix
15. Applied Grid Search for hyperparameter tuning
16. Re-evaluated the model after tuning and compared results

## Libraries Used

- **pandas** — to load and manipulate the dataset
- **numpy** — to handle numerical operations and NaN replacements
- **matplotlib** — to plot charts and visualize data
- **seaborn** — to make nicer-looking visualizations
- **sklearn** — to build, train, tune, and evaluate the machine learning model

## Approach

- Cleaned and preprocessed the raw data before doing anything else
- Handled class imbalance by using `class_weight='balanced'` in the model
- Used one-hot encoding to convert text categories into numbers
- Applied feature scaling so all columns are on the same scale
- Tuned the model using Grid Search to find better hyperparameters


## Results

### Before Tuning
| Metric | Score |
|--------|-------|
| Accuracy | 0.84 |
| Precision | 0.42 |
| Recall | 0.81 |
| F1 Score | 0.55 |

### After Tuning
| Metric | Score |
|--------|-------|
| Accuracy | 0.90 |
| Precision | 0.65 |
| Recall | 0.38 |
| F1 Score | 0.48 |

**Comment:** After tuning, accuracy and precision went up but recall dropped. So there's a trade-off — the model became more careful but missed more actual subscribers. Depending on the goal, the pre-tuning model might actually be more useful if catching subscribers is the priority.
