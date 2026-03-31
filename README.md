# Problem Statement:
Prediction of in a bank whether a customer will subscribe to a term deposit based on their banking behavior.
## Dataset: 
For this assignment I am using a dataset called bank-full.csv, provided by the instructor. It contains 45,211 data points and 17 attributes. The target variable is y, which indicates whether a customer subscribed to a term deposit (yes or no)."
# Mothodology:
## Step1: Importing Libraries:
In this step, I imported all the necessary Python librearies required for the data preprocessing, model building, training and evaluation. 
here are the libraried I used: pandas, numpy, matplotlib, seaborn, sklearn.
## Step 2: Load the dataset:
In this step I loaded the given data set in the colab notebookusing pandas and specified the correct delimiter (;) since the dataset is not comma-separated. 
## Step 3: Data Understanding
I performed exploratory data analysis to understand the dataset better. I used df.info() to check data types and missing values, and df.describe() to view statistical summaries of numerical features. I also analyzed the distribution of the target variable using counts and visualization. This helped me identify class imbalance and understand the overall data structure.
## Step 4: Anomaly Detection
In this portion of code I checked for anomalies in both numerical and categorical features. I counted zero values in numerical columns and examined the pdays column for -1, which indicates missing data. I also checked categorical columns for "unknown" values. This step helped me identify inconsistencies that could affect model performance.
## Step 5: Handling Anomalies
Here to handle anomalies, I replaced -1 values in the pdays column with NaN. I also replaced "unknown" values in categorical features with NaN. This allowed me to treat all anomalies consistently as missing data.
## Step 6: Rechecking Data State
After cleaning the data, in this part I rechecked the dataset to verify the changes. I examined missing value counts and summary statistics again to ensure that anomalies were properly handled before proceeding further.
## Step 7: Handling Missing Values
Here in this portion I handled missing values based on feature types. For categorical columns such as job and education, I filled missing values with the most frequent value. For contact and poutcome, I replaced missing values with "none" to indicate the absence of information. This ensured that the dataset had no missing values.
## Step 8: Outlier Treatment
In this step I addressed outliers by applying clipping to numerical features. I limited extreme values in columns such as balance, campaign, previous, and duration to reasonable ranges. This helped reduce the impact of extreme values on the model.
## Step 9: Handling Outlier
To handle outliers, I applied a clipping technique to the numerical features. I constrained extreme values in variables such as balance, campaign, previous, and duration within acceptable limits. This approach minimized the influence of outliers on the model’s performance.
## Step 10: Target Mapping and Encoding
The target variable is first converted from categorical labels "Yes and No" into binary values 1 and 0. After that, categorical features are transformed into numerical format using one-hot encoding through pd.get_dummies(). These steps ensure the data is in a suitable format for machine learning algorithms to process effectively.
## Step 11: Feature and Target Separation
Here in this part I separated the dataset into input features (X) and the target variable (y). The features represent the input data, while the target represents the output I wanted to predict.
## Step 12: Train-Test Split
Here I split the dataset into training and testing sets using an 80–20 ratio. The training set was used to train the model, and the testing set was used to evaluate its performance on unseen data.
## Step 13: Feature Scaling
I applied feature scaling using StandardScaler to standardize the features. This ensured that all variables contributed equally to the model and improved the performance of Logistic Regression.
## Step 14: Model Training
I trained a Logistic Regression model using the scaled training data. I set class_weight='balanced' to handle class imbalance and assigned the maximum iterations to 10000 to ensure proper convergence.
## Step 15: Model Evaluation and Prediction
After training the model, I used it to make predictions on the test dataset, allowing me to assess how well it performs on unseen data. I then evaluated the model using accuracy, precision, recall, and F1-score. Additionally, I analyzed a confusion matrix and a classification report to gain a deeper understanding of the model’s performance, with particular attention to the minority class.
Here is the result: 
Accuracy: 0.8436359615171957
Precision: 0.42254196642685854
Recall: 0.8075160403299725
F1 Score: 0.5547858942065491

Confusion Matrix:
 [[6748 1204]
 [ 210  881]]

Classification Report:
               precision    recall  f1-score   support

           0       0.97      0.85      0.91      7952
           1       0.42      0.81      0.55      1091

    accuracy                           0.84      9043
   macro avg       0.70      0.83      0.73      9043
weighted avg       0.90      0.84      0.86      9043

## Step 16: Hyperparameter Tuning
I applied Grid Search with cross-validation to find the best hyperparameters for the model. I tested different values of the regularization parameter (C) to improve performance.
## Step 17: Final Evaluation After Tuning
After tuning, I evaluated the model again. I observed an improvement in accuracy and precision, although there was a trade-off with recall. This helped me understand how tuning affects model performance.


Accuracy: 0.901249585314608
Precision: 0.6546875
Recall: 0.384051329055912
F1 Score: 0.48411322934719814

Confusion Matrix:
 [[7731  221]
 [ 672  419]]

Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.97      0.95      7952
           1       0.65      0.38      0.48      1091

    accuracy                           0.90      9043
   macro avg       0.79      0.68      0.71      9043
weighted avg       0.89      0.90      0.89      9043

