# Neural-Network-Assignment-
# Problem Statement
A banking institution wants to predict whether a customer will subscribe to a term deposit based on their banking behavior.
## Dataset
For this assignment I am using a dataset called bank-full.csv, provided by the instructor. It contains 45,211 data points and 17 attributes. The target variable is y, which indicates whether a customer subscribed to a term deposit (yes or no)."
## Methodology
### **Step1:Importing Libraries**
In this step, I imported all the necessary Python libraries required for data preprocessing, model building, training, and evaluation.
### Step 2: Load Dataset
I loaded the dataset using `pandas` and specified the correct delimiter (`;`) since the dataset is not comma-separated.
### Step 3: Encode Categorical Data
Since machine learning models cannot process text data directly, I converted all categorical columns into numerical values using `LabelEncoder`.
### Step 4: Split Features and Target
I separated the dataset into:
- `X` → input features (all columns except `y`)
- `y` → target variable (subscription result)
### Step 5: Train-Test Split
I split the dataset into training and testing sets using an 80/20 ratio. The training set is used to train the model, while the test set is used to evaluate performance.
### Step 6: Model Selection
I selected **Logistic Regression** as the classification model and set `max_iter=1000` to ensure proper convergence.
### Step 7: Model Training
I trained the model using the training dataset so it could learn patterns between input features and the target variable.
### Step 8: Prediction
After training, I used the model to predict outcomes on the test dataset.
### Step 9: Model Evaluation
I evaluated the model using **accuracy score**, which measures the percentage of correct predictions.Here is the answer:
Accuracy: 0.8879796527700984
## Link:
https://colab.research.google.com/drive/1r0elaq0rmyleoA635A3JbQp9yJU1lhcG#scrollTo=59XSFUCC9ctG
Can be accessed by:"ashrafur.c@eastdelta.edu.bd"
## **Note:**
I did the coding in google colab and uploaded the .ipynb file here in the section where there suppose to be only code.
