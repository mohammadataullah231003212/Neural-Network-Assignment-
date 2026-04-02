# Chest X-Ray Pneumonia Classification Using CNN


## Problem Statement

In this project, I built a Convolutional Neural Network (CNN) from scratch to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The goal was to train a deep learning model capable of detecting pneumonia reliably, with a strong focus on minimizing missed diagnoses — because in a medical setting, missing a sick patient (a False Negative) is far more dangerous than a false alarm.


## Dataset

I used the **Chest X-Ray Images (Pneumonia)** dataset, organized into three splits and stored in Google Drive:

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

The training set is significantly **class-imbalanced** — there are nearly 3× more PNEUMONIA images than NORMAL images. I addressed this deliberately at multiple stages of the pipeline.


## Step 1: Importing Necessary Libraries

In this step, I imported all the necessary Python libraries required for data loading, preprocessing, model building, training, and evaluation.

The key libraries I used were:

- **TensorFlow 2.19.0 / Keras** — for building and training the CNN
- **NumPy, Pandas** — for numerical operations and data handling
- **OpenCV (cv2) and PIL** — for image reading and preprocessing
- **Matplotlib and Seaborn** — for visualizing training history, confusion matrix, and sample images
- **Scikit-learn** — for evaluation metrics including classification report, ROC AUC, F1-score, and class weight computation
- **Google Colab Drive** — to mount my dataset stored in Google Drive

I also verified the TensorFlow version and checked GPU availability, since training a CNN is much faster on GPU.

python
print("TensorFlow Version :", tf.__version__)
print("GPU Available       :", tf.config.list_physical_devices('GPU'))
print("All libraries loaded successfully!")

**Output:**
Mounted at /content/drive
TensorFlow Version : 2.19.0
GPU Available       : []
All libraries loaded successfully!

## Step 2: Loading the Dataset & Verifying Directory Structure

After mounting Google Drive, I defined the paths to the training, validation, and test directories and printed the image counts per class to make sure the data was structured correctly.

python
base_dir  = '/content/drive/MyDrive/Archive'
train_dir = base_dir + '/train'
val_dir   = base_dir + '/val'
test_dir  = base_dir + '/test'

I then looped through each split and category to count the images:

python
for split in ['train', 'val', 'test']:
    for category in ['NORMAL', 'PNEUMONIA']:
        path = os.path.join(base_dir, split, category)
        count = len(os.listdir(path))
        print(f"{split} / {category} : {count} images")


**Output:**
train / NORMAL    : 1341 images
train / PNEUMONIA : 3875 images
val   / NORMAL    : 8 images
val   / PNEUMONIA : 8 images
test  / NORMAL    : 234 images
test  / PNEUMONIA : 390 images

This step confirmed the data was loaded correctly and clearly showed the **class imbalance problem** in the training set. It also revealed that the validation set is extremely small (only 16 images), which would later affect the reliability of validation metrics during training.


## Step 3: Class Weights & Preprocessing

### Why Class Weights?

Because the training set had 3,875 PNEUMONIA images but only 1,341 NORMAL images, I computed class weights using scikit-learn's `compute_class_weight`. This tells the model to penalize mistakes on the minority class (NORMAL) more heavily during training, so it does not simply learn to always predict PNEUMONIA.

python
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=np.array([0]*1341 + [1]*3875)
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class Weights:", class_weight_dict)

**Output:**
Class Weights: {0: 1.9448173005219984, 1: 0.6730322580645162}

This means every NORMAL image was treated as approximately **1.94×** more important than a PNEUMONIA image during loss computation. Without this correction, the model would be biased toward predicting PNEUMONIA for everything.


## Step 4: Image Data Generators & Augmentation

### Validation and Test Generators

For the validation and test sets, I only applied **rescaling** — dividing pixel values by 255 to bring them into the [0, 1] range. No augmentation was applied here because I needed consistent, unmodified images for evaluation.

### Training Generator — Data Augmentation

For the training set, I applied several augmentation techniques using `ImageDataGenerator`:

python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


| Parameter | Value | Why I Chose This |
|-----------|-------|-----------------|
| `rescale` | 1./255 | Normalizes pixel values to [0,1] — required for stable neural network training |
| `rotation_range` | 15° | Chest X-rays can be slightly rotated depending on patient positioning |
| `width_shift_range` | 0.1 | Simulates slight horizontal shifts in X-ray alignment |
| `height_shift_range` | 0.1 | Simulates slight vertical shifts in patient positioning |
| `shear_range` | 0.1 | Adds mild geometric distortion to improve generalization |
| `zoom_range` | 0.1 | Simulates varying distances during X-ray capture |
| `horizontal_flip` | True | A mirrored chest X-ray is still anatomically valid |
| `fill_mode` | 'nearest' | Fills any empty pixels after transformation with the nearest pixel value |

I kept augmentation values small and conservative on purpose. Chest X-rays are clinical images — over-rotating or distorting them could create unrealistic images that do not represent real clinical scenarios.

### Flow from Directory

python
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


| Parameter | Value | Reason |
|-----------|-------|--------|
| `target_size` | (150, 150) | Standardizes all images to the same size; 150×150 balances detail and memory efficiency |
| `batch_size` | 32 | Standard mini-batch size; large enough for stable gradient estimates, small enough to fit in memory |
| `class_mode` | 'binary' | Two-class problem (NORMAL vs PNEUMONIA) |
| `shuffle=False` | Test only | Ensures predictions align with true labels when evaluating |

**Output:**
Found 5216 images belonging to 2 classes.
Found 16 images belonging to 2 classes.
Found 624 images belonging to 2 classes.
Train samples : 5216
Val samples   : 16
Test samples  : 624
Classes       : {'NORMAL': 0, 'PNEUMONIA': 1}

---

## Step 5: Visualizing Sample Images & Class Distribution

### Sample Images

I wrote a function to display 6 sample images from the training generator with their class labels. This gave me a visual sanity check that the images were loading correctly and that augmentation was being applied as expected.

### Class Distribution Charts

I visualized the class distribution across the training and test sets using bar charts:

| Dataset | NORMAL | PNEUMONIA |
|---------|--------|-----------|
| Train   | 1,341  | 3,875     |
| Test    | 234    | 390       |

The charts clearly showed that PNEUMONIA heavily outnumbers NORMAL in both splits. This reinforced why class weights and recall-focused evaluation were essential for this project.

## Step 6: Building the CNN Model

I built a custom CNN using a Sequential architecture with 4 convolutional blocks followed by fully connected layers.

python
model = Sequential([
    Conv2D(32, (3,3), activation='tanh', padding='same', input_shape=(150, 150, 3)),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(64, (3,3), activation='tanh', padding='same'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(128, (3,3), activation='tanh', padding='same'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    Conv2D(256, (3,3), activation='tanh', padding='same'),
    BatchNormalization(), MaxPooling2D(2,2), Dropout(0.25),

    GlobalAveragePooling2D(),

    Dense(256, activation='tanh', kernel_regularizer=l2(0.001)),
    BatchNormalization(), Dropout(0.5),

    Dense(128, activation='tanh', kernel_regularizer=l2(0.001)),
    BatchNormalization(), Dropout(0.5),

    Dense(1, activation='sigmoid')
])

### Architecture Breakdown

| Layer | Role |
|-------|------|
| `Conv2D(32/64/128/256)` | Detects increasingly complex spatial features — edges → textures → anatomical patterns |
| `BatchNormalization` | Normalizes activations after each conv layer to stabilize and speed up training |
| `MaxPooling2D(2,2)` | Reduces spatial dimensions by half, keeping only the most prominent features |
| `Dropout(0.25)` | Randomly disables 25% of neurons in conv blocks to prevent overfitting |
| `GlobalAveragePooling2D` | Replaces Flatten — averages each feature map into a single value, greatly reducing parameters |
| `Dense(256/128)` | Fully connected layers that combine extracted features for final classification |
| `Dropout(0.5)` | Stronger 50% dropout in dense layers — these are more prone to overfitting |
| `l2(0.001)` | L2 regularization adds a penalty for large weights, discouraging the model from memorizing training data |
| `Dense(1, sigmoid)` | Output layer — sigmoid outputs a probability between 0 and 1 for binary classification |

### Key Decisions

**Why tanh instead of ReLU?**
I chose `tanh` as the activation function for the convolutional and dense layers. Unlike ReLU, tanh outputs values in the range [-1, 1] and is zero-centered, which can help with gradient flow in combination with BatchNormalization. It was an intentional experiment to compare its behavior against the more common ReLU in a medical imaging context.

**Why GlobalAveragePooling2D instead of Flatten?**
Flatten converts the full feature map into a long 1D vector, which dramatically increases the number of parameters in the dense layers. GlobalAveragePooling2D instead averages each feature map to a single number, greatly reducing parameters and making the model less prone to overfitting — especially important given the relatively small dataset.

**Why increase filters from 32 → 256?**
Each convolutional block doubles the number of filters. The early layers detect simple features (edges, gradients), while deeper layers combine these into more complex patterns (lung boundaries, opacity regions). More filters in deeper layers allow the model to capture this increased complexity.

**Model Size:**
Total params      : 490,689  (1.87 MB)
Trainable params  : 488,961  (1.87 MB)
Non-trainable params : 1,728 (6.75 KB)

## Step 7: Compiling the Model

### Custom F1 Metric

I wrote a custom F1 score metric function since Keras does not provide a built-in training-time F1 score:

python
def f1_score_metric(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    precision = tp / (tp + fp + K.epsilon())
    recall    = tp / (tp + fn + K.epsilon())
    f1        = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

`K.epsilon()` is added to every denominator to avoid division-by-zero errors during early training when predictions may be all one class.

### Compilation Settings

python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision, Recall, AUC, f1_score_metric]
)


| Setting | Value | Why |
|---------|-------|-----|
| `optimizer` | Adam (lr=0.001) | Adam adapts the learning rate per parameter — more robust than SGD for medical imaging tasks |
| `loss` | binary_crossentropy | Standard loss for binary classification; measures the difference between predicted probabilities and true labels |
| `metrics` | Accuracy, Precision, Recall, AUC, F1 | Multiple metrics give a complete picture of model performance, especially important for imbalanced medical data |

**Why Adam?**
Adam (Adaptive Moment Estimation) combines the benefits of two other optimizers — it keeps a running average of past gradients (like momentum) and adapts the learning rate for each parameter. This makes it more stable and faster to converge compared to vanilla SGD, which is especially useful here since training on CPU is already slow.

**Why binary_crossentropy?**
Since this is a two-class problem with a sigmoid output, binary cross-entropy is the mathematically correct loss function. It penalizes confident wrong predictions more severely, which helps the model learn better decision boundaries.



## Step 8: Callbacks

I set up three callbacks to control training behavior:

python
callbacks = [
    EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max'),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_recall', save_best_only=True, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
]

| Callback | What It Does | Why I Used It |
|----------|-------------|---------------|
| `EarlyStopping` | Stops training if `val_recall` does not improve for 5 consecutive epochs | Prevents wasting training time and overfitting |
| `restore_best_weights=True` | Rolls back to the epoch with the best val_recall when stopping | Ensures the final model is the best-performing version, not the last |
| `ModelCheckpoint` | Saves the model to disk whenever `val_recall` improves | Protects against losing the best model if training crashes |
| `ReduceLROnPlateau` | Halves the learning rate if `val_loss` does not improve for 3 epochs | Helps the optimizer make finer adjustments when it gets close to a minimum |

**Why monitor val_recall specifically?**
In medical diagnosis, **Recall (Sensitivity)** is the most critical metric — it measures how many actual PNEUMONIA cases the model correctly identifies. Missing a true PNEUMONIA case (False Negative) is clinically far more dangerous than a false alarm. I deliberately chose to monitor and save based on recall rather than accuracy for this reason.


## Step 9: Training the Model

python
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

| Parameter | Value | Why |
|-----------|-------|-----|
| `epochs` | 30 | Maximum of 30 training cycles; EarlyStopping will halt before this if needed |
| `class_weight` | `class_weight_dict` | Applies the computed imbalance correction during training |
| `callbacks` | All 3 callbacks | Controls early stopping, model saving, and learning rate scheduling |

**Training Observations (from Epoch 1 output):**

Epoch 1/30
Train  — Accuracy: 0.8976 | AUC: 0.9606 | Recall: 0.8911 | Loss: 0.5058
Val    — Accuracy: 0.5000 | AUC: 0.5547 | Recall: 1.0000 | Val Loss: 2.4109
→ val_recall improved to 1.0000 — model saved.

Epoch 2/30
→ val_recall did not improve from 1.0000

The model achieved perfect val_recall of 1.0 from Epoch 1 and it never improved further (already at maximum), so training ended early. However, the very high val_loss (2.41) at Epoch 1 alongside perfect val_recall strongly suggests the model was predicting **PNEUMONIA for everything** on the validation set — which would give 100% recall but 0% specificity. This was confirmed in the evaluation step.

## Step 10: Plotting Training History

I plotted 6 training curves to monitor performance over epochs:

- **Accuracy** (Train vs Val)
- **Loss** (Train vs Val)
- **Precision** (Train vs Val)
- **Recall** (Train vs Val)
- **AUC** (Train vs Val)
- **F1 Score** (Train vs Val)

**What the plots show:**
- Training accuracy and AUC were strong (~0.90 and ~0.96 by Epoch 1), showing the model was learning meaningful features from the training data
- Validation metrics were highly unstable — this is a direct consequence of the **extremely small validation set (only 16 images)**. With 16 images, a single misclassified image swings the metric by 6.25%, making validation curves unreliable as a training signal
- The val_loss spike to 2.41 at Epoch 1 while val_recall was 1.0 clearly indicates the model was over-predicting PNEUMONIA on the validation set


## Step 11: Evaluating the Model

python
results = model.evaluate(test_generator, verbose=1)


### Final Model Performance Summary

| Metric | Score |
|--------|-------|
| Accuracy | 0.6250 |
| Precision | 0.6250 |
| Recall | 1.0000 |
| AUC | 0.7217 |
| F1 Score | 0.6344 |
| Sensitivity | 1.0000 |
| **Specificity** | **0.0000** |
| ROC AUC | 0.7283 |

### Classification Report


              precision    recall  f1-score   support

      NORMAL       0.00      0.00      0.00       234
   PNEUMONIA       0.62      1.00      0.77       390

    accuracy                           0.62       624
   macro avg       0.31      0.50      0.38       624
weighted avg       0.39      0.62      0.48       624


### Confusion Matrix Results


True Negatives  (TN) : 0
False Positives (FP) : 234
False Negatives (FN) : 0   ← Most critical in medical!
True Positives  (TP) : 390

Sensitivity (Recall) : 1.0000
Specificity          : 0.0000

**What the confusion matrix tells us:**
The model predicted **PNEUMONIA for every single test image** — all 390 PNEUMONIA cases were correctly identified (TP = 390, FN = 0), but all 234 NORMAL cases were also incorrectly labeled as PNEUMONIA (FP = 234, TN = 0). This is a classic case of a model collapsing to the majority class under strong recall pressure.

### ROC Curve


ROC AUC Score : 0.7283

The ROC AUC of 0.7283 tells a more nuanced story than the accuracy or confusion matrix. While the model classifies everything as PNEUMONIA at the default 0.5 threshold, the underlying predicted probabilities do have some discriminative ability — meaning that at different threshold values, the model could achieve a more balanced trade-off between sensitivity and specificity. This suggests the model learned *some* useful features, even if the threshold needs adjustment.


## Key Decisions Summary

| Decision | Justification |
|----------|--------------|
| **tanh activation** | Zero-centered output helps gradient flow; intentional alternative to ReLU for experimentation |
| **GlobalAveragePooling2D** | Reduces parameters significantly compared to Flatten, reducing overfitting risk |
| **Monitor val_recall in callbacks** | Recall is the most clinically critical metric for pneumonia detection |
| **class_weight='balanced'** | Compensates for 3:1 PNEUMONIA:NORMAL imbalance in training data |
| **Conservative augmentation** | Chest X-rays are clinical images — aggressive augmentation could create unrealistic samples |
| **Adam optimizer** | Adaptive learning rate makes it robust and fast-converging, especially useful on CPU |
| **L2 regularization (0.001)** | Prevents overfitting in dense layers by penalizing large weight values |


## Findings

- The model achieved **100% Recall / Sensitivity** — it missed zero PNEUMONIA cases in the test set
- However, **Specificity was 0.0000** — it also classified every NORMAL case as PNEUMONIA
- The ROC AUC of **0.7283** suggests the model has some underlying discriminative ability, but the threshold needs to be adjusted
- The extremely small validation set (16 images) made validation metrics during training unreliable
- Training on CPU was very slow (~600 seconds per epoch), limiting the number of experiments that could be run

## Medical Context

| Aspect | Clinical Significance |
|--------|----------------------|
| **100% Recall, 0% Specificity** | In an emergency screening tool, missing zero PNEUMONIA cases is the primary goal — but flagging every patient as sick would overwhelm clinical staff and is not practically deployable |
| **False Negatives = 0** | From a patient safety standpoint, this is the ideal outcome — no sick patient goes undetected |
| **False Positives = 234** | Every healthy patient was sent for unnecessary follow-up — a significant burden on healthcare resources and patient anxiety |
| **ROC AUC = 0.7283** | Indicates moderate discriminative ability; threshold tuning could make this model clinically useful as a first-pass screening tool that flags cases for radiologist review |
| **Class weights** | Reflects the real-world clinical principle that missing a disease case is more harmful than over-diagnosing |
| **Recall-based early stopping** | Directly encodes the medical priority into training — the model is saved when it is best at finding sick patients, not when it is merely most accurate overall 
