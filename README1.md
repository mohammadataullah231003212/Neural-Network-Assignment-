# Chest X-Ray Pneumonia Classifier

A CNN that looks at chest X-ray images and tries to figure out if the person has pneumonia or not.

## What This Is

I trained a convolutional neural network (CNN) on chest X-ray images to classify them as either **NORMAL** or **PNEUMONIA**. The main goal was to catch as many pneumonia cases as possible, because missing a sick patient is way worse than a false alarm.


## Dataset

- **Train:** 1,341 normal | 3,875 pneumonia
- **Val:** 8 normal | 8 pneumonia 
- **Test:** 234 normal | 390 pneumonia

There's a big class imbalance — nearly 3x more pneumonia images than normal ones. This needed to be handled carefully.


## Methodology (Step by Step)

1. **Loaded the dataset** — here i mounted google drive, set up folder paths, and counted images per class to confirm everything was good.
2. **Handled class imbalance** — in this portion I computed class weights so the model pays more attention to the minority class (NORMAL)
3. **Preprocessed & augmented images** — rescaled all images to [0,1], applied light augmentation on training data only
4. **Built the CNN model** — 4 convolutional blocks with increasing filters (32→64→128→256), followed by dense layers
5. **Compiled the model** — used Adam optimizer, binary crossentropy loss, and tracked accuracy, precision, recall, AUC, and F1
6. **Set up callbacks** — early stopping, model checkpointing, and learning rate reduction, all monitored on val_recall
7. **Trained the model** — ran for up to 30 epochs with class weights applied
8. **Evaluated on test set** — checked confusion matrix, ROC AUC, sensitivity, and specificity


## Approach

### Handling Class Imbalance
- I handled the imbalance by using compute_class_weight to balance the training data.
- I gave NORMAL images about 1.94× more weight than PNEUMONIA images.
- This helped me prevent the model from always predicting PNEUMONIA.

### Data Augmentation
- Small rotations (±15°)
- Slight horizontal and vertical shifts
- Mild zoom and shear
- Horizontal flipping
- Kept augmentation light — X-rays are clinical, too much distortion makes them unrealistic

### Model Architecture
- 4 conv blocks with filters: 32 → 64 → 128 → 256
- Each block has BatchNorm, MaxPooling, and Dropout (25%)
- using  `GlobalAveragePooling2D` insted of flatten — fewer parameters, less overfitting
- Two hidden layers (256, 128) with L2 regularization and 50% Dropout
- i used `tanh` activation instead of ReLU — because ReLU discards anything less then 0
- Output layer: single neuron with sigmoid activation, because our model is doing binary classification.

### Training Setup
- Optimizer: Adam (lr=0.001)
- Loss: Binary crossentropy
- Monitored **val_recall** for early stopping and saving the best model
- `ReduceLROnPlateau` halves the learning rate if val_loss doesn't improve for 3 epochs
- Recall was the priority metric — missing a sick patient is the worst outcome


## Libraries Used

- **TensorFlow / Keras** — main framework for building and training the CNN
- **NumPy** — used for array operations and setting up class label arrays
- **Pandas** — used for basic data handling
- **OpenCV (cv2) & PIL** — used for reading and preprocessing images
- **Matplotlib & Seaborn** — used to plot training curves, confusion matrix, and sample images
- **Scikit-learn** — used for computing class weights, classification report, and ROC AUC score
## Findings

| Metric | Score |
|--------|-------|
| Accuracy | 62.5% |
| Recall (Sensitivity) | 100% |
| Specificity | 0% |
| AUC | 0.72 |
| F1 Score | 0.63 |

The model ended up predicting **PNEUMONIA for every single image**. So it caught all 390 pneumonia cases (FN = 0), but also flagged all 234 healthy patients as sick (FP = 234, TN = 0).

This happened because of the strong recall pressure from class weights + monitoring val_recall. The model basically took a shortcut — "just say PNEUMONIA every time and you'll never miss a case."

The ROC AUC of 0.72 is more encouraging though — the raw probabilities do have some discriminative ability. At the default 0.5 threshold everything tips to PNEUMONIA, but adjusting the threshold could give a more balanced result.
## Limitations
- The validation set has only 16 images, which is too small to properly check how well the model is working. one wrong prediction swings the metric by 6%. It  made training unreliable
- The model ended up predicting every image as PNEUMONIA, which is not correct.

