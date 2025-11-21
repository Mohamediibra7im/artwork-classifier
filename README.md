
# Artwork Classification Project

- This project builds a system that classifies artwork images using deep learning.  
- It uses CNNs and transfer learning.  
- It compares three models and includes a GUI that allows users to upload images and view predictions with Grad-CAM results.

---

## Project Goals

- classify artwork types or styles  
- train and compare three CNN architectures  
- evaluate metrics including accuracy, precision, and recall  
- generate confusion matrices  
- build a simple GUI for predictions  
- display Grad-CAM heatmaps  
- maintain a clean and organized GitHub repository

---

## Dataset

The project uses the WikiArt dataset.  
It offers many artwork styles and enough samples for each class.  
Images are downloaded from public sources and organized into folders under `data/`.

---

## Folder Structure

```
data/
models/
src/
preprocessing/
training/
evaluation/
gradcam/
gui/
docs/
notebooks/

```

`data` stores the dataset.  
`models` holds trained weights.  
`src` contains the project code.  
`gui` includes the prediction interface.  
`docs` holds the report.  
`notebooks` stores experiments.

---

## Steps to Run the Project

1. install dependencies  
2. download the dataset and place it inside `data`  
3. run preprocessing scripts  
4. train VGG16, ResNet50, and EfficientNetB0  
5. evaluate the models  
6. run Grad-CAM on sample images  
7. start the GUI to test the classifier

---

## Models Used

- VGG16  
- ResNet50  
- EfficientNetB0  

All models start with pretrained weights and get fine-tuned for artwork styles.

---

## Evaluation

Key metrics:

- accuracy  
- precision  
- recall  
- confusion matrix  

Results from all models are compared to choose the best one for deployment.

---

## GUI

The GUI supports:

- image upload  
- predicted class output  
- confidence score  
- Grad-CAM heatmap display  

It gives a simple way to test the classifier.

---

## Team Members

- Mohammed Ibrahim  
- Mohammed Elfouly  
- Mohammed Ashraf  
- Mohammed Mohsen  
- Sarah Mahmoud  
- Rahma Nasser  

---

## How to Contribute

- create a branch for your task  
- commit your updates  
- write clear commit messages  
- open a pull request  
- wait for review before merging  
