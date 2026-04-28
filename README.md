# Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection

# CS4371 Group 9
### Christina Martinez · Mason Meinscher · Roman Merlick · Jay Suh · Zane Zezulka

*Based on:* [Securing Healthcare with Deep Learning: A CNN-Based Model for medical IoT Threat Detection](https://arxiv.org/abs/2410.23306) | [GitHub](https://github.com/alirezamohamadiam/Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection)


**The README file in the GitHub repository indicates clearly how to easily clone and build/deploy the code.**
**The README file in the GitHub repository clearly indicates what functionality does (and does not, if applicable) work in the final version of the application.**

**Additionally, it should include references to two scholarly papers:** 
- one representing prior research, serving as the foundational bedrock for the current study, and 
- another representing contemporary work that acknowledges and builds upon the findings of the current paper. 
- This practice is akin to the methodology of an archeologist, meticulously documenting the lineage of ideas and advancements within the academic landscape.


---

## Overview WIP
This repository provides implementations of Logistic Regression, AdaBoost, a Deep Neural Network (DNN), and Random Forest alongside the original CNN model proposed in the paper, each evaluated in both their uncompressed and compressed forms. All models are applied to the task of intrusion detection in Internet of Medical Things (IoMT) systems, classifying network traffic across three configurations: binary (2-class), categorical (6-class), and multiclass (19-class).
---

## Performance Metrics WIP

<div align="center">

![Model Comparison](https://github.com/user-attachments/assets/7dc2bd46-c2ea-49cb-b94f-7ee42b268d56)

*Performance comparison across different classification tasks*

</div>

---

## 🚀 Quick Start WIP

### Step 1: Clone Repository
```bash
git clone https://github.com/CS4371SecuringHealthcareWithCNN/Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection
```
> Make sure Git is installed on your machine. If not, grab it from: https://git-scm.com/install

### Step 2: Install Requirements
Navigate to Project Directory
```bash
cd Securing-Healthcare-with-Deep-Learning-A-CNN-Based-Model-for-medical-IoT-Threat-Detection
```
then:
```bash
pip install -r requirements.txt
```
> **Note:** Python 3.7+ is required

### Step 3: Run Training
Navigate to the `src` directory:
```bash
cd src
```
To run the model, execute `main.py` and specify the classification configuration:
```bash
python main.py --class_config <num_classes>
```

Replace `<num_classes>` with:
- **2** for binary classification,
- **6** for categorical,
- **19** for multiclass.

**Example (binary classification):**
```bash
python main.py --class_config 2
```
---

## 📂 Project Structure WIP

```
project/
├── data/
│   ├── train/            # Training CSV files
│   └── test/             # Testing CSV files
├── src/
│   ├── AdaBoost.py
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── dnn_main.py  
│   ├── dnn_model.py  
│   ├── dnn_quantize_model.py
│   ├── logistic_regression_model.py    # LR model definition, training, and execution
│   ├── main.py           # Main CNN execution script 
│   ├── model.py          # CNN model definition and training
│   ├── quantize_model.py   # LR model compression
│   └── random_forest.py  
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation (this file)
└── README_DATA.md        # Data preparation guide
```

































