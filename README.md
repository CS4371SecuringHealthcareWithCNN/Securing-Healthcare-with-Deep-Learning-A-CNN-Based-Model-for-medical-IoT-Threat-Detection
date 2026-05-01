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

## Overview
This repository provides implementations of Logistic Regression, AdaBoost, a Deep Neural Network (DNN), and Random Forest alongside the original CNN model proposed in the paper, each evaluated in both their uncompressed and compressed forms. All models are applied to the task of intrusion detection in Internet of Medical Things (IoMT) systems, classifying network traffic across three configurations: binary (2-class), categorical (6-class), and multiclass (19-class).
---

## Performance Metrics WIP

<div align="center">

![Model Comparison](/images/Performance_Metrics.png)

*Performance comparison across different classification tasks*

* CNN, and Random Forest seem to compress very well while also keeping performance metrics relatively high.
* AdaBoost on the other hand while having somewhat consistently high results does not compress well and results in compression being around 1%-5% per class configuration. Any techniques to compress further risk tanking performance further. 
* Next is Logistic Regression, it compresses extremely well, almost a 10x compression! The caveat however, is that precision, recall, and F1 scores plummet.
* Lastly Deep Neural Network, which compresses somewhat very well, but seeing as the model is already quite small ~0.21 MB it does still compress quite a bit ~68% resulting in a compressed model size of about 0.07 MB. Its kinda like AdaBoost in terms of starting small, but compresses way better, and having the worst scores out of all the models.
</div>

---

## 🚀 Quick Start

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

### Instructions for CNN

Navigate to the `src` directory:
```bash
cd src
```
Run the model with this configuration 
```bash
python cnn_main.py --class_config <num_classes>
```
Once trained and these files are present:
    baseline.keras- Trained CNN
    X_test.npy, Y_test.npy- preprocessed test data
    X_calib.npy- stratified training samples for int8 calibration
    class_names.npy- readable class names
Run compression and benchmarking:

python cnn_compression.py --model baseline.keras --X_test X_test.npy --y_test y_test.npy --X_calib X_calib.npy --class_names class_names.npy --subsample_test (between 50000 and 200000)

Output is printed in terminal and in results file

### Instructions for AdaBoost
Navigate to the src directory
```bash
cd src
```
Run the model with this configuration
```bash
python AdaBoost.py --class_config <2, 6, or 19>
```

Once the models are trained, and the current files are present

*  adaboost_model_class2.pkl for binary classification
* adaboost_model_class6.pkl for categorical classification
* adaboost_model_class19.pkl for multiclass classification

Once pretrained models are loaded a full classification report will be printed in the terminal. Followed by a quantization impact summary table.

### Instructions for Logistic Regression
Navigate to the src directory
```bash
cd src
```
Run the model with this configuration
```bash
python logistic_regression_model.py --class_config <2, 6, or 19>
```
## 📂 Project Structure

```
project/
├── data/
│   ├── train/            # Training CSV files
│   └── test/             # Testing CSV files
├── src/
│   ├── AdaBoost.py
|   |── cnn_compression.py
|   |── cnn_main.py
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── dnn_main.py  
│   ├── dnn_model.py  
│   ├── dnn_quantize_model.py
│   ├── logistic_regression_model.py    # LR model definition, training, and execution
│   ├── main.py           # Main CNN execution script 
│   ├── model.py          # CNN model definition and training
│   ├── quantize_model.py   # LR model compression
│   └── random_forest.py  
Pre-Trained Models
|   |──adaboost_model_class[x].pkl
|   |──baseline.keras
│   └── class_names.npy
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation (this file)
└── README_DATA.md        # Data preparation guide
```

## Next Steps

Quantization aware training.

* We want to integrate quantization directly into the pretraining and fine tuning process rather than Post Training Quantization which is what we have. [You can read more Post Training Quantization Here](https://www.ibm.com/think/topics/quantization-aware-training)

![pre-training quantization](/images/quantization-aware-training-116x9.png)
