# ThermalEye: Real-Time Blink Detection with Thermal Sensors

This project presents a lightweight, privacy-preserving blink detection system using the MLX9064x thermal infrared array sensors. It enables real-time, non-contact eye state monitoring suitable for static, natural environments like reading or AR/VR usage.

---

## 📦 Dataset Introduction

The dataset used in this project was collected using two synchronized sensors:

- ✅ **Thermal Sensor**: [MLX90641](https://www.melexis.com/en/product/mlx90641/high-operating-temperature-fir-thermal-sensor-array)  
  A compact, low-resolution thermal infrared array sensor (16×12), operating in the 7.5–14 µm range. It enables passive, privacy-preserving monitoring by capturing temperature distributions without relying on ambient light. This sensor is ideal for wearable applications and was used to record thermal signals from the eye region during blinking.

- ✅ **RGB Camera (Optional for Annotation)**: Intel RealSense  
  Used to capture synchronized RGB videos as a visual reference and to assist in blink annotation.  
  The RGB stream was aligned with the thermal frames during data collection but is not required for model inference.

> ⚠️ Note: All data used in this project was collected offline and is provided in the repository. No real-time hardware setup is needed to reproduce our training or evaluation results.

---


## 🧠 Train and Evaluate the Model

If you do not have the hardware setup to collect your own data, there are **two quick start options** to get started with our pretrained model or train one from scratch using the provided data.

The `ira_data/` directory contains our original raw thermal recordings, and `gt_output/` contains the corresponding ground truth labels.  
All model training and evaluation code is organized under the `training/` directory.

First, clone the repository:

```bash
git clone https://github.com/FlorrieCC/ThermalEye.git
cd ThermalEye
```

Then, create and activate a new conda environment:

```bash
conda create -n thermalEye python=3.10
conda activate thermalEye
```

Install training dependencies:

```bash
pip install -r training/requirements.txt
```

---

### 🏋️‍♀️ Option 1: Train and Evaluate a Model

Once the environment is ready, you can train and evaluate a model from scratch by running:

```bash
python training/main.py
```

This will train the model on the provided dataset and print evaluation metrics to the console after training completes.

> ⚠️ Training requires a GPU with **at least 24 GB** of memory.  
> If your device does not meet this requirement, you can reduce the values of `TRAIN_BATCH_SIZE` and `VAL_BATCH_SIZE` in `training/constants.py`.  
> If the training still fails due to memory constraints, you can skip training and proceed with **Option 2** to directly evaluate our pretrained model.

---

### ✅ Option 2: Evaluate a Pretrained Model

To quickly evaluate our pretrained model:

1. Ensure the checkpoint file exists:  
   `checkpoints/sample_model.pth`

2. Run the evaluation script:

   ```bash
   python training/evaluate.py
   ```

3. Results will be printed to the console and saved under:

   ```
   evaluate_output/
   ```






