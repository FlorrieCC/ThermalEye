# ThermalEye: Real-Time Blink Detection with Thermal Sensors

This project presents a lightweight, privacy-preserving blink detection system using the MLX9064x thermal infrared array sensors. It enables real-time, non-contact eye state monitoring suitable for static, natural environments like reading or AR/VR usage.

---

## 📦 Part 1: Data Collection

### 🔧 Environment Setup

ThermalEye is built around the **MLX90641** thermal infrared array sensor, a compact, uncooled FIR sensor ideal for wearable and privacy-preserving physiological monitoring.

- ✅ **Thermal Sensor**: MLX90641  
  ➤ [Product Introduction (Melexis)](https://www.melexis.com/en/product/mlx90641/high-operating-temperature-fir-thermal-sensor-array)  
  ➤ [Official Driver Library (C/C++)](https://github.com/melexis/mlx90641-library)  
  ➤ Python drivers and examples are also available in the community or through I2C wrappers.

> ⚠️ Currently, this project does **not** support MLX90640 or other variants.

- ✅ **RGB Camera (Optional)**:  
  Intel RealSense camera is used during data collection for annotation and alignment.  
  Please make sure it is properly installed and accessible via `pyrealsense2`.

### ▶ Run Data Collection Script

After setting up the MLX90641 and optionally the RealSense camera, you can start collecting synchronized thermal and RGB data using:

```bash
git clone https://github.com/FlorrieCC/ThermalEye.git
cd ThermalEye
python real_ira.py
```

This script records thermal frames from MLX90641 and aligns them with RGB images (if a RealSense camera is connected), saving them for downstream training and evaluation.

> 📝 Tip: You can modify `real_ira.py` to skip RealSense recording if only thermal data is required.

---


## 🧠 Part 2: Train and Evaluate the Model

The `ira_data/` directory contains our original raw thermal recordings, and `gt_output/` contains the corresponding ground truth labels.

All model training and evaluation code is organized under the `training/` directory.

### 🔧 Setup Training Environment

Install required dependencies for model training:

```bash
pip install -r training/requirements.txt
```

> This includes `torch`, `torchvision`, `scikit-learn`, and other relevant packages.

### ✅ Evaluate a Pretrained Model (Quick Start)

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

---

## 🚀 Part 3: Real-Time Demo (Quick Start)

_Coming soon..._

---


