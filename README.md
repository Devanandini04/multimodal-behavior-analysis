# 📌 Extraction of Gesture Features

This repository contains the complete implementation of the project **“Extraction of Gesture Features”**.

The project focuses on building an **end-to-end deep learning pipeline** to automatically extract, classify, and analyze human gestures from videos and correlate them with speech-related information at a frame level.

---

## 🎯 Project Objectives

The primary goals of this project are:

- Develop a robust **gesture extraction pipeline** using deep learning techniques.
- Detect and classify **human gestures** from video frames.
- Identify and associate:
  - the **type of gesture**
  - the **spoken words**
  - the **speaker**
  - the **speech type**
  within the same temporal frame.
- Perform **cross-parameter analysis** to uncover meaningful patterns between gestures and speech.
- Generate structured outputs for further research and visualization.

---

## 🧠 Key Features

- Frame-level gesture detection and classification  
- Multimodal alignment between **gesture and speech**
- Speaker and speech-type identification
- Automated analysis pipeline for behavioral insights
- Supports both **visual** and **tabular** outputs

---

## 📤 Output

The system produces one of the following outputs:

- 🎥 **Annotated video output** with overlaid gesture, speaker, and speech information  
- 📊 **Comprehensive dataframe** containing frame-wise details such as:
  - gesture type  
  - spoken word  
  - speaker identity  
  - speech category  

---
## Dataset

To ensure ethical compliance and reproducibility, this repository does **not**
include any video files, annotations, or extracted pose data.

The focus of this project is on reproducing the **data processing, annotation alignment,
and modeling pipeline**, which can be applied to any compatible dataset.


## 🗂️ Project Structure

```text
src/
│
├── data/              # Dataset processing and annotation alignment utilities
├── model/             # Deep learning models for gesture and speech processing
├── workflow/          # End-to-end pipeline for integrating models and generating outputs
├── Pose_Estimation/   # Pose extraction and keypoint generation modules
├── Segmentation/      # Shot and speaker segmentation utilities
├── Utils/             # Helper functions and shared utilities
├── config.py          # Centralized configuration for dataset and training parameters
