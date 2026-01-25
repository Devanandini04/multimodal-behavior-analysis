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

## 🗂️ Project Structure

```text
src/
│
├── model/        # Deep learning models for gesture and speech processing
├── workflow/     # Main pipeline for integrating models and overlaying outputs
├── Models/       # Pre-trained and fine-tuned models
