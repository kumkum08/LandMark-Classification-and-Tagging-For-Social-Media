# 🖼️ Convolutional Neural Networks (CNN) Landmark Detection Project

![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📌 Project Overview
This project demonstrates how **Convolutional Neural Networks (CNNs)** can be used to automatically predict the most likely location of an image based on recognizable landmarks. The pipeline processes **user-supplied images** and outputs the **top-k most relevant landmarks** (from 50 possible landmarks across the world). By building this project, I explored the challenges of piecing together multiple models into a real-world pipeline, the tradeoffs between training **CNN from scratch** vs. **Transfer Learning**, and building an end-to-end app powered by deep learning.

## 🎯 Motivation
Photo-sharing and storage platforms often rely on **location metadata** for features like automatic tagging, smart album organization, and location-based search. However, many images lack metadata (e.g., GPS disabled, privacy concerns, stripped EXIF data). This project addresses that gap by using **CNNs for landmark recognition** directly from image content.  

## ⚙️ Project Workflow
The project follows three main stages:
1. **CNN from Scratch** – Implemented a CNN and trained it on landmark images.  
2. **Transfer Learning** – Used pretrained models (e.g., ResNet, VGG) for higher accuracy and faster convergence.  
3. **App Development** – Built an application that accepts images as input and outputs the top probable landmarks.  

## 🛠️ Setup Instructions
### Local Setup
```bash
# Clone the repository
git clone https://github.com/your-username/cnn-landmark-detection.git
cd cnn-landmark-detection

# Create and activate conda environment
conda create --name cnn_project -y python=3.7.6
conda activate cnn_project

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
pip install jupyterlab
jupyter lab
