# CogniVision Setup Guide

Welcome to the project! Follow these steps to set up your environment using PyCharm.

## 🛠 Prerequisites
- **PyCharm Professional or Community Edition**
- Git installed on your system.

## 📥 Getting Started
1. **Clone the Repo**:
   ```bash
   git clone https://github.com/Khubaib7-del/Cognivision.git
   cd Cognivision
   ```
2. **Open in PyCharm**:
   Open the folder as a new project in PyCharm.

## 🐍 Configure Python Interpreter (Inside PyCharm)
PyCharm manages its own internal interpreters.
1. Go to **File > Settings > Project: CogniVision > Python Interpreter**.
2. Click **Add Interpreter...** and select **Add Local Interpreter...**.
3. Choose **Virtualenv Environment**.
   - Location: `[project-path]\venv`
   - Base interpreter: Select your local Python version (e.g., Python 3.10+).
4. Click **OK**.

## 📦 Install Dependencies
1. Open the **Terminal** tab at the bottom of PyCharm.
2. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

## 🔑 Environment Variables
1. Copy `.env.example` to `.env`.
2. Update the keys inside `.env` as needed (do NOT commit `.env`).
