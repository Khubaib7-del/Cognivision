# CogniVision Setup Guide

Welcome to the project! Follow these steps to set up your environment using PyCharm.

## 🛠 Prerequisites
- **PyCharm Professional or Community Edition**
- Git installed on your system.

## 0️⃣ Install Python (If not already installed)
If you don't have Python installed on your computer:
1. **Official Download**: Visit [python.org/downloads](https://www.python.org/downloads/) and download **Python 3.10** or higher.
2. **Installation**: Run the `.exe` file. 
   - **IMPORTANT**: Check the box that says **"Add Python to PATH"** before clicking Install.
3. **PyCharm Alternative**: When you follow Step 2 below, PyCharm can also download Python for you automatically!

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

## 🚀 Running the App
1. In the PyCharm terminal, run:
   ```bash
   python src/api/main.py
   ```
2. Open your browser at `http://localhost:8000`.

## 🔑 Environment Variables
1. Copy `.env.example` to `.env`.
2. Update the keys inside `.env` (do NOT commit `.env`).
