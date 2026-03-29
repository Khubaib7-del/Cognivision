# CogniVision Setup Guide (Antigravity Edition)

Welcome to the project! Follow these steps to set up your environment directly inside **Antigravity**.

## 🛠 Prerequisites
- **Antigravity IDE** (Current session)
- Git installed on your system.

## 0️⃣ Install Python (If not already installed)
If you don't have Python installed on your computer:
1. **Download**: Visit [python.org/downloads](https://www.python.org/downloads/) and download **Python 3.10** or higher.
2. **Installation**: Check the box **"Add Python to PATH"** before clicking Install.

## 🐍 Configure Python Interpreter in Antigravity
1. Click the highlighted **"Select Python Interpreter"** message at the bottom.
2. Select **"Enter interpreter path..."**.
3. Pick the file named simply **`python.exe`** from this folder:
   `C:\Users\T L S\AppData\Local\Python\bin\`
4. If it says "Unable to handle": **DON'T WORRY**. Your environment is already set up and ready in the terminal!

## 🚀 Running the Data Collector
If the IDE still shows red underlines, ignore them for now. You can still run the full project from the terminal:
```powershell
& "C:\Users\T L S\AppData\Local\Python\bin\python.exe" src/data/collector.py
```
**This is the most important command to get your project started!**

## 📦 Install Dependencies
I have already installed the main AI libraries for you! If you ever need to add more, run this in the Antigravity terminal:
```powershell
& "C:\Users\T L S\AppData\Local\Python\bin\python.exe" -m pip install -r requirements.txt
```

## 🚀 Running the Data Collector
Once the interpreter is selected, run our tool to start building your dataset:
```powershell
cd cognivision
& "C:\Users\T L S\AppData\Local\Python\bin\python.exe" src/data/collector.py
```

## 🔑 Environment Variables
1. Copy `.env.example` to `.env`.
2. Update the keys inside `.env` (do NOT commit `.env`).
