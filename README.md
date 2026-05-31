# 🏎️ F1 Pit Strategy Predictor

Welcome to the F1 Pit Strategy Predictor! This project analyzes historical Formula 1 data, engineers features like tire age and track conditions, and uses machine learning (Random Forest) to predict the optimal time to pit. It also integrates real-time telemetry using the `fastf1` API.

## 🚀 How to Run the Project

Follow these steps to run the Jupyter Notebook on your own system.

### 1. Prerequisites
You will need Python installed on your computer. You also need to install the required libraries. Open your terminal or command prompt and run the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn fastf1 notebook
```

### 2. Download the Project
Clone this repository to your local machine:

```bash
git clone https://github.com/Shane-dev29/F1_dmdw.git
cd F1_dmdw
```

### 3. Start Jupyter Notebook
Once you are inside the `F1_dmdw` folder in your terminal, start the Jupyter server by running:

```bash
jupyter notebook
```

### 4. Run the Code
- Your web browser will open automatically.
- Click on `F1.ipynb` to open the notebook.
- At the top of the screen, click **Cell** -> **Run All** (or **Run** -> **Run All Cells** depending on your version) to execute the entire project from top to bottom.

---

### 📂 What's Included:
- **`F1.ipynb`**: The main notebook containing all the data cleaning, feature engineering, and machine learning models.
- **`app.py`**: A Streamlit dashboard application for visualizing the data.
- **`model.pkl`**: The pre-trained Random Forest model.
- **`.csv` files**: All the historical Formula 1 datasets required for the project (races, drivers, pit stops, lap times, etc.).

*Note: The first time you run the notebook, the `fastf1` API will download telemetry data. This may take a moment, and it will create a local `cache/` folder on your machine automatically to speed up future runs.*
