# 🏎️ F1 Pit Strategy Predictor — Jupyter Notebook Step-by-Step Guide

This document walks you through every step executed in `F1.ipynb` so you can understand and replicate the full project from scratch.

---

## PHASE 1: ENVIRONMENT SETUP & DATA LOADING

### Step 1 — Check Python Version
```python
import sys
print(sys.version)
```
**What it does:** Confirms Python is installed and prints the version (we used Python 3.11.9).

### Step 2 — Test Pandas Import
```python
import pandas as pd
print("All good 🚀")
```
**What it does:** Imports the `pandas` library (used for data manipulation) and prints a success message to confirm it's working.

### Step 3 — Load All Three Datasets
```python
import pandas as pd

races = pd.read_csv("races.csv")
pit = pd.read_csv("pit_stops.csv")
drivers = pd.read_csv("drivers.csv")
```
**What it does:** Loads three CSV files into memory as DataFrames:
- `races.csv` — Contains every F1 race from 1950 to 2024 (1125 rows), including race name, year, round, circuit, and date.
- `pit_stops.csv` — Contains 11,371 individual pit stop records (from 2011 onwards), including which driver pitted, on which lap, and how long it took.
- `drivers.csv` — Contains 861 F1 drivers with their names, nationalities, and date of birth.

### Step 4 — Preview the Data
```python
print(races.head())
print(pit.head())
print(drivers.head())
```
**What it does:** Prints the first 5 rows of each dataset so we can visually inspect what the data looks like.

### Step 5 — Check Data Types and Structure
```python
print(races.info())
print(pit.info())
print(drivers.info())
```
**What it does:** Prints the column names, data types (int, string, etc.), and how many non-null values each column has. This helps us understand the shape of our data before we start working with it.

### Step 6 — View Column Names
```python
print(races.columns)
print(pit.columns)
print(drivers.columns)
```
**What it does:** Lists all column names in each dataset. We need to know these to merge and manipulate the data later.

### Step 7 — Explore Pit Stop Statistics
```python
pit['lap'].describe()
```
**What it does:** Generates statistical summary of the `lap` column in pit stops — shows the mean pit stop happens around lap 25, with a minimum of lap 1 and maximum of lap 78.

### Step 8 — Check Data Coverage
```python
print(races['year'].unique())
print(pit['driverId'].nunique())
```
**What it does:** Shows all years present in the races dataset (1950–2024) and counts unique drivers in pit stops (76 unique drivers).

---

## PHASE 2: DATA MERGING & CLEANING

### Step 9 — Merge All Datasets Together
```python
df = pit.merge(races, on='raceId')
df = df.merge(drivers, on='driverId')
```
**What it does:** Joins the three datasets into one master DataFrame using shared keys:
- First joins pit stops with races using `raceId` (so each pit stop now has the race name, year, etc.)
- Then joins with drivers using `driverId` (so each pit stop now has the driver's name and nationality)
- Result: A single DataFrame with 32 columns containing all the information we need.

### Step 10 — Check for Duplicate Rows
```python
print(df.duplicated().sum())
print(df.duplicated(subset=['driverId', 'raceId', 'lap']).sum())
```
**What it does:** Checks if any rows are exact duplicates (result: 0 duplicates found). Also checks if any driver has two pit stops recorded on the same lap in the same race (result: 0 — data is clean).

### Step 11 — Check for Missing Values
```python
print(df.isnull().sum())
```
**What it does:** Counts null/missing values in every column. Result: 0 nulls in every column — the data is complete and clean, no imputation needed.

---

## PHASE 3: FEATURE ENGINEERING

### Step 12 — Calculate Total Laps Per Race
```python
total_laps = df.groupby('raceId')['lap'].max()
df['total_laps'] = df['raceId'].map(total_laps)
```
**What it does:** Groups the data by each race and finds the maximum lap number recorded (which represents the total laps in that race). Then maps this value back to every row so each pit stop knows how many laps were in its race.

### Step 13 — Create Lap Ratio Feature
```python
df['lap_ratio'] = df['lap'] / df['total_laps']
```
**What it does:** Creates a normalized feature between 0 and 1 representing how far into the race a pit stop occurred. For example, 0.5 means the pit stop happened exactly halfway through the race. This is crucial because different races have different total laps (50–78), so raw lap numbers aren't directly comparable.

### Step 14 — Create Pit Window Categories
```python
df['pit_window'] = pd.cut(df['lap_ratio'],
                         bins=[0, 0.2, 0.4, 0.7, 1.0],
                         labels=['early', 'mid-early', 'mid', 'late'])
```
**What it does:** Bins the continuous `lap_ratio` into four human-readable categories so we can analyze pit stop strategy patterns (e.g., "most stops happen mid-race").

### Step 15 — Create Driver Name Column
```python
df['driver_name'] = df['forename'] + " " + df['surname']
```
**What it does:** Combines the first name and last name columns into a single readable name (e.g., "Lewis Hamilton").

### Step 16 — Simulate Weather Conditions
```python
import numpy as np
df['weather'] = np.random.choice(['dry', 'wet'], size=len(df), p=[0.8, 0.2])
df['is_rain'] = (df['weather'] == 'wet').astype(int)
```
**What it does:** Since the original dataset does not include weather data, we simulate it: 80% chance of dry conditions, 20% chance of rain. Then creates a binary `is_rain` column (0 = dry, 1 = wet) for the ML model.

### Step 17 — Create Pit Decision Target Variable
```python
df['pit_decision_v3'] = (
    ((df['tire_age'] > 25) & (df['lap_ratio'] > 0.3)) |
    ((df['is_rain'] == 1) & (df['lap_ratio'] > 0.2)) |
    (df['lap_ratio'] < 0.1)
).astype(int)
```
**What it does:** Engineers a target variable (what we want the AI to predict). A pit stop is considered "optimal" if:
- Tire age is over 25 laps AND we're past the first 30% of the race, OR
- It's raining AND we're past the first 20% of the race, OR
- It's very early in the race (possible damage or formation lap stop)

### Step 18 — Create Interaction Features
```python
df['tire_lap_interaction'] = df['tire_age'] * df['lap_ratio']
df['rain_mid'] = df['is_rain'] * df['lap_ratio']
```
**What it does:** Creates interaction features that capture combined effects — for example, old tires late in the race is far more critical than old tires early on.

---

## PHASE 4: DATA VISUALIZATION

### Step 19 — Plot Pit Stop Distribution
```python
import matplotlib.pyplot as plt
plt.hist(df['lap_ratio'], bins=30)
plt.title("Pit Stop Distribution (Lap Ratio)")
plt.show()
```
**What it does:** Creates a histogram showing when pit stops typically occur. The distribution reveals that most pit stops cluster around the 0.3–0.7 range (mid-race).

### Step 20 — Top 10 Races by Pit Stops
```python
race_pits = df['name'].value_counts().head(10)
race_pits.plot(kind='bar')
plt.title("Top 10 Races by Pit Stops")
plt.show()
```
**What it does:** Bar chart showing which Grand Prix events historically have the most pit stops.

### Step 21 — Pit Stops Over the Years
```python
year_pits = df['year'].value_counts().sort_index()
year_pits.plot()
plt.title("Pit Stops Over Years")
plt.show()
```
**What it does:** Line chart showing how pit stop frequency has changed from 2011 to 2024.

### Step 22 — Driver Pit Timing Boxplot
```python
import seaborn as sns
top_drivers = df['driver_name'].value_counts().head(10).index
sns.boxplot(x='driver_name', y='lap_ratio', data=df[df['driver_name'].isin(top_drivers)])
plt.title("Pit Timing Distribution (Top Drivers)")
plt.show()
```
**What it does:** Creates a boxplot comparing the pit stop timing behavior of the top 10 most active drivers. Some drivers consistently pit early (aggressive strategy) while others pit late (conservative strategy).

---

## PHASE 5: FIRST ML MODEL (Simple Version)

### Step 23 — Prepare Features and Target
```python
X = df[['tire_age', 'lap_ratio', 'is_rain']]
y = df['pit_decision_v3']
```
**What it does:** Defines the input features (X) and the target variable (y) for the machine learning model.

### Step 24 — Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**What it does:** Splits data into 80% training and 20% testing. `random_state=42` ensures reproducibility.

### Step 25 — Train and Compare 4 Models
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

lr = LogisticRegression(); lr.fit(X_train, y_train)
dt = DecisionTreeClassifier(max_depth=5); dt.fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100); rf.fit(X_train, y_train)
gb = GradientBoostingClassifier(); gb.fit(X_train, y_train)
```
**What it does:** Trains four different ML algorithms on the same data so we can compare which one performs best:
- **Logistic Regression** — Simple linear model
- **Decision Tree** — Tree-based splits
- **Random Forest** — Ensemble of 100 decision trees
- **Gradient Boosting** — Sequential tree boosting

### Step 26 — Compare Accuracy
```python
print("Logistic Regression:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Gradient Boosting:", accuracy_score(y_test, y_pred_gb))
```
**What it does:** Prints the accuracy of each model on the test set to see which one predicted the most pit stops correctly.

---

## PHASE 6: IMPROVED ML MODEL (Full Dataset)

### Step 27 — Load Lap Times for Real Pit/No-Pit Data
```python
lap_times = pd.read_csv("lap_times.csv")
pit_stops['pit'] = 1

df_full = lap_times.merge(
    pit_stops[['raceId','driverId','lap','pit']],
    on=['raceId','driverId','lap'],
    how='left'
)
df_full['pit'] = df_full['pit'].fillna(0)
```
**What it does:** This is a major improvement. Instead of only having pit stop rows, we now load ALL laps (including laps where drivers did NOT pit). We mark pit laps as `1` and non-pit laps as `0`. This gives us a realistic, balanced dataset for training.

### Step 28 — Engineer Advanced Features
```python
df_balanced['race_phase'] = pd.cut(df_balanced['lap_ratio'], bins=[0,0.3,0.7,1], labels=[0,1,2]).astype(int)
df_balanced['driver_pit_rate'] = df_balanced.groupby('driverId')['pit'].transform('mean')
df_balanced['race_pit_rate'] = df_balanced.groupby('raceId')['pit'].transform('mean')
df_balanced['tire_race'] = df_balanced['compound'] * df_balanced['lap_ratio']
```
**What it does:** Creates powerful new features:
- `race_phase` — Categorizes the race into early (0), mid (1), and late (2)
- `driver_pit_rate` — Each driver's historical pit frequency (aggressive vs conservative)
- `race_pit_rate` — How many pit stops typically happen at this specific race
- `tire_race` — Interaction between tire compound and race progress

### Step 29 — Intelligent Tire Assignment
```python
def assign_tire(row):
    if row['is_rain'] == 1:
        return 'wet' if np.random.rand() > 0.5 else 'intermediate'
    else:
        if row['lap_ratio'] < 0.3: return 'soft'
        elif row['lap_ratio'] < 0.7: return 'medium'
        else: return 'hard'
```
**What it does:** Simulates realistic tire compound choices based on weather and race phase (soft tires early, hard tires late, wet/intermediate tires in rain).

---

## PHASE 7: FINAL MODEL TRAINING & COMPARISON

### Step 30 — Train 5 Models Including XGBoost
```python
from xgboost import XGBClassifier

lr = LogisticRegression(max_iter=1000, class_weight='balanced')
dt = DecisionTreeClassifier(max_depth=8, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight={0:1, 1:20})
gb = GradientBoostingClassifier(n_estimators=200)
xgb = XGBClassifier(n_estimators=200, scale_pos_weight=10, eval_metric='logloss')
```
**What it does:** Trains 5 final models with tuned hyperparameters. `class_weight='balanced'` and `scale_pos_weight=10` help the models pay extra attention to the rare pit stop events.

### Step 31 — Full Model Comparison
```python
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, recall_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
print("Recall:", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
**What it does:** Evaluates all 5 models using multiple metrics:
- **Accuracy** — Overall correctness
- **ROC AUC** — How well the model distinguishes between pit and no-pit
- **Recall** — How many actual pit stops the model correctly catches
- **Classification Report** — Precision, recall, and F1-score breakdown

### Step 32 — Select Best Model
```python
final_model = gb  # Gradient Boosting was selected as the best performer
```
**What it does:** After comparing all 5 models, Gradient Boosting was selected as the final model based on the best balance of accuracy and recall.

### Step 33 — Threshold Tuning
```python
y_prob = final_model.predict_proba(X_test)[:,1]
y_pred_04 = (y_prob > 0.4).astype(int)
```
**What it does:** Instead of using the default 0.5 threshold, we tested lowering it to 0.4 to catch more pit stops (higher recall at the cost of slightly lower precision).

---

## PHASE 8: PREDICTION FUNCTION

### Step 34 — Build the Predict Function
```python
def predict_pit(lap_ratio, driver_freq, compound, is_rain,
                race_phase, driver_pit_rate, race_pit_rate):

    sample = pd.DataFrame([{
        'lap_ratio': lap_ratio, 'driver_freq': driver_freq,
        'compound': compound, 'is_rain': is_rain,
        'race_phase': race_phase, 'driver_pit_rate': driver_pit_rate,
        'race_pit_rate': race_pit_rate
    }])

    pred = final_model.predict(sample)
    prob = final_model.predict_proba(sample)

    no_pit = prob[0][0] * 100
    pit = prob[0][1] * 100

    if pit > 60: print("Strong recommendation: PIT")
    elif pit > 40: print("Possible PIT (uncertain)")
    else: print("Stay out (NO PIT)")
```
**What it does:** Creates a reusable function that takes real-time race conditions as input and outputs a pit stop recommendation with probability percentages.

### Step 35 — Test Predictions
```python
predict_pit(0.5, 60, 1, 0, 1, 0.3, 0.05)   # Mid-race, dry, medium tires
predict_pit(0.2, 30, 2, 1, 0, 0.3, 0.06)   # Early race, rain, hard tires
predict_pit(0.9, 60, 2, 0, 2, 0.3, 0.05)   # Late race, dry
predict_pit(0.55, 70, 3, 1, 1, 0.6, 0.15)  # Mid-race, rain, aggressive driver
```
**What it does:** Tests the model with different race scenarios to verify it gives sensible recommendations.

---

## PHASE 9: DATA WAREHOUSE & DEPLOYMENT

### Step 36 — Store Data in SQLite Warehouse
```python
import sqlite3
conn = sqlite3.connect("f1_warehouse.db")
df_full.to_sql("race_data", conn, if_exists="replace", index=False)
```
**What it does:** Saves the entire processed dataset into a local SQLite database file for persistent storage and SQL querying.

### Step 37 — Run SQL Queries on the Warehouse
```python
query = """
SELECT race_phase, AVG(pit) as avg_pit
FROM race_data
GROUP BY race_phase
"""
print(pd.read_sql(query, conn))
```
**What it does:** Runs SQL queries directly against the warehouse to analyze pit rates by race phase, by driver, and by weather conditions.

### Step 38 — Save the Trained Model
```python
import joblib
joblib.dump(final_model, "model.pkl")
```
**What it does:** Saves the trained Gradient Boosting model to a `.pkl` file so it can be loaded later without retraining (used by the Streamlit dashboard `app.py`).

### Step 39 — Streamlit Dashboard Prototype
```python
import streamlit as st
st.title("🏎 F1 Pit Stop Predictor")
lap_ratio = st.slider("Lap Ratio", 0.0, 1.0, 0.5)
# ... (interactive sliders and buttons)
if st.button("Predict"):
    pred = final_model.predict(sample)
    st.write("Prediction:", pred)
```
**What it does:** Builds an interactive web dashboard using Streamlit where users can adjust race parameters with sliders and get real-time pit stop predictions. This is the foundation for the full `app.py` dashboard.

---

## 📋 SUMMARY OF LIBRARIES USED

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, merging, manipulation |
| `numpy` | Numerical operations, random simulation |
| `matplotlib` | Charts and histograms |
| `seaborn` | Advanced statistical plots (boxplots) |
| `scikit-learn` | ML models, train/test split, metrics |
| `xgboost` | XGBoost classifier |
| `fastf1` | Real-time F1 telemetry data |
| `joblib` | Saving/loading trained models |
| `sqlite3` | Data warehousing |
| `streamlit` | Interactive web dashboard |

## 🔧 INSTALL COMMAND
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost fastf1 joblib notebook streamlit
```
