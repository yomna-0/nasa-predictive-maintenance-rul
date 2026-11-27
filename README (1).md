# NASA Predictive Maintenance: Remaining Useful Life (RUL) Prediction

Neural Networks course project (AIE231), Alamein International University.
Predicting how many operating cycles a turbofan engine has left before it fails, from its sensor readings.

\---

## 1\. Problem statement

**User:** a maintenance planner who schedules engine servicing.
**Prediction:** given the last 30 cycles of sensor readings of an engine, predict its **Remaining Useful Life (RUL)**, the number of operating cycles left before failure (a regression task).
**Cost of a wrong prediction:** an error in either direction is costly. Predicting too *little* life triggers an early, unnecessary maintenance and wastes engine life. Predicting too *much* life (a *late* prediction) is worse, because the engine may fail before it is serviced.

## 2\. Dataset

|||
|-|-|
|Name|NASA C-MAPSS turbofan engine degradation simulation, subset **FD001**|
|Content|Run-to-failure sensor data of simulated engines: 1 operating condition, 1 fault mode (HPC degradation)|
|Files|`PM\_train.csv` (20,631 rows, 100 engines), `PM\_test.csv` (13,096 rows, 100 engines), `PM\_truth.csv` (true RUL of the 100 test engines)|
|Columns|engine `id`, `cycle`, 3 operational settings, 21 sensors|
|Source|NASA Prognostics Center of Excellence Data Repository: [https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)|
|Terms|Publicly released by NASA for research. The data owners ask users to cite the source (see [References](#10-references)).|
|Data quality|No missing values and no duplicate rows. The sensor readings are noisy, and 7 of the 24 settings/sensors are constant.|

The raw data is **not stored in this repository**. See [How to run](#6-how-to-run).

## 3\. Task definition and metrics

* **Input:** a window of 30 consecutive cycles x 18 features, shape `(30, 18)`.
* **Output:** one number, the predicted RUL in cycles.
* **Loss:** mean squared error (MSE) for the Neural Networks.
* **Primary metric:** **RMSE** in cycles on the 100 test engines. It penalises large errors, which are the costly ones for maintenance planning.
* **Secondary metrics:** MAE (cycles), R2, and the **NASA asymmetric score**, which penalises *late* predictions more than early ones, matching the cost of errors above. Lower is better for RMSE, MAE and NASA score.
* **Test set:** the last 30 cycles of each test engine, with the true RUL from `PM\_truth.csv`. It is used **once, for final reporting only**. Models are chosen on the validation set.

## 4\. Data preparation (leak-free)

Implemented in `notebooks/02\_data\_preprocessing.ipynb`.

1. **Target:** `RUL = last cycle of the engine - current cycle` for each training row.
2. **Split by engine:** 80 training engines / 20 validation engines (`GroupShuffleSplit`, seed 42). No engine appears in both sets, so no engine leaks across the split.
3. **Constant features removed** (decided on the training engines only): `setting3, s1, s5, s10, s16, s18, s19`. This leaves 18 input features.
4. **Scaling:** `StandardScaler` fitted on the training engines only, then applied to validation and test.
5. **Windows:** sliding windows of 30 cycles over each engine, target = RUL at the last cycle of the window (14,241 training windows). For the test set, only the last window of each engine is used, because the true RUL refers to the end of the observed trajectory.
6. **Export:** the prepared arrays are saved to `prepared\_data.npz`, so that every model starts from the same data.

## 5\. Baseline results (test set, 100 engines)

Implemented in `notebooks/BaselineModels.ipynb`. All three baselines use the flattened window (540 values). Hyperparameters were chosen on the validation set only.

|Model|RMSE|MAE|R2|NASA score|
|-|-|-|-|-|
|Trivial (always predicts the mean training RUL, 94.75)|45.79|36.62|-0.214|73,243|
|Ridge Regression (alpha = 1000)|26.11|22.42|0.605|3,200|
|Random Forest (100 trees)|**25.30**|**18.94**|**0.629**|4,534|

Both learned baselines cut the RMSE by about 43-45% compared with the trivial baseline. Every Neural Network in this project is reported against these numbers. The full table, including validation results, is in `experiments/baseline\_results.csv`.

## 6\. How to run

**Requirements:** Python 3.12 (the version the pinned packages were tested with) and either Google Colab or Jupyter.

```bash
git clone https://github.com/yomna-0/nasa-predictive-maintenance-rul.git
cd nasa-predictive-maintenance-rul
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install jupyterlab
```

1. **Get the data.** Download the FD001 files from the source above and put `PM\_train.csv`, `PM\_test.csv` and `PM\_truth.csv` in a folder named `data/` at the repository root.
2. **Prepare the data.** Run all cells of `notebooks/02\_data\_preprocessing.ipynb`. It creates `prepared\_data.npz` next to the notebook.
3. **Run the baselines.** Run all cells of `notebooks/BaselineModels.ipynb`. It reads `prepared\_data.npz` from the same folder and writes the result tables next to the notebook.

**On Google Colab:** open the notebook from GitHub (File > Open notebook > GitHub), run it, and upload the files when asked. Notebook 02 asks for the three CSV files, and `BaselineModels.ipynb` asks for `prepared\_data.npz`.

All random seeds are fixed (`42`).

## 7\. Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 02\_data\_preprocessing.ipynb   data preparation
│   └── BaselineModels.ipynb          trivial, Ridge and Random Forest baselines
├── experiments/
│   └── baseline\_results.csv          baseline results (validation and test)
├── report/
│   └── section\_baselines.md          report section: baselines
├── data/                             raw data goes here (not in git)
├── configs/  src/  models/  app/     planned
```

## 8\. Project status

|Milestone|Status|
|-|-|
|1. Topic settled (problem, dataset, metric, roles)|Done|
|2. Baselines ready (EDA, leak-free preprocessing, two baselines)|Done|
|3. Neural Networks (3 architectures, hand parameter counts, first curves)|In progress|
|4. Tuning, ablations, 3-seed final result, error analysis|Planned|
|Final: inference pipeline (`predict.py`), demo, report, slides|Planned|

Commands to train, evaluate and run the demo will be added here when those modules are merged.

## 9\. Team

|Member|Name|GitHub|Work package|
|-|-|-|-|
|1|*Rahma Ahmed*|*@ra3944944-crypto*|Data preprocessing|
|2|*Yomna Ayman*|*@yomna-0*|Baseline models|
|3|*Salma Waleed*|*@SalmaWaleed-28*|Neural Network model 1|
|4|*Malak Mahmoud*|*@malakmahmoud0006-cell*|Neural Network models 2 and 3|
|5|*Heba Mohammed*|*@heba-mansour6*|Experiments and optimization|
|6|*Noor Refaat*|*@NoorRefaat2006*|Evaluation and deployment|

## 10\. References

1. A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
2. A. Saxena, K. Goebel, D. Simon and N. Eklund (2008). "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO.

