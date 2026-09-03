# Classification of Network Traffic in the UNSW-NB15 Data Set

Source code and report for CS741 Assignment 2: a comparison of k-nearest neighbours and a classification tree on an attack category prediction task, using a version of the UNSW-NB15 network traffic data set.

The full write-up, including methodology, results, and statistical comparison, is in [`report/main.tex`](report/main.tex) (compiles to the submitted PDF).

## Repository structure

```
code/
  eda.py                    exploratory data analysis (missing values, class distribution, etc.)
  correlation_outliers.py   correlation and outlier analysis, produces the correlation heatmap and boxplots
  preprocessing.py          shared data loading/cleaning and the two algorithm-specific pipelines
  run_knn.py                grid search, cross-validation, and evaluation for k-nearest neighbours
  run_tree.py                grid search, cross-validation, and evaluation for the classification tree
  make_figures.py           builds every figure and the statistical comparison table from the results

results/    every number produced by the scripts above, saved as CSV/JSON
figures/    every chart, saved as PDF
report/     the LaTeX report source
assignment2/ course-supplied reference material (feature descriptions, category mapping)
```

The raw data set (`assignment2/networkTraffic.csv`) is not included in this repository, since it is large and supplied by the course rather than freely redistributable. Place a copy at that path before running any script.

## Reproducing the results

Requires Python 3 with pandas, numpy, scikit-learn, scipy, matplotlib, and seaborn.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

Then, in order:

```bash
python3 code/eda.py
python3 code/correlation_outliers.py
python3 code/run_knn.py
python3 code/run_tree.py
python3 code/make_figures.py
```

Note: the scripts currently use absolute file paths (`/home/john/cs741/Assignment2/...`) rather than paths relative to the repository. Either clone this repository to that exact path, or update `DATA_PATH`, `RESULTS_DIR`, and `FIG_DIR` at the top of each script to match your own machine.

## Building the report

The report is written in LaTeX (IEEE conference format) and compiles with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd report
tectonic main.tex
```

## Summary of findings

Both classification algorithms were tuned through a control parameter search under stratified cross-validation, then compared under a repeated stratified partition across four performance measures. The classification tree reached a higher macro-averaged F1 score and balanced accuracy than k-nearest neighbours, a margin found to be statistically significant using the Wilcoxon signed-rank test, while k-nearest neighbours reached a marginally higher plain accuracy. Full details, evidence, and discussion are in the report.
