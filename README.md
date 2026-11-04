# Classifying stars, galaxies, and quasars

How well can four standard classifiers distinguish stars, galaxies, and quasars from telescope survey measurements? This Python project compares random forest, logistic regression, a linear support vector machine, and Gaussian naive Bayes on the same held-out observations from the Sloan Digital Sky Survey.

The random forest classified **19,512 of 20,000 held-out objects correctly (97.56%)**. Its most common mistake was predicting a galaxy for a quasar: it did this for 247 of the 3,797 quasars in the test set.

| Model | Correct predictions out of 20,000 | Accuracy |
| --- | ---: | ---: |
| Random forest | 19,512 | 97.6% |
| Linear support vector machine | 17,064 | 85.3% |
| Logistic regression | 16,805 | 84.0% |
| Gaussian naive Bayes | 13,438 | 67.2% |

All four models trained on the same 80,000 observations. These results come from the command below with Python 3.12 and scikit-learn 1.9.1.

## Run the comparison

1. Download and extract `star_classification.csv` from the [Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17). Put the CSV in a `data` folder at the root of the repository, or use its full path in the command below.
2. From the repository root, use a Python environment with the packages in `requirements.txt`:

   ```sh
   python -m pip install -r requirements.txt
   python scripts/classify.py data/star_classification.csv
   ```

The script prints precision, recall, and F1 for each class, followed by a confusion matrix whose rows are the observed classes and columns are the predictions. It trains all four models on one 80/20 split (seed 42). The linear support vector machine takes the longest.

## What the models use

The inputs include sky position, brightness in five wavelength bands, redshift, and survey identifiers. The code adds differences between adjacent brightness bands and bins the sky coordinates. It learns replacement values, scaling, and bin boundaries from the training observations before applying them to the held-out observations.

The results describe a random split within one survey catalog. The models have access to redshift and survey metadata, so their scores do not measure how well they would classify an object from an image alone or generalize to another survey.

## Explore the project

- [Classification code](scripts/classify.py) runs the four models on the raw CSV and reports their performance.
- [Exploratory notebook](notebooks/stellar-classification.ipynb) examines the catalog and correlations. GitHub displays the notebook with its saved outputs. To run it locally, install JupyterLab, Matplotlib, and seaborn alongside the project requirements.
- [Original figures](figures/original/) contain the plots from the first version of the project. The original plotting script uses recorded scores. For a fresh comparison, use the classification command above.
- [Practice notebooks](notebooks/practice/) preserve separate preprocessing exercises that used Titanic and COVID datasets.
