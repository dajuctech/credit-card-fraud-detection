# How I Built a Machine Learning System to Catch Credit Card Fraud

Every second, millions of credit card transactions happen around the world. Most of them are perfectly normal — someone buying groceries, paying a bill, or ordering online. But hidden among those millions of transactions are a small number of fraudulent ones, made by people who have stolen card details and are trying to spend money that is not theirs.

The question I set out to answer in this project was simple: **can a machine learning model learn the difference between a real transaction and a fraudulent one, and flag the fraud before any damage is done?**

This post walks you through exactly how I built that system — from collecting the data to training six different machine learning models and comparing their results. Whether you are completely new to machine learning or just curious about how fraud detection actually works, I have written this in a way that makes every step easy to follow.

[**View the full project on GitHub →**](https://github.com/dajuctech/credit-card-fraud-detection)

[**Read this post on Medium →**](https://medium.com/p/ee1a53397c72)

---

## The Problem

Fraud detection is not a straightforward classification problem. It has one particular challenge that makes it much harder than most beginner machine learning projects: **the data is extremely imbalanced.**

Think about it this way. Out of every 1,000 credit card transactions, only a tiny handful are fraudulent. In the dataset used for this project, fraudulent transactions make up less than 0.2% of the total records. That means if you built a model that simply predicted "not fraud" for every single transaction, it would be right 99.8% of the time — and completely useless, because it would never catch a single fraudster.

This imbalance problem is the central challenge of this project. Everything — from how we handle the data to which metrics we use to judge our models — is shaped by the fact that fraud is rare.

---

## Step 1: Collecting the Data

The dataset used in this project comes from **Kaggle** and contains real anonymised credit card transactions made by European cardholders over two days in September 2013.

Because of privacy and security regulations, the original transaction features (things like merchant name, location, and card number) cannot be shared publicly. Instead, the dataset providers applied a mathematical technique called **Principal Component Analysis (PCA)** to transform the raw data into 28 anonymised numerical columns, labelled V1 through V28. These columns still contain the patterns the model needs to learn, but the original sensitive information is hidden.

The dataset also contains:
- **Time** — the number of seconds elapsed since the first transaction in the dataset
- **Amount** — the transaction value in euros
- **Class** — the target variable, where 0 means legitimate and 1 means fraudulent

The dataset was downloaded directly inside the notebook using the `gdown` library, which pulls files from Google Drive without any manual downloading.

---

## Step 2: Data Preprocessing

Before training any model, the data needs to be cleaned and prepared. Feeding raw, messy data into a model produces poor results — this is the classic principle of "garbage in, garbage out."

To handle this step in a clean and organised way, all preprocessing was written inside a **Python class** called `DataPreprocessing`. Using a class is a good software engineering habit because it groups related functions together and makes the code reusable. You create the class once with your dataset, and then call individual methods as needed.

### 2.1 Missing Values

The first check was for missing values — rows where data was not recorded. After running `check_missing_values()`, the result showed **zero missing values** across all 31 columns. This meant no filling or imputation was required, and the dataset was clean in that regard.

The class included a `handle_missing_values()` method for cases where missing data is found. For numerical columns, it fills gaps with the **mean** of that column. For categorical columns (text-based data), it fills with the **mode** — the most frequently occurring value.

### 2.2 Data Types

The `check_data_types()` method confirmed that **all columns are numerical**. There were no text columns, which meant no encoding step was needed. This is a direct result of the PCA transformation applied to the original data.

### 2.3 Duplicate Records

Running `check_duplicates()` revealed **1,081 duplicate rows** in the dataset. These are exact copies of existing records that add no new information and could skew the model's learning. They were removed using `handle_duplicates()`.

### 2.4 Summary Statistics

The `get_summary_statistics()` method provided a table of key metrics — minimum, maximum, mean, and standard deviation — for each column. This gives a quick overview of what the data looks like and whether any values seem unusually large or small.

### 2.5 Class Distribution

This is where the imbalance problem became clearly visible. Calling `check_class_distribution()` showed that the overwhelming majority of transactions were labelled 0 (legitimate), while only a tiny fraction were labelled 1 (fraud). The model will need to account for this imbalance during training.

### 2.6 Additional Checks

The preprocessing class also ran checks for:
- **Unique values per column** — how many distinct values each feature contains
- **Skewness** — whether any columns have distributions that lean heavily to one side, which can affect model performance
- **Outliers** — visualised using box plots to spot values that are far outside the normal range
- **Correlation matrix** — a heatmap showing how strongly each feature relates to the others and to the target variable

---

## Step 3: Exploratory Data Analysis (EDA)

Once the data was cleaned, the next step was to explore it visually and statistically to find patterns. This stage is called Exploratory Data Analysis, or EDA, and it is one of the most important parts of any machine learning project. It helps you understand what the data is telling you before you ask a model to make predictions from it.

An `EDA` class was written to handle all of this, with dedicated methods for each type of analysis.

### 3.1 Class Distribution

The `visualize_class_distribution()` method produced a bar chart showing how many legitimate versus fraudulent transactions exist. Visually, the difference is stark — the bar for Class 0 towers over the bar for Class 1. This chart makes it immediately obvious why a naive model that always predicts "not fraud" would appear accurate while being practically worthless.

### 3.2 Transaction Amount by Class

The `visualize_transaction_amount()` method plotted the distribution of transaction amounts separately for fraud and non-fraud cases. One of the interesting findings here is that **fraudulent transactions tend to involve smaller amounts than you might expect.** Fraudsters often test stolen cards with small purchases before attempting larger ones, and larger transactions are more likely to trigger security checks.

### 3.3 Time Distribution by Class

The `visualize_time_distribution()` method revealed how fraud transactions are distributed over time. Fraud does not follow the same daily rhythm as legitimate spending — it can happen at any hour, and the time feature may reveal useful patterns for detection.

### 3.4 Correlation Matrix

The `correlation_matrix()` method produced a heatmap of all feature correlations. Several of the V columns show moderate correlations with the Class column, which confirms they contain signals the model can use to distinguish fraud from legitimate activity.

### 3.5 Outliers

The `visualize_outliers()` method used box plots to display the spread of each feature. Many of the V columns contain outliers — data points far from the centre of the distribution. This is expected, because fraudulent transactions are themselves statistical outliers in terms of transaction behaviour.

### 3.6 Relationship Between Time and Fraud

The `plot_time_vs_fraud()` method visualised how fraudulent transactions are spread across the time axis. This helps identify whether fraud is concentrated at certain times of day, which could be a useful signal.

### 3.7 Relationship Between Amount and Fraud

The `visualize_amount_vs_fraud()` method compared transaction amounts for fraud versus legitimate cases. This visualisation reinforces the earlier finding that fraud transactions tend to cluster at lower amounts.

### 3.8 Pairplot

The `pairplot_for_features()` method was used to visualise the relationship between Amount and Time for fraud versus legitimate transactions. These scatter plots can reveal clusters or separations between the two classes that a model can learn to exploit.

---

## Step 4: Feature Selection and Engineering

### 4.1 Dropping the Time Column

The `Time` column records the number of seconds since the first transaction in the dataset. After analysis, this column was dropped from the feature set. While time patterns could theoretically be informative, the way this feature is constructed — as a raw elapsed second count — makes it difficult for the model to generalise to new data. Dropping it simplifies the input without a significant loss of predictive power.

### 4.2 Defining Features and Target

After dropping Time, the features and target were defined:
- **X (features)** = all columns except Class — the 28 V columns plus Amount
- **y (target)** = the Class column, where 0 = legitimate and 1 = fraud

### 4.3 Normalising the Amount Column

The Amount column has a very different scale to the V columns. It ranges from a few cents to thousands of euros, while the PCA-transformed V columns are already scaled to a standard range. If Amount is left as-is, models that are sensitive to scale — such as SVM and KNN — may give it disproportionate weight.

To fix this, **StandardScaler** was applied to the Amount column. StandardScaler transforms the values so they have a mean of zero and a standard deviation of one. This puts Amount on the same scale as all other features. This normalised version of the dataset was saved separately as `X_scaled` and `y_scaled`.

---

## Step 5: Handling the Class Imbalance

This is the most technically significant step in the entire project. Because fraud accounts for such a tiny fraction of transactions, a model trained on the raw imbalanced data will almost always predict "not fraud" and achieve high accuracy without learning anything genuinely useful.

Three different techniques were applied to address this imbalance, each producing a separate, balanced version of the dataset for model training and comparison.

### 5.1 Oversampling

Oversampling increases the number of fraud examples by **copying them repeatedly** until they match the size of the non-fraud class. This gives the model many more fraud examples to learn from.

The `resample()` function from scikit-learn was used here, duplicating the 491 fraud cases up to 284,315 records to match the legitimate class. The risk with this technique is **overfitting** — the model may memorise the copied examples rather than learning the underlying patterns.

### 5.2 Undersampling

Undersampling takes the opposite approach. Instead of making the minority class bigger, it **shrinks the majority class** by randomly removing non-fraud records until both classes have the same number of examples.

The `RandomUnderSampler` from the imbalanced-learn library was used. After undersampling, both classes had an equal number of records. The trade-off is that a large portion of legitimate transaction data is discarded, which means the model sees less of the normal spending patterns.

### 5.3 SMOTE (Synthetic Minority Oversampling Technique)

SMOTE is the most sophisticated of the three techniques. Rather than simply copying existing fraud examples, SMOTE **creates brand new, artificial fraud examples** by interpolating between real ones.

It works by looking at each fraud example and its nearest fraud neighbours in the feature space, then generating new data points between them. This produces a more varied set of fraud examples, which helps the model generalise better without simply memorising identical copies.

SMOTE was applied using the `SMOTE` class from imbalanced-learn with `sampling_strategy='minority'`, targeting only the minority (fraud) class.

---

## Step 6: Training Six Machine Learning Models

With five versions of the dataset ready — normal, scaled, undersampled, oversampled, and SMOTE — six different machine learning models were trained and evaluated on each combination. This produces a comprehensive comparison of both model strengths and data handling strategies.

Each model was implemented as its own Python class with a `fit_and_evaluate()` method. This method splits the data into training and test sets using an 80/20 split, fits the model on the training portion, and then reports accuracy, confusion matrix, and a full classification report on the held-out test set.

### 6.1 Logistic Regression

Logistic Regression is one of the most fundamental classification algorithms in machine learning. Despite its name, it is used for classification rather than regression. It learns a set of numerical weights for each feature and combines them to produce the probability that a transaction is fraudulent. If the probability exceeds 50%, the transaction is flagged as fraud.

**Conclusion:** Logistic Regression achieved high accuracy on the imbalanced normal dataset, but this is misleading. The model was largely predicting "not fraud" for almost everything. On balanced datasets — undersampled, oversampled, and SMOTE — it performed significantly better at actually detecting fraud cases, which is the metric that truly matters.

### 6.2 Random Forest

Random Forest builds a large collection of individual decision trees and combines their predictions by majority vote. Each tree is trained on a randomly selected subset of the data and features, which reduces the chance of overfitting and makes the overall model more robust and reliable.

**Conclusion:** Random Forest performed strongly across all datasets. On imbalanced data, its recall for fraud cases was lower, but balanced datasets — especially oversampled and SMOTE — significantly improved fraud detection. Random Forest is one of the two clear winners in this comparison.

### 6.3 K-Nearest Neighbours (KNN)

K-Nearest Neighbours is one of the most intuitive algorithms in machine learning. To classify a new transaction, it looks at the K most similar transactions already in the training data and takes a majority vote among them. With K set to 3, the model looks at the three closest neighbours.

**Conclusion:** KNN performed exceptionally well on balanced datasets, particularly oversampled and SMOTE, where it achieved near-perfect fraud detection. Its performance on the raw imbalanced datasets was slightly lower, because the legitimate class dominates the neighbourhood lookups and the rare fraud examples get outvoted.

### 6.4 Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem. It assumes that all features are statistically independent of each other — which is a simplification (hence the word "naive"), but it often works well in practice. The Gaussian variant was used here, which assumes that each feature follows a bell-curve (normal) distribution.

**Conclusion:** Naive Bayes struggled most with the imbalanced datasets, producing low precision and F1 scores for fraud cases. Balanced datasets improved its performance, but it consistently underperformed compared to the other models in this comparison. The independence assumption likely breaks down for this dataset, since the V columns (which are PCA components) carry correlated information.

### 6.5 XGBoost

XGBoost, short for Extreme Gradient Boosting, is one of the most powerful and widely-used machine learning algorithms in the world today. It builds models sequentially, where each new model focuses specifically on correcting the mistakes made by the previous one. This process — called boosting — produces highly accurate predictions by continuously refining the output.

**Conclusion:** XGBoost achieved a perfect accuracy score of 1.0 (100%) across all five dataset types. While this is impressive, the notebook notes that further investigation is needed to confirm whether the model is genuinely learning the underlying patterns or overfitting to the training data. In machine learning, a perfect score is always worth scrutinising carefully before trusting it.

### 6.6 Support Vector Machine (SVM)

Support Vector Machine finds the optimal decision boundary — called a hyperplane — that separates the two classes with the maximum possible margin. The RBF (Radial Basis Function) kernel was used, which allows the model to learn non-linear boundaries and handle more complex patterns.

**Conclusion:** SVM performed well on the normal and scaled datasets but struggled with larger balanced datasets. The oversampled and SMOTE datasets caused extremely long training times — the model could not complete training within a practical time limit. This is a well-known limitation of SVM: its computational cost grows significantly with the size of the dataset, making it unsuitable for large-scale problems without careful tuning or approximation.

---

## Step 7: Model Comparison

After all six models were trained and evaluated, a `ModelComparison` class was used to visualise the accuracy results side by side as a colour-coded heatmap. This made it immediately clear which models performed consistently well and which ones were less reliable.

Here is the full accuracy table across all models and data handling techniques:

| Model | Normal | Scaled | Undersampled | Oversampled | SMOTE |
|---|---|---|---|---|---|
| Logistic Regression | 99.92% | 99.92% | 95.43% | 94.90% | 95.98% |
| Random Forest | 99.97% | 99.97% | 98.98% | 99.26% | 98.67% |
| K-Nearest Neighbours | 99.95% | 99.97% | 94.72% | 99.99% | 99.95% |
| Naive Bayes | 97.78% | 97.78% | 91.77% | 91.71% | 91.88% |
| XGBoost | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| SVM | 99.88% | 99.96% | 83.43% | — | — |

A few important observations stand out from this table:

**High accuracy on the normal dataset does not mean a model is good at fraud detection.** Most models score above 99% on the imbalanced raw dataset simply because they are predicting the dominant class most of the time. Accuracy alone is a poor metric for this type of problem.

**Balanced datasets reveal the real capability of each model.** When trained and tested on balanced data, accuracy scores shift in a way that reflects how well each model is actually distinguishing fraud from legitimate transactions.

**XGBoost and Random Forest are the clear winners.** They perform well across every dataset type. Random Forest is arguably more trustworthy for a production system, as XGBoost's perfect scores need further validation to rule out overfitting.

**SVM is not practical for this problem at scale.** Its inability to train on oversampled and SMOTE datasets within a reasonable time is a real-world constraint that eliminates it from consideration for production fraud detection.

**Naive Bayes is the weakest performer.** While it can serve as a useful baseline, it consistently underperforms compared to the other five models across all dataset types.

---

## Overall Conclusion

In conclusion, **data handling techniques such as scaling, oversampling, and SMOTE significantly improved model performance** for fraud detection. Without addressing the class imbalance, even highly accurate-looking models fail at the only task that matters: correctly identifying fraud.

**XGBoost and Random Forest delivered the best results** across all data handling techniques. For a production fraud detection system, Random Forest would be the safer starting point, as XGBoost's perfect scores require further investigation to confirm they reflect genuine generalisation rather than overfitting.

The project also demonstrated the practical value of structuring code inside Python classes. By encapsulating preprocessing, EDA, and model development into clean, reusable classes, the notebook remains organised and maintainable, and each step can be tested independently.

---

## What I Learned

Building this project taught me several lessons that go beyond the technical steps:

**Accuracy is a misleading metric for imbalanced problems.** A model that says "not fraud" for every transaction looks 99.8% accurate but catches zero fraudsters. Precision, recall, and F1 score tell a far more honest story.

**Data preparation matters more than model choice.** The improvement from training on SMOTE-balanced data versus raw imbalanced data is often larger than the difference between choosing Logistic Regression and Random Forest.

**Perfect scores should make you suspicious.** XGBoost achieving 1.0 across all five datasets is exciting, but it is also a signal worth investigating carefully before deploying to production.

**Good code structure pays off.** Writing each pipeline stage as a Python class made it easy to test different datasets and models without repeating code. This is a habit worth building from the very beginning of any project.

---

## How to Run This Project Yourself

If you want to run this project on your own machine, here is everything you need.

**Install the required libraries:**
```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost gdown termcolor
```

**Steps to run:**
1. Clone the repository from GitHub using `git clone`
2. Open `Credit Card Fraud Detection.ipynb` in Jupyter Notebook or Google Colab
3. Run all cells from top to bottom
4. The dataset will be downloaded automatically via the `gdown` library — no manual download needed

[**View the full project on GitHub →**](https://github.com/dajuctech/credit-card-fraud-detection)

---

*Thank you for reading. If you found this post helpful, please give it a clap on Medium and follow for more machine learning write-ups.*
