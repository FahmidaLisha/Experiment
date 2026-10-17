# Methodology

## 1. Refined Research Question

> Can daily screen time, sleep, late-night use, and content-related habits (passive vs. active use, dominant content type, social-comparison triggers) predict a university-age user's GAD-7 anxiety score? Do the content-related variables add predictive value beyond screen time and sleep alone?

My broader interest is mental health harm from over-consuming low-quality content. No column in the dataset measures "content quality" directly, so I use three proxies: **passive use** (scrolling without interacting), **dominant content type**, and **social-comparison triggers**. The second part of the question tests whether these proxies matter more than the simple amount of time spent online.

## 2. Dataset Description

**Source:** `data/social_media_mental_health.csv` (a Kaggle-style survey dataset on social media habits and mental health).

**Size:** 8,000 rows (one per user) and 15 columns. There are no missing values and no duplicate rows.

**Target:** `GAD_7_Score`, the Generalized Anxiety Disorder 7-item scale. It is an integer from 0 to 21, where higher means more anxiety. The observed mean is about 7.4 and the standard deviation is about 4.6. This makes the task a **regression** problem.

**Features:**

| Feature | Type | Notes |
|---|---|---|
| `Age` | numeric | 18-22 only |
| `Gender` | categorical | Male / Female |
| `Primary_Platform` | categorical | 7 platforms (TikTok, Instagram, Twitter/X, YouTube, LinkedIn, Snapchat, Facebook) |
| `Daily_Screen_Time_Hours` | numeric | 0.5 to about 11.3 hours |
| `Dominant_Content_Type` | categorical | 6 types (e.g., Lifestyle/Fashion, Gaming, News/Politics) |
| `Activity_Type` | categorical | Active / Passive |
| `Late_Night_Usage` | binary | 1 = uses the platform late at night |
| `Social_Comparison_Trigger` | binary | 1 = content triggers social comparison |
| `Sleep_Duration_Hours` | numeric | 3 to about 11.6 hours |

**Columns I will not use as features:**

- `User_ID` is only an identifier.
- `GAD_7_Severity` is a category computed directly from the target, so using it would be **target leakage**.
- `PHQ_9_Score` and `PHQ_9_Severity` are a second mental health outcome (depression), measured at the same time as the target. They are not predictors of anxiety.
- `User_Archetype` is a pre-assigned user segment. Its "Passive Scroller" group matches `Activity_Type == Passive` exactly, and the other labels appear to summarize the usage and outcome patterns I want to model. I drop it to avoid redundancy and circularity.

**Limitations:**

- The data is **self-reported and cross-sectional**, so any relationship is correlational. I cannot claim that screen time *causes* anxiety. Anxious people may also use phones more.
- The sample covers only **ages 18-22** and **two genders**, so results will not generalize to other age groups.
- The dataset is unusually clean: no missing values, near-perfectly balanced categories, and strong, tidy relationships. This suggests it may be **simulated or synthetic**, so real-world data would likely give weaker results.
- "Content quality" is not measured directly, so the proxies are imperfect.
- The platform is each user's *primary* platform only, so cross-platform use is not captured.

## 3. Data Cleaning Plan

1. **Check for missing values and duplicates** (`isna().sum()`, `duplicated()`). Drop duplicate rows or impute only if any are found.
2. **Validate ranges:** sleep and screen time must be between 0 and 24 hours, and their sum must not exceed 24. Age must be a plausible adult age. GAD-7 must be between 0 and 21.
3. **Validate categories:** strip whitespace, check spelling and capitalization consistency, and confirm that binary columns contain only 0 and 1.
4. **Check outliers** with the IQR rule on screen time and sleep. Keep values that are plausible (for example, 11 hours of screen time), and remove a row only if it is impossible.
5. **Remove leakage and redundant columns** as listed above, after confirming with a cross-tabulation that the severity labels are derived from the score and that archetype duplicates activity type.
6. **Check data types** and convert them where needed.

## 4. Feature Engineering Plan

- **One-hot encode** `Gender`, `Primary_Platform`, `Dominant_Content_Type`, and `Activity_Type` (dropping the first level for the linear model).
- **Standardize** numeric columns (`StandardScaler`). This is required for the linear model, and harmless for tree models.
- **New features**, with thresholds fixed in advance from common guidance rather than tuned on the data:
  - `Screen_Sleep_Ratio` = screen time / sleep hours
  - `Heavy_User` = 1 if screen time is at least 6 hours per day
  - `Short_Sleep` = 1 if sleep is under 6 hours
  - `LateNight_x_Screen` = late-night flag x screen time (heavy use late at night)
- **Split first, then fit transformers on the training set only** (80/20 split, `random_state=42`), using a scikit-learn `Pipeline` so that scaling and encoding never see test data.

## 5. Models and Why

1. **Mean-predictor baseline (`DummyRegressor`)** is a sanity check. Any real model must beat it.
2. **Linear Regression** is a simple, interpretable baseline. Its coefficients show the direction and size of each effect, and it gives a fair first test of whether simple additive effects are enough.
3. **Random Forest Regressor** can capture non-linear effects and interactions (for example, heavy use that happens late at night) without manual specification. It is robust to outliers and needs little preprocessing. I tune `max_depth` and `min_samples_leaf` with 5-fold cross-validation on the training set only.

If Random Forest does not beat Linear Regression, that is itself a finding: it would suggest the relationships are mostly linear.

## 6. Evaluation Metrics and Why

- **RMSE** (primary). It is in the same units as the GAD-7 score and penalizes large errors more than small ones, which matters when missing a severe case costs more than a small miss.
- **MAE**, which gives the average error in score points and is easy to explain.
- **R²**, which shows the share of variance in anxiety explained, and lets me compare against the mean baseline (R² of about 0).
- **5-fold cross-validated R²** on the training data, to check that the result is not an accident of one split.

I use regression metrics rather than accuracy or F1 because the target is a continuous score. Finally, I will run a **feature-group comparison** (usage and sleep only vs. usage plus content variables) to answer the second half of the research question.
