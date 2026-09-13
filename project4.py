import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             roc_auc_score, classification_report,
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"D:/Python/creditcard_python_ml/Datasets/UCI_Credit_Card.csv") 

# Sanity checks
print(df.shape)
print(df.dtypes)
print(df.head())
print(df.columns.tolist())#
print(df['PAY_0'].unique())
print(df["LIMIT_BAL"].describe())
print(df["AGE"].describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df['default.payment.next.month'].value_counts(normalize=True)*100)

# Fixing EDUCATION
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}) # 0, 5, 6 were undocumented; treating them as 'Others' (same as 4)

# Fixing MARRIAGE
df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3}) # 0 was undocumented; treating it as 'Others' (same as 3)

# Verifying the fix
print(df['EDUCATION'].value_counts())
print(df['MARRIAGE'].value_counts())

# Dropping ID column
df.drop(columns=['ID'], inplace=True)

# ------------------------------- PART- 1 EDA ------------------------------------

# STEP-1 UNIVARIATE ANALYSIS

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# 1. Distribution of Credit Limit (LIMIT_BAL): 
plt.figure(figsize=(10, 5))
sns.histplot(df['LIMIT_BAL'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Credit Limit (LIMIT_BAL)')
plt.xlabel('Credit Limit (NT Dollars)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Distribution of Age:
plt.figure(figsize=(10, 5))
sns.histplot(df['AGE'], bins=40, kde=True, color='coral')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Categorical Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# SEX
df['SEX'].map({1: 'Male', 2: 'Female'}).value_counts().plot(
    kind='bar', ax=axes[0], color=['steelblue', 'coral'], edgecolor='black'
)
axes[0].set_title('Distribution of Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

# EDUCATION
df['EDUCATION'].map({1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Others'}) \
    .value_counts().plot(
    kind='bar', ax=axes[1], color='mediumpurple', edgecolor='black'
)
axes[1].set_title('Distribution of Education')
axes[1].set_xlabel('Education Level')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=15)

# MARRIAGE
df['MARRIAGE'].map({1: 'Married', 2: 'Single', 3: 'Others'}).value_counts().plot(
    kind='bar', ax=axes[2], color='mediumseagreen', edgecolor='black'
)
axes[2].set_title('Distribution of Marital Status')
axes[2].set_xlabel('Marital Status')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# Gender: Dataset is female-heavy (60:40), 
# suggesting women were primary credit card adopters in Taiwan 2005.

# Education: ~47% university educated customers — 
# bank's credit approval skews toward educated applicants.

# Marital Status: Single customers slightly outnumber married, 

# STEP - 2 BIVARIATE ANALYSIS

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Default Rate by Gender
gender_default = df.groupby('SEX')['default.payment.next.month'] \
    .mean() * 100
gender_default.index = ['Male', 'Female']
gender_default.plot(kind='bar', ax=axes[0], 
                    color=['steelblue', 'coral'], 
                    edgecolor='black')
axes[0].set_title('Default Rate by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Default Rate (%)')
axes[0].tick_params(axis='x', rotation=0)

# Default Rate by Education
edu_default = df.groupby('EDUCATION')['default.payment.next.month'] \
    .mean() * 100
edu_default.index = ['Graduate', 'University', 'High School', 'Others']
edu_default.plot(kind='bar', ax=axes[1], 
                 color='mediumpurple', 
                 edgecolor='black')
axes[1].set_title('Default Rate by Education')
axes[1].set_xlabel('Education Level')
axes[1].set_ylabel('Default Rate (%)')
axes[1].tick_params(axis='x', rotation=15)

# Default Rate by Marriage
mar_default = df.groupby('MARRIAGE')['default.payment.next.month'] \
    .mean() * 100
mar_default.index = ['Married', 'Single', 'Others']
mar_default.plot(kind='bar', ax=axes[2], 
                 color='mediumseagreen', 
                 edgecolor='black')
axes[2].set_title('Default Rate by Marital Status')
axes[2].set_xlabel('Marital Status')
axes[2].set_ylabel('Default Rate (%)')
axes[2].tick_params(axis='x', rotation=0)

plt.suptitle('Default Rate by Demographic Features', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("=== Default Rates by Demographic ===\n")

print("By Gender:")
print(df.groupby('SEX')['default.payment.next.month']
      .mean().mul(100).round(2))

print("\nBy Education:")
print(df.groupby('EDUCATION')['default.payment.next.month']
      .mean().mul(100).round(2))

print("\nBy Marriage:")
print(df.groupby('MARRIAGE')['default.payment.next.month']
      .mean().mul(100).round(2))

# Most Important Bivariate PLot 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Boxplot - LIMIT_BAL by Default
df.boxplot(column='LIMIT_BAL', 
           by='default.payment.next.month',
           ax=axes[0],
           patch_artist=True)
axes[0].set_title('Credit Limit by Default Status')
axes[0].set_xlabel('Default (0=No, 1=Yes)')
axes[0].set_ylabel('Credit Limit')
axes[0].set_xticklabels(['No Default', 'Default'])

# Boxplot - AGE by Default
df.boxplot(column='AGE',
           by='default.payment.next.month', 
           ax=axes[1],
           patch_artist=True)
axes[1].set_title('Age by Default Status')
axes[1].set_xlabel('Default (0=No, 1=Yes)')
axes[1].set_ylabel('Age')
axes[1].set_xticklabels(['No Default', 'Default'])

plt.suptitle('')
plt.tight_layout()
plt.show()

# STEP - 3 CORRELATION HEATMAP 

plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            annot_kws={'size': 7})

plt.title('Correlation Heatmap of All Features', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

'''
KEY CORRELATION FINDINGS:
1. PAY_0 strongest predictor of default (r=0.32)
2. LIMIT_BAL negatively correlated with default (r=-0.15) → Higher credit limit → lower default risk
3. BILL_AMTs highly correlated with each other (r=0.92-0.95) → Multicollinearity issue → will use avg_bill_amt in features
4. PAY columns correlated (r=0.67-0.82) → Payment behavior is consistent over time
5. AGE and SEX have near-zero correlation with default → Weak predictors as standalone features
'''

# P(Default)
p_default = df['default.payment.next.month'].mean()
print(f"P(Default) = {p_default:.4f} ({p_default*100:.2f}%)")

# P(Default | PAY_0 > 0) meaning payment was delayed
pay_delayed = df[df['PAY_0'] > 0]
p_default_given_delay = pay_delayed['default.payment.next.month'].mean()
print(f"P(Default | Payment Delayed) = {p_default_given_delay:.4f} ({p_default_given_delay*100:.2f}%)")

# P(Default | High Utilization) - Bill > 80% of limit
df['utilization'] = df['BILL_AMT1'] / df['LIMIT_BAL']
high_util = df[df['utilization'] > 0.8]
p_default_high_util = high_util['default.payment.next.month'].mean()
print(f"P(Default | High Utilization >80%) = {p_default_high_util:.4f} ({p_default_high_util*100:.2f}%)")

# P(Default | Low Credit Limit) - Bottom 25%
low_limit = df[df['LIMIT_BAL'] <= 50000]
p_default_low_limit = low_limit['default.payment.next.month'].mean()
print(f"P(Default | Low Credit Limit ≤50K) = {p_default_low_limit:.4f} ({p_default_low_limit*100:.2f}%)")

'''
P(Default) = 22.12%. This is our baseline probability. Without knowing anything about a customer, there's 
a 22.12% chance they will default. Every other probability below is compared against this baseline

P(Default | Payment Delayed) = 50.29%. If a customer has already delayed their payment even once — their 
default probability jumps from 22% to 50%. That's more than 2x the baseline risk.
BUSINESS TERMS: The moment a customer misses a payment, the bank should immediately flag them. Half of all customers 
who delay payment will go on to default. This single signal (PAY_0 > 0) is your most powerful early warning indicator.

P(Default | High Utilization >80%) = 26.73%. Customers using more than 80% of their credit limit have a 26.73% default 
rate — slightly above baseline (22.12%) but not dramatically higher.
BUSINESS TERMS: High utilization alone is a mild risk signal. A customer maxing out their card isn't necessarily going 
to default — but combined with payment delays, risk compounds significantly. This justifies creating interaction features later.

P(Default | Low Credit Limit ≤50K) = 31.79%. Customers with the lowest credit limits (bottom 25%) have a 31.79% default rate — 1.4x
the baseline.
BUSINESS TERMS: The bank already sensed these customers were risky when assigning low limits. But even among low-limit customers, 
nearly 1 in 3 defaults — suggesting limit assignment alone isn't enough to manage risk.

Using Bayes theorem intuitively — P(Default | Payment Delayed) = 50%, which is 2.3x the prior probability of 22%. This posterior update
tells us that PAY_0 is our most informative feature. In a Bayesian framework, each piece of evidence — delay, high utilization, low limit 
— updates our belief about default risk. A customer showing all three signals simultaneously would have compounding risk far above baseline.
'''

def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin

# CI for Credit Limit
mean_limit, lower_limit, upper_limit = confidence_interval(df['LIMIT_BAL'])
print(f"Credit Limit - Mean: {mean_limit:,.0f}")
print(f"95% CI: ({lower_limit:,.0f}, {upper_limit:,.0f})")

print()

# CI for Age
mean_age, lower_age, upper_age = confidence_interval(df['AGE'])
print(f"Age - Mean: {mean_age:.2f}")
print(f"95% CI: ({lower_age:.2f}, {upper_age:.2f})")

print()

# CI for Default Rate
mean_def, lower_def, upper_def = confidence_interval(
    df['default.payment.next.month'])
print(f"Default Rate - Mean: {mean_def:.4f} ({mean_def*100:.2f}%)")
print(f"95% CI: ({lower_def*100:.2f}%, {upper_def*100:.2f}%)")

'''
NoteWith 30,000 samples, our confidence intervals are very tight — for example, the true default rate lies between 21.65% and 
22.59% with 95% confidence. The narrow intervals give us high statistical confidence in our EDA findings before we move to modeling.
'''

# ------------------------------- PART- 2 FEATURE ENGINEERING ------------------------------------

# 1. Credit Utilization Ratio → How much of their credit limit are they using?
df['credit_utilization_ratio'] = df['BILL_AMT1'] / df['LIMIT_BAL']

# 2. Average Bill Amount (last 6 months) → Captures overall debt burden — replaces 6 correlated columns
df['avg_bill_amt'] = df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                          'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']].mean(axis=1)

# 3. Average Payment Amount (last 6 months)
df['avg_pay_amt'] = df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                         'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']].mean(axis=1)

# 4. Repayment Ratio
#If someone owes 10,000 and pays 8,000 → ratio = 0.8 (good)
#If someone owes 10,000 and pays 1,000 → ratio = 0.1 (risky)
df['repayment_ratio'] = df['avg_pay_amt'] / (df['avg_bill_amt'] + 1) # +1 avoids division by zero

# 5. Max Delay (worst payment behavior in 6 months)
df['max_delay'] = df[['PAY_0', 'PAY_2', 'PAY_3',
                       'PAY_4', 'PAY_5', 'PAY_6']].max(axis=1)

# 6. Delay Count (how many months had any delay)
pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
df['delay_count'] = (df[pay_cols] > 0).sum(axis=1)

# 7. Fixing utilization anomalies 
df['credit_utilization_ratio'] = df['credit_utilization_ratio'].clip(0, 1)

# Verify all features created
new_features = ['credit_utilization_ratio', 'avg_bill_amt', 
                'avg_pay_amt', 'repayment_ratio', 
                'max_delay', 'delay_count']
print(df[new_features].describe().round(2)) 

# Visualize new features vs default
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

features = ['credit_utilization_ratio', 'avg_bill_amt',
            'avg_pay_amt', 'repayment_ratio',
            'max_delay', 'delay_count']

titles = ['Credit Utilization Ratio', 'Avg Bill Amount',
          'Avg Payment Amount', 'Repayment Ratio',
          'Max Delay', 'Delay Count']

for i, (feat, title) in enumerate(zip(features, titles)):
    df.boxplot(column=feat,
               by='default.payment.next.month',
               ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_xlabel('Default (0=No, 1=Yes)')
    axes[i].set_xticklabels(['No Default', 'Default'])

plt.suptitle('Engineered Features vs Default Status', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Select features for clustering
# We use engineered + raw features that describe customer behavior
cluster_features = [
    'LIMIT_BAL',                  # Credit limit
    'AGE',                        # Age
    'credit_utilization_ratio',   # How much limit they use
    'avg_bill_amt',               # Average monthly bill
    'avg_pay_amt',                # Average monthly payment
    'repayment_ratio',            # How well they repay
    'max_delay',                  # Worst payment delay
    'delay_count'                 # How many months delayed
]

# Extract clustering data
X_cluster = df[cluster_features].copy()

# Check for any nulls or infinity values
print("Any nulls:", X_cluster.isnull().sum().sum())
print("Any inf:", np.isinf(X_cluster).sum().sum())
print("Shape:", X_cluster.shape)

# Find where the nulls are
print("Nulls per column:")
print(X_cluster.isnull().sum())

# Fix, fill nulls with median of that column. Median is better than mean here because our features are skewed
X_cluster = X_cluster.fillna(X_cluster.median())

# Verify fix
print("Any nulls:", X_cluster.isnull().sum().sum())

# Scale the data
# KMeans and PCA are both distance-based, without scaling, LIMIT_BAL (167,000) would completely dominate AGE (35) just because of magnitude difference
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Scaled data shape:", X_scaled.shape)
print("Mean of scaled data:", X_scaled.mean(axis=0).round(3)) #should be ~0
print("Std of scaled data (should be ~1):", X_scaled.std(axis=0).round(3))

# First run PCA with all 8 components to see how much variance each explains
pca_full = PCA(n_components=8)
pca_full.fit(X_scaled)

# Plot explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))

# Bar chart — individual variance
plt.bar(range(1, 9), explained_variance * 100, 
        color='steelblue', alpha=0.7, 
        label='Individual Variance')

# Line chart — cumulative variance
plt.plot(range(1, 9), cumulative_variance * 100, 
         color='red', marker='o', 
         linewidth=2, label='Cumulative Variance')

# Mark 90% threshold
plt.axhline(y=90, color='green', 
            linestyle='--', label='90% threshold')

plt.title('PCA — Explained Variance by Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.xticks(range(1, 9))
plt.legend()
plt.tight_layout()
plt.show()

# Print exact numbers
print("Variance explained per component:")
for i, (ind, cum) in enumerate(zip(explained_variance, cumulative_variance)):
  print(f"PC{i+1}: {ind*100:.2f}%  |  Cumulative: {cum*100:.2f}%")

# Apply PCA with 6 components
pca = PCA(n_components=6)
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", X_scaled.shape)
print("After PCA shape:", X_pca.shape)
print(f"Variance preserved: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Elbow Method - find best number of clusters
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, 
                    random_state=42, 
                    n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, 
         marker='o', color='steelblue', linewidth=2)
plt.title('Elbow Method — Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-cluster Sum of Squares)')
plt.xticks(K_range)
plt.tight_layout()
plt.show()

# Apply KMeans with K=4
kmeans = KMeans(n_clusters=4, 
                random_state=42, 
                n_init=10)
df['cluster'] = kmeans.fit_predict(X_pca)

# Check cluster sizes
print("Cluster sizes:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster sizes (%):")
print((df['cluster'].value_counts(normalize=True)
       .sort_index() * 100).round(2))

plt.figure(figsize=(10, 7))

colors = ['steelblue', 'coral', 'mediumseagreen', 'mediumpurple']
labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

for i in range(4):
    mask = df['cluster'] == i
    plt.scatter(X_pca[mask, 0], 
                X_pca[mask, 1],
                c=colors[i],
                label=labels[i],
                alpha=0.3,
                s=5)

plt.title('Customer Segments — PCA Visualization (PC1 vs PC2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(markerscale=3)
plt.tight_layout()
plt.show()

# Removing extreme outliers using Z-score. Any row where a feature is more than 3 std devs away
z_scores = np.abs(stats.zscore(X_cluster))
outlier_mask = (z_scores < 3).all(axis=1)

print(f"Rows before outlier removal: {len(df)}")
print(f"Outliers removed: {(~outlier_mask).sum()}")
print(f"Rows after outlier removal: {outlier_mask.sum()}")

# Apply mask
X_cluster_clean = X_cluster[outlier_mask]
df_clean = df[outlier_mask].copy()

# Re-scale
X_scaled_clean = scaler.fit_transform(X_cluster_clean)

# Re-apply PCA
X_pca_clean = pca.fit_transform(X_scaled_clean)

# Re-run KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_pca_clean)

# Check new cluster sizes
print("\nNew Cluster sizes:")
print(df_clean['cluster'].value_counts().sort_index())
print("\nNew Cluster sizes (%):")
print((df_clean['cluster'].value_counts(normalize=True)
       .sort_index() * 100).round(2))


# =============================================
# PREPARE FEATURES AND TARGET
# =============================================

# Features used for modeling

# Step 1 - Define features
model_features = [
    'LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'credit_utilization_ratio',
    'avg_bill_amt',
    'avg_pay_amt',
    'repayment_ratio',
    'max_delay',
    'delay_count'
]

target = 'default.payment.next.month'

# Step 2 - Extract X and y
X = df[model_features].copy()
y = df[target].copy()

# Step 3 - Fix ALL problematic values first
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# Step 4 - Verify completely clean
print("NaN count:", X.isnull().sum().sum())
print("Inf count:", np.isinf(X).sum().sum())
print("Shape:", X.shape)

# Step 5 - train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

# Step 6 - Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train)

print("\nAfter SMOTE:")
print("Training shape:", X_train_smote.shape)
print("Class distribution:")
print(y_train_smote.value_counts())

# =============================================
# MODEL 1 — LOGISTIC REGRESSION
# =============================================
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_smote, y_train_smote)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

# =============================================
# MODEL 2 — RANDOM FOREST
# =============================================
rf = RandomForestClassifier(n_estimators=100, 
                             random_state=42, 
                             n_jobs=-1)
rf.fit(X_train_smote, y_train_smote)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# =============================================
# MODEL 3 — XGBOOST
# =============================================
xgb = XGBClassifier(n_estimators=100,
                     random_state=42,
                     eval_metric='logloss',
                     n_jobs=-1)
xgb.fit(X_train_smote, y_train_smote)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

# =============================================
# COMPARISON TABLE
# =============================================
def get_metrics(y_test, y_pred, y_prob, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }

results = pd.DataFrame([
    get_metrics(y_test, y_pred_lr, y_prob_lr, 'Logistic Regression'),
    get_metrics(y_test, y_pred_rf, y_prob_rf, 'Random Forest'),
    get_metrics(y_test, y_pred_xgb, y_prob_xgb, 'XGBoost')
])

results = results.set_index('Model')
print("\nModel Comparison:")
print(results.round(4))

'''
Logistic Regression → Best Recall (catches more defaulters)
                    → But more false alarms (low precision)

Random Forest       → Best Precision (more accurate predictions)
                    → But misses more defaulters (low recall)

XGBoost            → Best ROC-AUC (best overall discrimination)
                   → Balanced between precision and recall


For a credit card company — which model to choose?

Logistic Regression for this use case because:

Missing a defaulter costs the bank money (false negative is expensive)
Recall of 60% vs 47-49% is a significant difference
Every 1% recall improvement = catching more risky customers early
'''

plt.figure(figsize=(10, 7))

# Plot ROC curve for each model
models = {
    'Logistic Regression': y_prob_lr,
    'Random Forest': y_prob_rf,
    'XGBoost': y_prob_xgb
}

colors = ['steelblue', 'coral', 'mediumseagreen']

for (name, y_prob), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, 
             color=color, 
             linewidth=2,
             label=f'{name} (AUC = {auc:.4f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1], 
         'k--', linewidth=1, 
         label='Random Classifier (AUC = 0.5)')

plt.title('ROC Curve — Model Comparison', 
          fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

'''
Interesting observation — Logistic Regression wins early:
Look at the left side of the plot (FPR 0.0 to 0.2). Logistic Regression (blue) is actually above RF and XGBoost. This means at low false positive rates — 
when you want to be very careful about false alarms — LR catches more defaulters. This further supports choosing LR for conservative credit risk decisions.
'''

# =============================================
# FEATURE IMPORTANCE — XGBoost
# =============================================
feature_importance = pd.DataFrame({
    'Feature': model_features,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], 
         feature_importance['Importance'],
         color='steelblue',
         edgecolor='black')
plt.title('Feature Importance — XGBoost', 
          fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Print top 5
print("Top 5 Most Important Features:")
print(feature_importance.sort_values(
    'Importance', ascending=False).head())

'''
The complete story this tells: The most predictive features were behavioral — max_delay and delay_count, both engineered features — confirming that how a
customer behaves with payments is far more predictive than how much they earn, how old they are, or how much they spend. This is consistent with our EDA 
finding that P(Default | Payment Delayed) = 50% vs baseline of 22%.

EDA found → Payment delay is strongest signal
Feature Engineering created → max_delay, delay_count
Model confirmed → max_delay is #1 feature
Business insight → Monitor payment behavior, not just spend
'''

#A/B Testing: The business scenario:

#Control group → 5% cashback offer
#Treatment group → 10% cashback offer
#Measure → Does treatment group spend more? (using BILL_AMT1 as spend proxy)

# =============================================
# A/B TESTING SETUP
# =============================================

# Randomly assign customers to control and treatment
np.random.seed(42)
df['ab_group'] = np.random.choice(
    ['control', 'treatment'], 
    size=len(df), 
    p=[0.5, 0.5]  # 50-50 split
)

# Check group sizes
print("Group sizes:")
print(df['ab_group'].value_counts())

# Extract spend metric for each group
control_spend = df[df['ab_group'] == 'control']['BILL_AMT1']
treatment_spend = df[df['ab_group'] == 'treatment']['BILL_AMT1']

print("\nControl Group (5% cashback):")
print(f"  Size: {len(control_spend)}")
print(f"  Mean Spend: {control_spend.mean():,.2f}")
print(f"  Std: {control_spend.std():,.2f}")

print("\nTreatment Group (10% cashback):")
print(f"  Size: {len(treatment_spend)}")
print(f"  Mean Spend: {treatment_spend.mean():,.2f}")
print(f"  Std: {treatment_spend.std():,.2f}")

#Interesting — treatment group spent LESS than control.
#This is counterintuitive. The hypothesis test will tell us if this difference is statistically significant or just random noise.

# =============================================
# HYPOTHESIS TEST
# =============================================
# H0: Mean spend is same in both groups
# H1: Mean spend is different between groups
# Significance level: 0.05

# Step 1 — Check normality (sample size is large
# so Central Limit Theorem applies anyway)
print("=" * 50)
print("HYPOTHESIS TEST — A/B Testing")
print("=" * 50)
print("H0: Control mean spend = Treatment mean spend")
print("H1: Control mean spend ≠ Treatment mean spend")
print(f"Significance level: α = 0.05")
print()

# Step 2 — Run two sample t-test
t_stat, p_value = stats.ttest_ind(
    control_spend, 
    treatment_spend,
    equal_var=False  # Welch's t-test
    # doesn't assume equal variance
)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print()

# Step 3 — Decision
if p_value < 0.05:
    print("Result: REJECT H0")
    print("The difference in spend IS statistically significant")
    print("The cashback offer DID impact spending")
else:
    print("Result: FAIL TO REJECT H0")
    print("The difference in spend is NOT statistically significant")
    print("The cashback offer did NOT meaningfully impact spending")

# Step 4 — Confidence Interval on difference
mean_diff = treatment_spend.mean() - control_spend.mean()
se_diff = np.sqrt(
    (control_spend.std()**2 / len(control_spend)) + 
    (treatment_spend.std()**2 / len(treatment_spend))
)
ci_lower = mean_diff - 1.96 * se_diff
ci_upper = mean_diff + 1.96 * se_diff

print()
print(f"Mean difference: {mean_diff:,.2f}")
print(f"95% CI: ({ci_lower:,.2f}, {ci_upper:,.2f})")

#Interview explanation: Our A/B test showed p-value of 0.42, well above the 0.05 significance threshold. The 95% confidence interval on spend difference was
#(-2,349 to +983) — containing zero — meaning we cannot conclude the 10% cashback offer drove meaningfully higher spending. This is a realistic outcome — in 
#practice, cashback offers need to be tested on real experiment data with actual behavioral response, not randomly assigned on historical data. This exercise
#demonstrated the statistical framework: hypothesis setup, t-test, p-value interpretation, and confidence interval analysis."

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1 — Distribution of spend by group
axes[0].hist(control_spend, bins=50, 
             alpha=0.5, color='steelblue', 
             label=f'Control (5% cashback)\nMean: {control_spend.mean():,.0f}')
axes[0].hist(treatment_spend, bins=50, 
             alpha=0.5, color='coral',
             label=f'Treatment (10% cashback)\nMean: {treatment_spend.mean():,.0f}')
axes[0].set_title('Spend Distribution by Group')
axes[0].set_xlabel('Bill Amount (NT Dollars)')
axes[0].set_ylabel('Count')
axes[0].legend()

# Plot 2 — Mean spend with confidence interval
groups = ['Control\n(5% cashback)', 'Treatment\n(10% cashback)']
means = [control_spend.mean(), treatment_spend.mean()]
stds = [control_spend.sem() * 1.96, treatment_spend.sem() * 1.96]

axes[1].bar(groups, means, 
            color=['steelblue', 'coral'],
            edgecolor='black',
            width=0.4)
axes[1].errorbar(groups, means, 
                 yerr=stds,
                 fmt='none',
                 color='black',
                 capsize=5,
                 linewidth=2)
axes[1].set_title(f'Mean Spend by Group\n(p-value = 0.4216 — Not Significant)')
axes[1].set_ylabel('Mean Bill Amount')

plt.suptitle('A/B Test — Cashback Offer Impact on Spending',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================
# DATA DRIFT — Population Stability Index (PSI)
# =============================================

# Simulate drift by splitting dataset
# First 20,000 rows = "2022 training population"
# Last 10,000 rows  = "2025 current population"

df_train_pop = df.iloc[:20000]   # Training population
df_current_pop = df.iloc[20000:] # Current population

print("Training population shape:", df_train_pop.shape)
print("Current population shape:", df_current_pop.shape)

# PSI Function
def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index
    expected = training population
    actual   = current population
    
    PSI < 0.10  → No drift (stable)
    PSI 0.10-0.25 → Moderate drift (monitor)
    PSI > 0.25  → Significant drift (retrain model)
    """
    # Create bins from expected distribution
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )
    
    # Calculate proportions in each bin
    expected_counts = np.histogram(expected, breakpoints)[0]
    actual_counts = np.histogram(actual, breakpoints)[0]
    
    # Add small value to avoid division by zero
    expected_pct = expected_counts / len(expected) + 1e-6
    actual_pct = actual_counts / len(actual) + 1e-6
    
    # PSI formula
    psi = np.sum(
        (actual_pct - expected_pct) * 
        np.log(actual_pct / expected_pct)
    )
    
    return psi

# Calculate PSI for key features
features_to_check = [
    'LIMIT_BAL', 
    'AGE',
    'credit_utilization_ratio',
    'max_delay',
    'delay_count',
    'repayment_ratio'
]

print("\nPSI Results:")
print("=" * 45)
print(f"{'Feature':<25} {'PSI':>8} {'Status':>12}")
print("=" * 45)

psi_results = {}
for feature in features_to_check:
    psi = calculate_psi(
        df_train_pop[feature], 
        df_current_pop[feature]
    )
    psi_results[feature] = psi
    
    if psi < 0.10:
        status = "Stable"
    elif psi < 0.25:
        status = "Monitor"
    else:
        status = "Retrain"
        
    print(f"{feature:<25} {psi:>8.4f} {status:>12}")

print("=" * 45)

# Visualize PSI results
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1 — PSI bar chart
features = list(psi_results.keys())
psi_values = list(psi_results.values())
colors = ['mediumseagreen' if p < 0.10 
          else 'orange' if p < 0.25 
          else 'red' 
          for p in psi_values]

axes[0].barh(features, psi_values, 
             color=colors, edgecolor='black')
axes[0].axvline(x=0.10, color='orange', 
                linestyle='--', 
                linewidth=2, 
                label='Monitor threshold (0.10)')
axes[0].axvline(x=0.25, color='red', 
                linestyle='--', 
                linewidth=2,
                label='Retrain threshold (0.25)')
axes[0].set_title('Population Stability Index (PSI)\nper Feature')
axes[0].set_xlabel('PSI Value')
axes[0].legend()

# Plot 2 — Distribution comparison for most drifted feature
# max_delay had highest PSI (0.0332)
axes[1].hist(df_train_pop['max_delay'], 
             bins=20, alpha=0.6,
             color='steelblue',
             label='Training Population (2022)',
             density=True)
axes[1].hist(df_current_pop['max_delay'],
             bins=20, alpha=0.6,
             color='coral',
             label='Current Population (2025)',
             density=True)
axes[1].set_title('max_delay Distribution\nTraining vs Current Population')
axes[1].set_xlabel('Max Delay (months)')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.suptitle('Data Drift Analysis — PSI Monitoring',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


# =============================================
# TIME SERIES — Monthly Spend Forecasting
# =============================================

# Reshape bill amounts from wide to long format
# Each customer has 6 months of bill data
# BILL_AMT1 = most recent (Sept 2005)
# BILL_AMT6 = oldest (April 2005)

# Create monthly portfolio average spend
# (aggregate all customers by month)

months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
bill_cols = ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4',
             'BILL_AMT3', 'BILL_AMT2', 'BILL_AMT1']

# Calculate average spend per month across all customers
monthly_avg_spend = df[bill_cols].mean()
monthly_avg_spend.index = months

print("Monthly Average Spend:")
print(monthly_avg_spend.round(2))

# Plot the time series
plt.figure(figsize=(12, 5))
plt.plot(months, monthly_avg_spend.values,
         marker='o', color='steelblue',
         linewidth=2, markersize=8)
plt.title('Average Monthly Spend — Portfolio Level',
          fontsize=14, fontweight='bold')
plt.xlabel('Month (2005)')
plt.ylabel('Average Bill Amount (NT Dollars)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================
# TIME SERIES — Features + Forecasting
# =============================================

# Create a proper dataframe
ts_df = pd.DataFrame({
    'month': range(1, 7),
    'month_name': months,
    'avg_spend': monthly_avg_spend.values
})

# Moving averages
ts_df['ma_2'] = ts_df['avg_spend'].rolling(window=2).mean()
ts_df['ma_3'] = ts_df['avg_spend'].rolling(window=3).mean()

# Lag features
ts_df['lag_1'] = ts_df['avg_spend'].shift(1)
ts_df['lag_2'] = ts_df['avg_spend'].shift(2)

# Spend growth rate month over month
ts_df['growth_rate'] = ts_df['avg_spend'].pct_change() * 100

print("Time Series Features:")
print(ts_df.round(2).to_string())

# =============================================
# SIMPLE FORECAST — Month 7 (October)
# =============================================

# Method 1 — Moving Average forecast
ma3_forecast = ts_df['avg_spend'].tail(3).mean()

# Method 2 — Linear Trend forecast
from numpy.polynomial import polynomial as P
x = ts_df['month'].values
y = ts_df['avg_spend'].values
coeffs = np.polyfit(x, y, 1)
trend_forecast = np.polyval(coeffs, 7)

# Method 3 — Growth rate forecast
avg_growth = ts_df['growth_rate'].dropna().mean() / 100
growth_forecast = ts_df['avg_spend'].iloc[-1] * (1 + avg_growth)

print("\n" + "=" * 45)
print("OCTOBER 2005 SPEND FORECAST")
print("=" * 45)
print(f"Method 1 - 3-Month Moving Average: "
      f"{ma3_forecast:,.2f}")
print(f"Method 2 - Linear Trend:           "
      f"{trend_forecast:,.2f}")
print(f"Method 3 - Avg Growth Rate:        "
      f"{growth_forecast:,.2f}")
print(f"\nAverage of all methods:            "
      f"{np.mean([ma3_forecast, trend_forecast, growth_forecast]):,.2f}")

plt.figure(figsize=(12, 6))

# Actual data
plt.plot(ts_df['month'], ts_df['avg_spend'],
         marker='o', color='steelblue',
         linewidth=2, markersize=8,
         label='Actual Spend')

# Moving averages
plt.plot(ts_df['month'], ts_df['ma_2'],
         color='orange', linestyle='--',
         linewidth=1.5, label='2-Month MA')
plt.plot(ts_df['month'], ts_df['ma_3'],
         color='green', linestyle='--',
         linewidth=1.5, label='3-Month MA')

# Forecast point
forecasts = [ma3_forecast, trend_forecast, growth_forecast]
avg_forecast = np.mean(forecasts)

plt.scatter(7, avg_forecast,
            color='red', s=150, zorder=5,
            label=f'Oct Forecast: {avg_forecast:,.0f}')
plt.plot([6, 7],
         [ts_df['avg_spend'].iloc[-1], avg_forecast],
         color='red', linestyle='--', linewidth=2)

# Labels
plt.xticks(range(1, 8),
           ['Apr', 'May', 'Jun', 'Jul',
            'Aug', 'Sep', 'Oct (Forecast)'])
plt.title('Monthly Spend — Trend, Moving Averages & Forecast',
          fontsize=14, fontweight='bold')
plt.xlabel('Month (2005)')
plt.ylabel('Average Spend (NT Dollars)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




