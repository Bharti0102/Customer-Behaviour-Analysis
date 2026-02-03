import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Step 1: Load Excel
# --------------------------
df = pd.read_excel('Online Retail(AutoRecovered).xlsx')
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --------------------------
# Step 2: Feature Engineering
# --------------------------
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Aggregate per customer
customer_agg = df.groupby('CustomerID').agg({
    'InvoiceDate': [np.min, np.max, 'nunique'],
    'Quantity': ['sum', 'mean'],
    'UnitPrice': ['mean', 'max']
})
customer_agg.columns = ['FirstPurchase', 'LastPurchase', 'InvoiceCount',
                        'TotalQuantity', 'AvgQuantity', 'AvgPrice', 'MaxPrice']
customer_agg = customer_agg.reset_index()

# Compute Recency (for target only)
customer_agg['Recency'] = (snapshot_date - customer_agg['LastPurchase']).dt.days

# --------------------------
# Step 3: Define Churn
# --------------------------
customer_agg['Churn'] = (customer_agg['Recency'] > 180).astype(int)

# --------------------------
# Step 4: Prepare Features
# --------------------------
features = ['InvoiceCount', 'TotalQuantity', 'AvgQuantity', 'AvgPrice', 'MaxPrice']
X = customer_agg[features].values
y = customer_agg['Churn'].values.reshape(-1,1)

# Normalize features
for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

# Add bias term
X = np.hstack((np.ones((X.shape[0],1)), X))

# --------------------------
# Step 5: Stratified Train/Test Split
# --------------------------
churn_idx = np.where(y.flatten()==1)[0]
active_idx = np.where(y.flatten()==0)[0]

np.random.seed(42)
np.random.shuffle(churn_idx)
np.random.shuffle(active_idx)

train_churn = churn_idx[:int(0.8*len(churn_idx))]
test_churn = churn_idx[int(0.8*len(churn_idx)):]
train_active = active_idx[:int(0.8*len(active_idx))]
test_active = active_idx[int(0.8*len(active_idx)):]

train_idx = np.concatenate([train_churn, train_active])
test_idx = np.concatenate([test_churn, test_active])

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# --------------------------
# Step 6: Oversample Minority Class (Churn)
# --------------------------
churn_train = X_train[y_train.flatten()==1]
y_churn_train = y_train[y_train.flatten()==1]

active_train = X_train[y_train.flatten()==0]
y_active_train = y_train[y_train.flatten()==0]

# Repeat churn to match active class count
repeat_times = int(len(active_train)/len(churn_train)) - 1
X_churn_oversampled = np.vstack([churn_train] + [churn_train]*repeat_times)
y_churn_oversampled = np.vstack([y_churn_train] + [y_churn_train]*repeat_times)

X_train_bal = np.vstack([active_train, X_churn_oversampled])
y_train_bal = np.vstack([y_active_train, y_churn_oversampled])

# Shuffle training data
shuffle_idx = np.arange(len(X_train_bal))
np.random.shuffle(shuffle_idx)
X_train_bal, y_train_bal = X_train_bal[shuffle_idx], y_train_bal[shuffle_idx]

# --------------------------
# Step 7: Logistic Regression Functions
# --------------------------
def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_loss(y, y_pred, weights=None, l2_lambda=0.0):
    m = len(y)
    loss = -1/m*np.sum(y*np.log(y_pred+1e-15) + (1-y)*np.log(1-y_pred+1e-15))
    if weights is not None:
        loss += (l2_lambda/(2*m))*np.sum(weights**2)
    return loss

def logistic_regression_train(X, y, lr=0.1, epochs=2000, l2_lambda=0.01):
    n_features = X.shape[1]
    weights = np.zeros((n_features,1))
    for i in range(epochs):
        z = np.dot(X, weights)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / len(y)
        gradient += (l2_lambda / len(y)) * weights
        weights -= lr*gradient
        
        if i % 200 == 0:
            loss = compute_loss(y, y_pred, weights, l2_lambda)
            print(f'Epoch {i}, Loss: {loss:.4f}')
    return weights

# --------------------------
# Step 8: Train Model
# --------------------------
weights = logistic_regression_train(X_train_bal, y_train_bal, lr=0.1, epochs=2000, l2_lambda=0.01)

# --------------------------
# Step 9: Predict
# --------------------------
def predict(X, weights, threshold=0.4):  # lower threshold to capture minority class
    probs = sigmoid(np.dot(X, weights))
    return (probs>=threshold).astype(int), probs

y_pred_test, probs_test = predict(X_test, weights, threshold=0.4)

# --------------------------
# Step 10: Confusion Matrix
# --------------------------
def confusion_matrix_manual(y_true, y_pred):
    TP = np.sum((y_true==1) & (y_pred==1))
    TN = np.sum((y_true==0) & (y_pred==0))
    FP = np.sum((y_true==0) & (y_pred==1))
    FN = np.sum((y_true==1) & (y_pred==0))
    return TP, TN, FP, FN

TP, TN, FP, FN = confusion_matrix_manual(y_test, y_pred_test)
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP+1e-15)
recall = TP/(TP+FN+1e-15)
f1_score = 2*(precision*recall)/(precision+recall+1e-15)

print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}')

# --------------------------
# Step 11: Visualizations
# --------------------------
conf_matrix = np.array([[TN, FP],[FN, TP]])
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0','Pred 1'], yticklabels=['Actual 0','Actual 1'])
plt.title('Confusion Matrix')
plt.show()


# --------------------------
# Step 12: ROC Curve for Manual Logistic Regression
# --------------------------
from sklearn.metrics import roc_curve, auc

# True labels and predicted probabilities
y_true = y_test.flatten()
y_scores = probs_test.flatten()

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')  # Random chance line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Manual Logistic Regression')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
