import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def prepare_features(X, y):
    """
    Feature preprocessing
    """
    df = pd.DataFrame(X)
    df.fillna(df.median(), inplace=True)

    # Remove constant features
    df = df.loc[:, df.std()>0]

    # Remove highly correlated features
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column]>0.85)]
    df.drop(columns=to_drop, inplace=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled, np.array(y)

def feature_selection(X,y):
    """
    Multi-algorithm feature selection
    """
    rf = RandomForestClassifier(n_estimators=100,max_depth=5,class_weight='balanced')
    rf.fit(X,y)
    importances_rf = rf.feature_importances_

    lr = LogisticRegression(class_weight='balanced',penalty='l2',solver='liblinear')
    lr.fit(X,y)
    importances_lr = np.abs(lr.coef_).flatten()

    gb = GradientBoostingClassifier(alpha=0.1, learning_rate=0.1)
    gb.fit(X,y)
    importances_gb = gb.feature_importances_

    mean_importance = (importances_rf + importances_lr + importances_gb) / 3.0
    idx = np.argsort(-mean_importance)[:20] # top 20

    return idx

def evaluate_model(X,y,model):
    """
    Evaluate model with stratified 5-fold CV
    """
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    metrics = {'accuracy':[], 'precision':[], 'recall':[], 'f1':[], 'roc_auc':[]}

    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        metrics['accuracy'].append(accuracy_score(y_test,y_pred))
        metrics['precision'].append(precision_score(y_test,y_pred))
        metrics['recall'].append(recall_score(y_test,y_pred))
        metrics['f1'].append(f1_score(y_test,y_pred))
        metrics['roc_auc'].append(roc_auc_score(y_test,y_prob))

    summary = {k:np.mean(v) for k,v in metrics.items()}
    return summary
