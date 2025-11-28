import numpy as np
from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.svm import SVC

from myinputdatasetXD import myinputdatasetXD

# ======================
# 1. FILTERS từ ITMO_FS
# ======================
from ITMO_FS.filters.univariate import (
    f_ratio_measure,         # Fisher / F-score
    gini_index,
    su_measure,              # Symmetric Uncertainty
    spearman_corr,
    pearson_corr,
    fechner_corr,
    kendall_corr,
    reliefF_measure,
    chi2_measure,            # cần X >= 0
    information_gain,
)

UNIVARIATE_FILTERS = OrderedDict([
    ("FILTER_f_ratio",    f_ratio_measure),
    ("FILTER_gini",       gini_index),
    ("FILTER_sym_uncert", su_measure),
    ("FILTER_spearman",   spearman_corr),
    ("FILTER_pearson",    pearson_corr),
    ("FILTER_fechner",    fechner_corr),
    ("FILTER_kendall",    kendall_corr),
    ("FILTER_reliefF",    reliefF_measure),
    ("FILTER_chi2",       chi2_measure),
    ("FILTER_info_gain",  information_gain),
])


def run_all_filters(X_train_raw, y_train, k=20):
    """
    Chạy tất cả filter univariate trên TẬP TRAIN.
    Trả về:
      - scores[name] : vector score (n_features,)
      - selected[name]: index top-k feature theo |score|
    """
    scores = {}
    selected = {}
    y_train = np.asarray(y_train).ravel()

    for name, func in UNIVARIATE_FILTERS.items():
        X_input = X_train_raw

        # chi2 yêu cầu dữ liệu không âm
        if "chi2" in name:
            min_val = X_train_raw.min()
            if min_val < 0:
                X_input = X_train_raw - min_val + 1e-9

        try:
            s = func(X_input, y_train)
            s = np.asarray(s).ravel()
        except Exception as e:
            print(f"[WARN] Filter {name} failed: {e}")
            continue

        scores[name] = s
        idx = np.argsort(-np.abs(s))[:k]
        selected[name] = idx

    return scores, selected


# ==========================
# 2. EMBEDDED (sklearn)
# ==========================

def embedded_l1_logreg(X_train_raw, y_train):
    clf = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=200,
    )
    clf.fit(X_train_raw, y_train)
    w = np.abs(clf.coef_).mean(axis=0)
    return w


def embedded_rf_importance(X_train_raw, y_train,
                           n_estimators=200, random_state=0):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_raw, y_train)
    return rf.feature_importances_


EMBEDDED_METHODS = OrderedDict([
    ("EMB_L1_LogReg", embedded_l1_logreg),
    ("EMB_RF_importance", embedded_rf_importance),
])


def run_embedded_methods(X_train_raw, y_train, k=20):
    scores = {}
    selected = {}
    y_train = np.asarray(y_train).ravel()

    for name, func in EMBEDDED_METHODS.items():
        try:
            w = func(X_train_raw, y_train)
            w = np.asarray(w).ravel()
        except Exception as e:
            print(f"[WARN] Embedded {name} failed: {e}")
            continue

        scores[name] = w
        idx = np.argsort(-np.abs(w))[:k]
        selected[name] = idx

    return scores, selected


# ==========================
# 3. WRAPPER (RFE, SFS)
# ==========================

def wrapper_rfe_svm(X_train_raw, y_train, k=20):
    estimator = SVC(kernel="linear")
    selector = RFE(estimator, n_features_to_select=k, step=0.1)
    selector.fit(X_train_raw, y_train)
    idx = np.where(selector.support_)[0]
    return idx


def wrapper_sfs_svm_forward(X_train_raw, y_train, k=20, cv=5):
    estimator = SVC(kernel="linear")
    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select=k,
        direction="forward",
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    sfs.fit(X_train_raw, y_train)
    idx = np.where(sfs.get_support())[0]
    return idx


WRAPPER_METHODS = OrderedDict([
    ("WRAP_RFE_SVM", wrapper_rfe_svm),
    ("WRAP_SFS_SVM", wrapper_sfs_svm_forward),
])


def run_wrapper_methods(X_train_raw, y_train, k=20, cv=5):
    selected = {}
    y_train = np.asarray(y_train).ravel()

    for name, func in WRAPPER_METHODS.items():
        try:
            if "SFS" in name:
                idx = func(X_train_raw, y_train, k=k, cv=cv)
            else:
                idx = func(X_train_raw, y_train, k=k)
        except Exception as e:
            print(f"[WARN] Wrapper {name} failed: {e}")
            continue

        selected[name] = np.asarray(idx, dtype=int)

    return selected


# ==========================
# 4. EVALUATION: KNN hold-out
# ==========================

def evaluate_with_knn_holdout(
    X_train_raw, y_train, X_test_raw, y_test,
    selected, n_neighbors=5
):
    """
    selected: dict[name] -> index array
    """
    results = {}

    for name, idx in selected.items():
        # scale theo subset feature
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw[:, idx])
        X_test = scaler.transform(X_test_raw[:, idx])

        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

    return results