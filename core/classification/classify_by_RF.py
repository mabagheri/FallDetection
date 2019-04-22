def classify(X_train, y_train, X_test, y_test, configs):

    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    n_trees = configs["classification"]["RF: n_trees"]
    md = configs["classification"]["RF: max_depth"]
    msl = configs["classification"]["RF: min_samples_split"]
    w = eval(configs["classification"]["RF: class_weight"])

    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=md, min_samples_split=msl, class_weight="balanced_subsample") #
    # class_weight="balanced_subsample" OR balanced OR {0: 1, 1: 1, 2: 3, 3: 3}
    # Train model
    clf.fit(X_train, y_train)

    probY_train = clf.predict_proba(X_train)
    y_pred_train = np.argmax(probY_train, axis=1)

    probY = clf.predict_proba(X_test)
    y_pred_test = np.argmax(probY, axis=1)

    confidences = np.max(probY, axis=1)
    # feature_importances = clf.feature_importances_
    # feature_importances[feature_importances < 0.001] = 0

    return y_pred_test, confidences, clf, y_pred_train
