def compute_feature_importance(X_train, y_train, channels):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_importance = sorted(zip(channels, importances), key=lambda x: x[1], reverse=True)

    return feature_importance

def select_top_features(feature_importance, percentage):
    top_n = int(len(feature_importance) * percentage)
    return feature_importance[:top_n]

def get_feature_importance_and_select(X_train, y_train, channels):
    feature_importance = compute_feature_importance(X_train, y_train, channels)
    
    top_25_percent = select_top_features(feature_importance, 0.25)
    top_50_percent = select_top_features(feature_importance, 0.50)

    return top_25_percent, top_50_percent