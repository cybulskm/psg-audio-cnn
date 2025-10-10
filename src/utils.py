def get_top_features(importances, channels, percentage):
    """Get top features based on importance scores."""
    num_features = int(len(importances) * percentage)
    top_features = sorted(importances, key=lambda x: x[1], reverse=True)[:num_features]
    return [feature for feature, _ in top_features]

def log_results(results):
    """Log results to a file or console."""
    for key, value in results.items():
        print(f"{key}: {value}")