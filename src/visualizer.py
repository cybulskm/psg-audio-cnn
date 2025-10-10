import matplotlib.pyplot as plt
def visualize_rfc():
    channel_ranking = [
    {
      "rank": 1,
      "channel": "ECG I",
      "importance": 0.12922185652326731
    },
    {
      "rank": 2,
      "channel": "EEG C4-A1",
      "importance": 0.10522430213135968
    },
    {
      "rank": 3,
      "channel": "EOG ROC-A2",
      "importance": 0.10504052280906777
    },
    {
      "rank": 4,
      "channel": "EOG LOC-A2",
      "importance": 0.1046605688535001
    },
    {
      "rank": 5,
      "channel": "EEG A1-A2",
      "importance": 0.1030968145534642
    },
    {
      "rank": 6,
      "channel": "EEG C3-A2",
      "importance": 0.10003537479888994
    },
    {
      "rank": 7,
      "channel": "Leg 1",
      "importance": 0.09957886051818113
    },
    {
      "rank": 8,
      "channel": "Leg 2",
      "importance": 0.09651934374720879
    },
    {
      "rank": 9,
      "channel": "EMG Chin",
      "importance": 0.09110107448648373
    }
    ]
    ranks = [item['rank'] for item in channel_ranking]
    channels = [item['channel'] for item in channel_ranking]
    importances = [item['importance'] for item in channel_ranking]
    plt.figure(figsize=(10, 6))
    plt.barh(channels, importances, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Channel Importance Ranking from Random Forest Classifier')
    plt.gca().invert_yaxis()  # Highest rank at the top
    for index, value in enumerate(importances):
        plt.text(value, index, f'{value:.4f}')
    plt.show()

if __name__ == "__main__":
    visualize_rfc()