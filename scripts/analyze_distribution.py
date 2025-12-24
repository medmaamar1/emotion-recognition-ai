import pandas as pd
import matplotlib.pyplot as plt

EMOTION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_emolabel.txt"
PARTITION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_partition.txt"

emotion_names = {
    0: "Happily surprised", 1: "Happily disgusted", 2: "Sadly fearful", 3: "Sadly angry",
    4: "Sadly surprised", 5: "Sadly disgusted", 6: "Fearfully angry", 7: "Fearfully surprised",
    8: "Fearfully disgusted", 9: "Angrily surprised", 10: "Angrily disgusted",
    11: "Disgustedly surprised", 12: "Happily fearful", 13: "Happily sad"
}

# Load data
emotions = pd.read_csv(EMOTION_FILE, sep=' ', header=None, names=['image_id', 'label'])
partitions = pd.read_csv(PARTITION_FILE, sep=' ', header=None, names=['image_id', 'partition_id'])

# Merge
df = pd.merge(emotions, partitions, on='image_id')

# Analysis
total_counts = df['label'].value_counts().sort_index()
train_counts = df[df['partition_id'] == 0]['label'].value_counts().sort_index()

print("Full Dataset Distribution:")
for label, count in total_counts.items():
    print(f"{label} ({emotion_names[label]}): {count}")

print("\nTrain Set Distribution:")
for label, count in train_counts.items():
    print(f"{label} ({emotion_names[label]}): {count}")
