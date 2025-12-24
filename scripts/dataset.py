import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class RAFCEDataset(Dataset):
    """
    RAF-CE Dataset for Compound Emotion Recognition and AU Detection.
    Categories:
    0: Happily surprised, 1: Happily disgusted, 2: Sadly fearful, 3: Sadly angry,
    4: Sadly surprised, 5: Sadly disgusted, 6: Fearfully angry, 7: Fearfully surprised,
    8: Fearfully disgusted, 9: Angrily surprised, 10: Angrily disgusted,
    11: Disgustedly surprised, 12: Happily fearful, 13: Happily sad
    """
    def __init__(self, root_dir, partition_file, emotion_file, au_file, partition_id, transform=None, use_aligned=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_aligned = use_aligned
        
        # Load partitions
        partition_df = pd.read_csv(partition_file, sep=' ', header=None, names=['image_id', 'partition_id'])
        self.image_ids = partition_df[partition_df['partition_id'] == partition_id]['image_id'].values
        
        # Load emotion labels
        emotion_df = pd.read_csv(emotion_file, sep=' ', header=None, names=['image_id', 'label'])
        self.emotions = emotion_df.set_index('image_id')['label'].to_dict()
        
        # Load AU labels
        self.aus = {}
        with open(au_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                img_id = parts[0]
                au_str = parts[1] if len(parts) > 1 else "null"
                self.aus[img_id] = au_str

        # Filter image_ids that exist in emotion labels
        self.image_ids = [img_id for img_id in self.image_ids if img_id in self.emotions]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        if self.use_aligned:
            # Map 0001.jpg to 0001_aligned.jpg
            filename = img_id.replace('.jpg', '_aligned.jpg')
        else:
            filename = img_id
            
        img_path = os.path.join(self.root_dir, filename)
        
        image = Image.open(img_path).convert('RGB')
        label = self.emotions[img_id]
        au_labels = self.aus[img_id]
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': img_id,
            'aus': au_labels
        }

if __name__ == "__main__":
    # Test the dataset loading
    DATA_ROOT_RAW = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\dataset_root\RAF-AU\original"
    DATA_ROOT_ALIGNED = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAF-AU\aligned"
    PARTITION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_partition.txt"
    EMOTION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_emolabel.txt"
    AU_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_AUlabel.txt"
    
    # Test Aligned
    print("Testing Aligned Dataset...")
    try:
        aligned_dataset = RAFCEDataset(DATA_ROOT_ALIGNED, PARTITION_FILE, EMOTION_FILE, AU_FILE, partition_id=0, use_aligned=True)
        print(f"Aligned Train Dataset size: {len(aligned_dataset)}")
        if len(aligned_dataset) > 0:
            sample = aligned_dataset[0]
            print(f"Sample Image ID (label ref): {sample['image_id']}")
            print(f"Sample Label: {sample['label']}")
            print(f"Sample AUs: {sample['aus']}")
    except Exception as e:
        print(f"Error loading aligned dataset: {e}")
