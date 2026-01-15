import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_config

# Action Units list (18 AUs)
AU_LABELS = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
             'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU43', 'AU45']

def parse_au_labels(au_str):
    """
    Parse AU string labels to binary vector.
    
    Args:
        au_str: String like "AU1 AU2 AU4" or "null"
    
    Returns:
        Binary vector of length 18 indicating which AUs are present
    """
    au_vector = np.zeros(len(AU_LABELS), dtype=np.float32)
    
    if au_str == "null" or au_str is None:
        return au_vector
    
    aus = au_str.split()
    for au in aus:
        if au in AU_LABELS:
            idx = AU_LABELS.index(au)
            au_vector[idx] = 1.0
    
    return au_vector

class RAFCEDataset(Dataset):
    """
    RAF-CE Dataset for Compound Emotion Recognition and AU Detection.
    Categories:
    0: Happily surprised, 1: Happily disgusted, 2: Sadly fearful, 3: Sadly angry,
    4: Sadly surprised, 5: Sadly disgusted, 6: Fearfully angry, 7: Fearfully surprised,
    8: Fearfully disgusted, 9: Angrily surprised, 10: Angrily disgusted,
    11: Disgustedly surprised, 12: Happily fearful, 13: Happily sad
    """
    def __init__(self, root_dir=None, partition_file=None, emotion_file=None, au_file=None, partition_id=0, transform=None, use_aligned=False, return_au_vector=True):
        """
        Initialize RAF-CE Dataset.
        
        Args:
            root_dir: Path to image directory. If None, uses config default.
            partition_file: Path to partition file. If None, uses config default.
            emotion_file: Path to emotion label file. If None, uses config default.
            au_file: Path to AU label file. If None, uses config default.
            partition_id: 0 for train, 1 for test, 2 for validation
            transform: Image transformations
            use_aligned: Whether to use aligned images
            return_au_vector: Whether to return AU labels as binary vector (default: True)
        """
        # Load config for default paths
        config = get_config()
        
        self.root_dir = root_dir if root_dir is not None else (config['data_root_aligned'] if use_aligned else config['data_root_raw'])
        self.partition_file = partition_file if partition_file is not None else config['partition_file']
        self.emotion_file = emotion_file if emotion_file is not None else config['emotion_file']
        self.au_file = au_file if au_file is not None else config['au_file']
        self.transform = transform
        self.use_aligned = use_aligned
        self.return_au_vector = return_au_vector
        
        # Verify paths exist
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Image directory not found: {self.root_dir}")
        if not os.path.exists(self.partition_file):
            raise FileNotFoundError(f"Partition file not found: {self.partition_file}")
        if not os.path.exists(self.emotion_file):
            raise FileNotFoundError(f"Emotion file not found: {self.emotion_file}")
        if not os.path.exists(self.au_file):
            raise FileNotFoundError(f"AU file not found: {self.au_file}")
        
        # Load partitions
        partition_df = pd.read_csv(self.partition_file, sep=' ', header=None, names=['image_id', 'partition_id'])
        self.image_ids = partition_df[partition_df['partition_id'] == partition_id]['image_id'].values
        
        # Load emotion labels
        emotion_df = pd.read_csv(self.emotion_file, sep=' ', header=None, names=['image_id', 'label'])
        self.emotions = emotion_df.set_index('image_id')['label'].to_dict()
        
        # Load AU labels
        self.aus = {}
        with open(self.au_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                img_id = parts[0]
                au_str = parts[1] if len(parts) > 1 else "null"
                self.aus[img_id] = au_str

        # Filter image_ids that exist in emotion labels
        self.image_ids = [img_id for img_id in self.image_ids if img_id in self.emotions]
        
        print(f"Loaded {len(self.image_ids)} images for partition {partition_id} from {self.root_dir}")

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
            
        # Parse AU labels to binary vector if requested
        if self.return_au_vector:
            au_vector = parse_au_labels(au_labels)
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'image_id': img_id,
                'aus': au_labels,
                'au_vector': torch.tensor(au_vector, dtype=torch.float)
            }
        else:
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'image_id': img_id,
                'aus': au_labels
            }

if __name__ == "__main__":
    # Test the dataset loading using config
    config = get_config()
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    print(f"Running on Kaggle: {config['is_kaggle']}")
    print()
    
    # Test Aligned
    print("Testing Aligned Dataset (Train)...")
    try:
        aligned_dataset = RAFCEDataset(partition_id=0, use_aligned=True)
        print(f"Aligned Train Dataset size: {len(aligned_dataset)}")
        if len(aligned_dataset) > 0:
            sample = aligned_dataset[0]
            print(f"Sample Image ID: {sample['image_id']}")
            print(f"Sample Label: {sample['label'].item()}")
            print(f"Sample AUs: {sample['aus']}")
    except Exception as e:
        print(f"Error loading aligned dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test Raw
    print("Testing Raw Dataset (Train)...")
    try:
        raw_dataset = RAFCEDataset(partition_id=0, use_aligned=False)
        print(f"Raw Train Dataset size: {len(raw_dataset)}")
        if len(raw_dataset) > 0:
            sample = raw_dataset[0]
            print(f"Sample Image ID: {sample['image_id']}")
            print(f"Sample Label: {sample['label'].item()}")
            print(f"Sample AUs: {sample['aus']}")
    except Exception as e:
        print(f"Error loading raw dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    
    # Test Test partition
    print("Testing Aligned Dataset (Test)...")
    try:
        test_dataset = RAFCEDataset(partition_id=1, use_aligned=True)
        print(f"Aligned Test Dataset size: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
