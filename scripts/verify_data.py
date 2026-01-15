import os
import sys
# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from scripts.dataset import RAFCEDataset, parse_au_labels, AU_LABELS
from config import get_config

def verify_au_parsing():
    print("Verifying AU Label Parsing...")
    
    # Test cases found in file
    test_cases = [
        ("1+4+25", ["1", "4", "25"]),
        ("4+9+12+25+26+43", ["4", "9", "12", "25", "26", "43"]),
        ("L12+24", ["12", "24"]),
        ("null", []),
        ("5+6+7+12+25", ["5", "6", "7", "12", "25"])
    ]
    
    for au_str, expected_nums in test_cases:
        print(f"Testing input: '{au_str}'")
        vector = parse_au_labels(au_str)
        
        # Check if expected indices are 1.0
        active_indices = np.where(vector == 1.0)[0]
        active_labels = [AU_LABELS[i] for i in active_indices]
        print(f"  -> Detected: {active_labels}")
        
        # Verify
        expected_labels = [f"AU{n}" for n in expected_nums if f"AU{n}" in AU_LABELS]
        # Note: AU43 might not be in AU_LABELS depending on dataset.py definition.
        # Let's check what's in AU_LABELS in dataset.py
        # AU_LABELS from dataset.py: ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10', 
        # 'AU12', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU43', 'AU45']
        
        if sorted(active_labels) == sorted(expected_labels):
            print("  Result: PASS")
        else:
            print(f"  Result: FAIL. Expected {expected_labels}, got {active_labels}")

    print("\nLoading Dataset to check statistics...")
    try:
        # Load a small subset only
        dataset = RAFCEDataset(partition_id=0, use_aligned=True)
        print(f"Dataset loaded. Size: {len(dataset)}")
        
        valid_au_count = 0
        total = 0
        
        # Check first 50 samples
        for i in range(min(50, len(dataset))):
            sample = dataset[i]
            au_vec = sample['au_vector']
            au_str = sample['aus']
            
            if torch.is_tensor(au_vec):
                au_vec = au_vec.numpy()
                
            if np.sum(au_vec) > 0:
                valid_au_count += 1
            
            # print(f"ID: {sample['image_id']}, AU Str: {au_str}, Parsed Sum: {np.sum(au_vec)}")
            
        print(f"Found {valid_au_count} samples with valid parsed AUs out of first {min(50, len(dataset))}")
        
        if valid_au_count == 0:
             print("WARNING: Parsed 0 valid AU vectors. Parsing might still be broken or dataset has only nulls.")
        else:
             print("SUCCESS: Parsing seems to represent data.")
             
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

import torch
if __name__ == "__main__":
    verify_au_parsing()
