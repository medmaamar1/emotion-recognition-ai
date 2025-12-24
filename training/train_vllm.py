import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from PIL import Image
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.dataset import RAFCEDataset

def train_vllm():
    model_id = "llava-hf/llava-1.5-7b-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    DATA_ROOT = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAF-AU\aligned"
    PARTITION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_partition.txt"
    EMOTION_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_emolabel.txt"
    AU_FILE = r"c:\Users\OrdiOne\Desktop\emotion recognition ai\RAFCE_AUlabel.txt"

    print(f"Initializing LLaVA training on {device}...")
    
    # Note: Requires GPU for efficient training
    # For CPU check, we will initialize the processor and outline data collator
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 1. Load Dataset
    raw_dataset = RAFCEDataset(DATA_ROOT, PARTITION_FILE, EMOTION_FILE, AU_FILE, partition_id=0, use_aligned=True)
    
    def transform_to_hf(example):
        # Format for LLaVA
        prompt = f"USER: <image>\nIdentify the emotion in this face and explain based on Action Units.\nASSISTANT: The emotion is {raw_dataset.emotion_names[example['label']]}. This is characterized by {example['aus']}."
        return {"prompt": prompt, "image": example['image']}

    # More advanced: Create custom data collator for LLaVA
    
    # 2. LoRA Setup
    # model = LlavaForConditionalGeneration.from_pretrained(...)
    # model = apply_lora(model)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./vllm_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="none"
    )

    print("Configured Fine-tuning parameters. Ready for GPU execution.")

if __name__ == "__main__":
    train_vllm()
