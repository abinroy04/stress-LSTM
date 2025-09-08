import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
import json
import argparse
from model import StressDetectionLSTM
from data_loader import load_data, add_wav2vec_features
from data_collator import DataCollatorSpeechSeq2SeqWithPadding
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoFeatureExtractor
from processor import DSProcessor
from sklearn.metrics import f1_score, precision_score, recall_score

load_dotenv()

# Hardcoded configuration
CONFIG = {
    # Dataset arguments
    "dataset": "abinroy04/ITA-timed",
    "data_path": "./preprocessed_data",
    "force_download": True,
    "max_samples": None,  # Set to an integer (e.g., 100) for debugging
    
    # Model arguments
    "wav2vec_model": "facebook/wav2vec2-base",
    "hidden_size": 256,
    "lstm_layers": 3,
    "dropout": 0.1,
    "max_words": 20,
    
    # Training arguments
    "batch_size": 8,
    "epochs": 10,
    "lr": 1e-4,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "output_dir": "./outputs",
    "seed": 42,
    
    # Other settings
    "transcription_column_name": "transcription",
    "hf_token": os.environ.get("HF_TOKEN")
}

# Create a simple processor that only handles text tokenization - moved outside main() to be picklable
class SimpleProcessor:
    def __init__(self, model_name=CONFIG["wav2vec_model"]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = None
            
    def __call__(self, inputs, **kwargs):
        # Only handle text - we'll use the wav2vec processor separately for audio
        if isinstance(inputs, str) or (isinstance(inputs, list) and all(isinstance(item, str) for item in inputs)):
            return self.tokenizer(inputs, **kwargs)
        return {'input_values': torch.tensor(inputs)}
    
    def pad(self, features, return_tensors="pt"):
        # Simple padding function for batch preparation
        if all("input_features" in f for f in features):
            input_features = [f["input_features"] for f in features]
            max_length = max(f.shape[0] for f in input_features)
            
            padded_features = []
            for feat in input_features:
                if feat.shape[0] < max_length:
                    padding = torch.zeros((max_length - feat.shape[0], feat.shape[1]))
                    padded_feat = torch.cat([feat, padding], dim=0)
                else:
                    padded_feat = feat
                padded_features.append(padded_feat)
            
            return {"input_features": torch.stack(padded_features)}
        return features

def train_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=CONFIG["epochs"], 
    lr=CONFIG["lr"], 
    warmup_steps=CONFIG["warmup_steps"], 
    weight_decay=CONFIG["weight_decay"],
    device='cuda',
    output_dir=CONFIG["output_dir"]
):
    """
    Train the LSTM stress detection model.
    
    Args:
        model: StressDetectionLSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for AdamW optimizer
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save model checkpoints
        
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": []
    }
    
    best_val_f1 = 0.0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch)
            loss = outputs['loss']
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Collect predictions and labels for metrics
            logits = outputs['logits']
            masks = outputs['masks']
            
            # Get labels - check if they're in the expected format
            if 'labels_head' in batch:
                labels = batch['labels_head']
            else:
                # Skip metrics for this batch if no labels
                continue
            
            # Get predictions
            preds = torch.argmax(logits, dim=2)
            
            # Mask out padding and special tokens
            valid_mask = (labels != -100)
            
            # Ensure valid_indices is the right size and properly aligned with predictions
            for i in range(len(preds)):
                try:
                    valid_indices = valid_mask[i].cpu().numpy()
                    
                    # Check if shapes match, and handle mismatch
                    if valid_indices.shape[0] != preds[i].shape[0]:
                        print(f"Shape mismatch: valid_indices={valid_indices.shape}, preds={preds[i].shape}")
                        
                        # Use only the common part of both arrays
                        min_len = min(valid_indices.shape[0], preds[i].shape[0])
                        
                        # Extract the valid predictions for the common part only
                        if np.any(valid_indices[:min_len]):
                            # Create masks for the common part
                            common_mask = valid_indices[:min_len]
                            
                            # Get predictions and labels for the common part only
                            curr_preds = preds[i][:min_len][common_mask].cpu().numpy()
                            curr_labels = labels[i][:min_len][common_mask].cpu().numpy()
                            
                            all_preds.extend(curr_preds)
                            all_labels.extend(curr_labels)
                    else:
                        # No mismatch, use the full arrays
                        if np.any(valid_indices):
                            all_preds.extend(preds[i][valid_indices].cpu().numpy())
                            all_labels.extend(labels[i][valid_indices].cpu().numpy())
                except Exception as e:
                    print(f"Error processing metrics for sample {i}: {e}")
                    print(f"Shapes - valid_mask: {valid_mask[i].shape}, preds: {preds[i].shape}, labels: {labels[i].shape}")
                    continue
        
        train_loss /= len(train_loader)
        
        # Calculate metrics
        if len(all_labels) > 0:
            train_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        else:
            train_accuracy = 0.0
            train_f1 = 0.0
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                # Move batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
                
                # Forward pass
                outputs = model(batch)
                
                if outputs['loss'] is not None:
                    val_loss += outputs['loss'].item()
                
                # Collect predictions and labels for metrics
                logits = outputs['logits']
                masks = outputs['masks']
                
                # Get labels
                if 'labels_head' in batch:
                    labels = batch['labels_head']
                else:
                    # Skip metrics for this batch if no labels
                    continue
                
                # Get predictions
                preds = torch.argmax(logits, dim=2)
                
                # Mask out padding and special tokens
                valid_mask = (labels != -100)
                
                # Ensure valid_indices is the right size
                for i in range(len(preds)):
                    try:
                        valid_indices = valid_mask[i].cpu().numpy()
                        
                        # Check if shapes match, and handle mismatch
                        if valid_indices.shape[0] != preds[i].shape[0]:
                            print(f"Shape mismatch: valid_indices={valid_indices.shape}, preds={preds[i].shape}")
                            
                            # Truncate the longer one to match the shorter one
                            min_len = min(valid_indices.shape[0], preds[i].shape[0])
                            valid_indices = valid_indices[:min_len]
                            
                        if np.any(valid_indices):
                            curr_preds = preds[i][:len(valid_indices)][valid_indices[:len(preds[i])]].cpu().numpy()
                            curr_labels = labels[i][:len(valid_indices)][valid_indices[:len(labels[i])]].cpu().numpy()
                            all_preds.extend(curr_preds)
                            all_labels.extend(curr_labels)
                    except Exception as e:
                        print(f"Error processing validation metrics for sample {i}: {e}")
                        continue
        
        val_loss /= len(val_loader)
        
        # Calculate metrics
        if len(all_labels) > 0:
            val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        else:
            val_accuracy = 0.0
            val_f1 = 0.0
            val_precision = 0.0
            val_recall = 0.0
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(float(train_accuracy))
        history["val_accuracy"].append(float(val_accuracy))
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(val_f1))
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # Save model if it's the best so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model_path = os.path.join(output_dir, f"stress_detection_best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved with Val F1: {val_f1:.4f}")
        
        # Save checkpoint every epoch
        model_path = os.path.join(output_dir, f"stress_detection_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)
        
        # Save training history
        with open(os.path.join(output_dir, "training_history.json"), "w") as f:
            json.dump(history, f)
    
    return model, history


def main():
    # Set random seed
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating LSTM Stress Detection model...")
    model = StressDetectionLSTM(
        wav2vec_model_name=CONFIG["wav2vec_model"],
        lstm_hidden_size=CONFIG["hidden_size"],
        lstm_layers=CONFIG["lstm_layers"],
        dropout=CONFIG["dropout"],
        max_words=CONFIG["max_words"]
    )
    
    # Create model for data loading
    model_with_emphasis_head = type('', (), {})()
    
    # Set the processor using our externally defined class
    model_with_emphasis_head.processor = SimpleProcessor(CONFIG["wav2vec_model"])
    
    # Add wav2vec processor to the model
    model_with_emphasis_head.wav2vec_processor = model.wav2vec_processor
    
    model_with_emphasis_head.whisper_backbone_name = None
    
    # Load dataset using existing data loader
    print(f"Loading dataset: {CONFIG['dataset']}")
    dataset_loader = load_data(
        model_with_emphasis_head=model_with_emphasis_head,
        transcription_column_name=CONFIG["transcription_column_name"],
        dataset_name=CONFIG["dataset"],
        save_path=CONFIG["data_path"],
        force_download=CONFIG["force_download"],
        max_samples=CONFIG["max_samples"]
    )
    
    # Split into train, val, test
    train_dataset, val_dataset, test_dataset = dataset_loader.split_train_val_test()
    
    # Ensure we have wav2vec features in the dataset
    train_has_wav2vec = hasattr(train_dataset, 'column_names') and 'wav2vec_inputs' in train_dataset.column_names
    if not train_has_wav2vec:
        print("Adding wav2vec features to training dataset...")
        train_dataset = add_wav2vec_features(train_dataset, model_with_emphasis_head)
    
    val_has_wav2vec = hasattr(val_dataset, 'column_names') and 'wav2vec_inputs' in val_dataset.column_names
    if not val_has_wav2vec:
        print("Adding wav2vec features to validation dataset...")
        val_dataset = add_wav2vec_features(val_dataset, model_with_emphasis_head)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data collator for training
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=model_with_emphasis_head.processor,
        decoder_start_token_id=0,
        forced_decoder_ids=0,
        eos_token_id=0,
        transcription_column_name=CONFIG["transcription_column_name"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=0,  # Changed from 4 to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=0,  # Changed from 4 to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train model
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=CONFIG["epochs"],
        lr=CONFIG["lr"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        device=device,
        output_dir=CONFIG["output_dir"]
    )
    
    # Save final model
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    torch.save(trained_model.state_dict(), os.path.join(CONFIG["output_dir"], "stress_detection_final.pt"))
    print(f"Training completed. Final model saved to {CONFIG['output_dir']}/stress_detection_final.pt")


if __name__ == "__main__":
    main()