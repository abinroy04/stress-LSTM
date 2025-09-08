import torch
import numpy as np
from model import StressDetectionLSTM
from scipy.io import wavfile
import argparse
import json
import os
from transformers import Wav2Vec2Processor
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from huggingface_hub import login

# Hardcoded configuration
CONFIG = {
    "model_path": "./outputs/stress_detection_best.pth",  # Path to the trained model
    "output_file": None,                                  # Will be generated from audio filename
    "wav2vec_model": "facebook/wav2vec2-base",           # Should match the model used for training
    "hf_dataset": "abinroy04/ITA-timed",           # HuggingFace dataset for evaluation
    "hf_token": "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", # HuggingFace API token
    "split": "test",                                      # Dataset split to use
    "max_samples": None                                   # Maximum samples to process (None = all)
}

def load_audio(file_path):
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dictionary with audio array and sampling rate
    """
    # Load audio file
    sample_rate, audio = wavfile.read(file_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Convert to float and normalize
    audio = audio.astype(np.float32)
    audio = audio / np.max(np.abs(audio))
    
    return {'array': audio, 'sampling_rate': sample_rate}


def predict_stress(model_path=CONFIG["model_path"], audio_file=None, audio=None, word_boundaries=None, output_file=None, device=None):
    """
    Predict word stress in audio.
    
    Args:
        model_path: Path to the trained model
        audio_file: Path to the audio file (optional if audio is provided)
        audio: Dictionary with audio array and sampling rate (optional if audio_file is provided)
        word_boundaries: Word boundary information (optional, will be extracted if not provided)
        output_file: Path to save predictions (optional)
        device: Device to run inference on (optional)
        
    Returns:
        Dictionary with word stress predictions
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = StressDetectionLSTM(wav2vec_model_name=CONFIG["wav2vec_model"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Set output file if not specified and audio_file is provided
    if output_file is None and audio_file is not None:
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = f"{base_name}_stress.json"
    
    # Load audio if provided as file path
    if audio_file is not None:
        audio = load_audio(audio_file)
    
    # Ensure we have audio data
    if audio is None:
        raise ValueError("Either audio_file or audio must be provided")
    
    # Process with wav2vec
    wav2vec_processor = model.wav2vec_processor
    inputs = wav2vec_processor(
        audio['array'], 
        sampling_rate=16000, 
        return_tensors="pt"
    )
    
    # Extract wav2vec features
    with torch.no_grad():
        wav2vec_inputs = inputs.input_values.to(device)
        wav2vec_features = model.wav2vec_model(wav2vec_inputs).last_hidden_state
    
    
    # Ensure word_boundaries is properly formatted (list of dictionaries)
    if isinstance(word_boundaries, list):
        # Check if the first item is a string - convert if needed
        if word_boundaries and isinstance(word_boundaries[0], str):
            # Convert list of strings to proper format
            words = word_boundaries
            # Create dummy boundaries with even spacing
            audio_duration = len(audio['array']) / audio['sampling_rate']
            word_duration = audio_duration / len(words)
            word_boundaries = [
                {
                    'word': word,
                    'start': idx * word_duration,
                    'end': (idx + 1) * word_duration
                }
                for idx, word in enumerate(words)
            ]
    
    # Predict stress
    predictions = model.predict(wav2vec_features.squeeze(0), word_boundaries)
    
    # Create output
    result = {
        "audio_file": audio_file,
        "words": [
            {
                "word": boundary['word'] if isinstance(boundary, dict) and 'word' in boundary else f"word_{i+1}",
                "start": float(boundary['start'] if isinstance(boundary, dict) and 'start' in boundary else 0.0),
                "end": float(boundary['end'] if isinstance(boundary, dict) and 'end' in boundary else 0.0),
                "stress": int(predictions[i] if i < len(predictions) else 0)
            }
            for i, boundary in enumerate(word_boundaries)
        ]
    }
    
    
    return result

def load_hf_dataset(dataset_name=CONFIG["hf_dataset"], 
                   token=CONFIG["hf_token"], 
                   split=CONFIG["split"], 
                   max_samples=CONFIG["max_samples"]):
    """
    Load dataset from HuggingFace
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        token: HuggingFace authentication token
        split: Dataset split to use (default: test)
        max_samples: Maximum number of samples to load (optional)
    
    Returns:
        Dataset object
    """
    print(f"Loading {split} split from HuggingFace dataset: {dataset_name}...")
    
    # Login to authenticate with HuggingFace
    login(token=token)
    
    # Load the specified split
    dataset = load_dataset(dataset_name, split=split, token=token)
    
    # Preprocess the dataset to adapt it to our code
    def preprocess_sample(sample):
        # Convert audio array to numpy if it's a list
        if isinstance(sample['audio']['array'], list):
            import numpy as np
            sample['audio']['array'] = np.array(sample['audio']['array'], dtype=np.float32)
        return sample
    
    dataset = dataset.map(preprocess_sample)
    
    # Print a sample to verify structure
    print("\nSample after preprocessing:")
    sample = dataset[0]
    for key, value in sample.items():
        if key != 'audio':  # Skip audio for brevity
            print(f"  {key}: {value}")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited dataset to first {len(dataset)} samples")
    
    return dataset

def evaluate_model(model_path=CONFIG["model_path"], dataset=None, device=None):
    """
    Evaluate the model on a dataset.
    
    Args:
        model_path: Path to the trained model
        dataset: Dataset to evaluate on (if None, will load from HuggingFace)
        device: Device to run inference on
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset if not provided
    if dataset is None:
        dataset = load_hf_dataset()
    
    # Load model
    model = StressDetectionLSTM(wav2vec_model_name=CONFIG["wav2vec_model"])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    predictions = []
    references = []
    skipped_samples = 0
    
    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        # Handle different formats of stress annotations
        if 'emphasis_indices' in sample and isinstance(sample['emphasis_indices'], dict) and 'binary' in sample['emphasis_indices']:
            gt_stresses = sample['emphasis_indices']['binary']
        elif 'emphasis_indices' in sample:
            gt_stresses = sample['emphasis_indices']
        else:
            print(f"Skipping sample {i}: Could not find stress information")
            skipped_samples += 1
            continue
        
        try:
            # GIVE HIGHER PRIORITY TO TRANSCRIPTION - Start with transcription-based boundaries
            word_boundaries = None
            
            # Use transcription as highest priority source for word boundaries
            if 'transcription' in sample:
                print(f"Sample {i}: Using transcription to create word boundaries")
                words = sample['transcription'].split()
                
                # Create word boundaries that match ground truth length
                if 'emphasis_indices' in sample:
                    gt_length = len(gt_stresses)
                    
                    # Ensure word count matches ground truth length
                    if len(words) != gt_length:
                        print(f"Warning: Transcription word count ({len(words)}) doesn't match emphasis count ({gt_length})")
                        
                        # Try to align words and stresses
                        if len(words) > gt_length:
                            # If we have more words than stress marks, truncate words
                            words = words[:gt_length]
                        else:
                            # If we have fewer words than stress marks, we'll pad the words
                            # or truncate stress marks later
                            pass
                
                # Create evenly spaced word boundaries from the transcription
                audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
                word_duration = audio_duration / len(words)
                word_boundaries = [
                    {
                        'word': word,
                        'start': j * word_duration,
                        'end': (j + 1) * word_duration
                    }
                    for j, word in enumerate(words)
                ]
            
            # Only if transcription didn't work, check for existing word_boundaries
            if not word_boundaries and 'word_boundaries' in sample:
                existing_boundaries = sample.get('word_boundaries', None)
                
                # Handle different word_boundaries formats
                if isinstance(existing_boundaries, str):
                    print(f"Sample {i}: Cannot use string word_boundaries")
                    # We'll create boundaries from energy-based segmentation later
                elif isinstance(existing_boundaries, list) and existing_boundaries:
                    if isinstance(existing_boundaries[0], str):
                        print(f"Sample {i}: Converting list-of-strings word_boundaries")
                        words = existing_boundaries
                        audio_duration = len(sample['audio']['array']) / sample['audio']['sampling_rate']
                        word_duration = audio_duration / len(words)
                        word_boundaries = [
                            {
                                'word': word,
                                'start': idx * word_duration,
                                'end': (idx + 1) * word_duration
                            }
                            for idx, word in enumerate(words)
                        ]
                    elif isinstance(existing_boundaries[0], dict):
                        print(f"Sample {i}: Using existing word_boundaries dictionary")
                        word_boundaries = existing_boundaries
            
            # Process audio and predict stress
            result = predict_stress(
                model_path=model_path,
                audio=sample['audio'],
                word_boundaries=word_boundaries,
                device=device
            )
            
            # Extract predictions
            pred_stresses = [word['stress'] for word in result['words']]
            
            # Handle length mismatch - prioritize preserving all ground truth labels
            if len(pred_stresses) != len(gt_stresses):
                print(f"Sample {i}: Length mismatch - pred: {len(pred_stresses)}, gt: {len(gt_stresses)}")
                
                # If we have more predictions than ground truth, truncate predictions
                if len(pred_stresses) > len(gt_stresses):
                    pred_stresses = pred_stresses[:len(gt_stresses)]
                    print(f"Truncated predictions to match {len(gt_stresses)} ground truth labels")
                
                # If we have more ground truth than predictions, truncate ground truth
                else:
                    gt_stresses = gt_stresses[:len(pred_stresses)]
                    print(f"Truncated ground truth to match {len(pred_stresses)} predictions")
            
            predictions.extend(pred_stresses)
            references.extend(gt_stresses)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            skipped_samples += 1
            continue
    
    print(f"Processed {len(dataset) - skipped_samples} samples, skipped {skipped_samples}")
    
    # Only calculate metrics if we have predictions
    if len(predictions) == 0:
        print("No predictions collected, cannot compute metrics.")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(references, predictions),
        "precision": precision_score(references, predictions, average="binary", zero_division=0),
        "recall": recall_score(references, predictions, average="binary", zero_division=0),
        "f1": f1_score(references, predictions, average="binary", zero_division=0)
    }
    
    # Add detailed analysis for debugging
    class_counts = {
        "predictions": {
            "0": predictions.count(0),
            "1": predictions.count(1),
            "other": len(predictions) - predictions.count(0) - predictions.count(1)
        },
        "references": {
            "0": references.count(0),
            "1": references.count(1),
            "other": len(references) - references.count(0) - references.count(1)
        }
    }
    
    print("\nClass Distribution:")
    print(f"Predictions: {class_counts['predictions']}")
    print(f"References:  {class_counts['references']}")
    
    # Check if model is predicting all 0s
    if class_counts['predictions']['1'] == 0:
        print("\nWARNING: Model is predicting ALL unstressed words (class 0)!")
        print("This explains why precision, recall, and F1 are 0 - no positive predictions.")
        
        # Try calculating metrics with different average parameters
        print("\nTrying different metric calculations:")
        print(f"F1 (micro): {f1_score(references, predictions, average='micro', zero_division=0):.4f}")
        print(f"F1 (macro): {f1_score(references, predictions, average='macro', zero_division=0):.4f}")
        print(f"F1 (weighted): {f1_score(references, predictions, average='weighted', zero_division=0):.4f}")
    
    # Add these alternative metrics to the return value
    metrics.update({
        "f1_micro": f1_score(references, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(references, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(references, predictions, average="weighted", zero_division=0),
        "class_counts": class_counts
    })
    
    return metrics

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate stress detection model on HuggingFace dataset")
    parser.add_argument("--model_path", default=CONFIG["model_path"], help="Path to model checkpoint")
    parser.add_argument("--dataset", default=CONFIG["hf_dataset"], help="HuggingFace dataset name")
    parser.add_argument("--split", default=CONFIG["split"], help="Dataset split (train, validation, test)")
    parser.add_argument("--max_samples", type=int, default=CONFIG["max_samples"], help="Maximum samples to evaluate")
    parser.add_argument("--mode", choices=["evaluate", "predict"], default="evaluate", 
                        help="'evaluate' on a dataset or 'predict' on a single file")
    parser.add_argument("--audio_file", default=CONFIG["output_file"], 
                        help="Audio file to predict (used when mode='predict')")
    parser.add_argument("--output_file", default=None, help="Output file for predictions")
    
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        # Evaluate on dataset
        print(f"Evaluating model on {args.dataset} ({args.split} split)...")
        
        # Update configuration
        CONFIG["hf_dataset"] = args.dataset
        CONFIG["split"] = args.split
        CONFIG["max_samples"] = args.max_samples
        CONFIG["model_path"] = args.model_path
        
        # Load dataset
        dataset = load_hf_dataset()
        
        # Evaluate model
        metrics = evaluate_model(model_path=args.model_path, dataset=dataset)
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        
        # Save results if output file is specified
        if args.output_file:
            results = {
                "dataset": args.dataset,
                "split": args.split,
                "num_samples": len(dataset),
                "metrics": metrics
            }
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output_file}")
    
    elif args.mode == "predict":
        # Predict on single file
        if not args.audio_file:
            parser.error("--audio_file is required when mode='predict'")
        
        # Predict stress
        result = predict_stress(
            model_path=args.model_path, 
            audio_file=args.audio_file, 
            output_file=args.output_file
        )
        
        # Print results
        print("\nWord stress patterns (1=stressed, 0=unstressed):")
        for word in result["words"]:
            stress_label = "STRESSED" if word["stress"] == 1 else "unstressed"
            print(f"{word['word']} ({word['start']:.2f}-{word['end']:.2f}s): {stress_label}")

if __name__ == "__main__":
    main()
