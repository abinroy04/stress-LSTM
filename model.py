import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np
import os

class StressDetectionLSTM(nn.Module):
    """
    LSTM model for stress detection in speech.
    
    This model:
    1. Uses wav2vec features averaged per word using provided word boundaries
    2. Processes features through 3 LSTM layers
    3. Outputs stress predictions (1 for stressed, 0 for unstressed) for each word
    4. Handles variable-length sentences with padding and masking
    """
    def __init__(
        self,
        wav2vec_model_name="facebook/wav2vec2-base",
        lstm_hidden_size=256,
        lstm_layers=3,
        dropout=0.1,
        max_words=20
    ):
        super(StressDetectionLSTM, self).__init__()
        
        # Load wav2vec model and processor
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        self.wav2vec_feature_dim = self.wav2vec_model.config.hidden_size
        
        # Freeze wav2vec parameters for feature extraction
        for param in self.wav2vec_model.parameters():
            param.requires_grad = False
        
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(
            input_size=self.wav2vec_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layer: bidirectional LSTM gives 2*hidden_size features
        self.classifier = nn.Linear(lstm_hidden_size * 2, 2)
        
        self.dropout = nn.Dropout(dropout)
        self.max_words = max_words
        
        # Store model name for later reference
        self.wav2vec_backbone_name = wav2vec_model_name
    
    def extract_wav2vec_features(self, batch):
        """
        Extract wav2vec features from input features or wav2vec_inputs.
        
        Args:
            batch: Dictionary containing input data
            
        Returns:
            Tensor of wav2vec features
        """
        device = next(self.parameters()).device
        
        # Check if wav2vec_inputs is already available
        if 'wav2vec_inputs' in batch:
            # Move to device and return
            wav2vec_inputs = batch['wav2vec_inputs'].to(device)
            
            # Fix dimensionality issues - ensure shape is [batch_size, seq_len, feature_dim]
            if len(wav2vec_inputs.shape) == 2:
                # If it's [batch_size, seq_len] - add feature dimension
                print(f"Adding feature dimension to wav2vec_inputs with shape {wav2vec_inputs.shape}")
                return self.wav2vec_model(wav2vec_inputs).last_hidden_state
            elif len(wav2vec_inputs.shape) == 3:
                # Check if the feature dimension is in the right place
                if wav2vec_inputs.shape[2] == 1:
                    # If shape is [batch_size, seq_len, 1], reshape to [batch_size, seq_len]
                    print(f"Reshaping wav2vec_inputs from {wav2vec_inputs.shape} to [batch, seq_len]")
                    wav2vec_inputs = wav2vec_inputs.squeeze(2)
                    return self.wav2vec_model(wav2vec_inputs).last_hidden_state
                else:
                    # If it's already [batch_size, seq_len, feature_dim], return as is
                    return wav2vec_inputs
            else:
                # Handle unexpected shape - try to process with wav2vec model
                print(f"WARNING: Unexpected wav2vec_inputs shape: {wav2vec_inputs.shape}. Attempting to process with wav2vec model.")
                try:
                    return self.wav2vec_model(wav2vec_inputs).last_hidden_state
                except Exception as e:
                    print(f"ERROR processing wav2vec_inputs: {e}")
                    # Create dummy tensor with expected shape
                    batch_size = wav2vec_inputs.shape[0] if len(wav2vec_inputs.shape) > 0 else 1
                    seq_len = wav2vec_inputs.shape[1] if len(wav2vec_inputs.shape) > 1 else 100
                    return torch.zeros((batch_size, seq_len, self.wav2vec_feature_dim), device=device)
        
        # Otherwise, extract from audio using input_features
        if 'input_features' in batch:
            # Process with wav2vec model
            with torch.no_grad():
                outputs = self.wav2vec_model(batch['input_features'].to(device))
                return outputs.last_hidden_state
        
        raise ValueError("Neither wav2vec_inputs nor input_features found in batch")
    
    def average_features_per_word(self, features, word_boundaries):
        """
        Average wav2vec features for each word based on word boundaries.
        
        Args:
            features: Wav2vec features [batch_size, sequence_length, feature_dim]
            word_boundaries: List of lists of word boundary dictionaries
            
        Returns:
            Word-averaged features [batch_size, max_words, feature_dim]
            Masks indicating valid positions [batch_size, max_words]
        """
        # Handle dimensionality issues - ensure we have 3D tensor [batch_size, seq_len, feature_dim]
        if len(features.shape) != 3:
            print(f"WARNING: features shape is {features.shape}, expected 3D tensor")
            batch_size = features.shape[0] if len(features.shape) > 0 else 1
            
            # If we have [batch_size, seq_len], run through wav2vec model to get features
            if len(features.shape) == 2:
                print("Running 2D features through wav2vec model to get 3D features")
                with torch.no_grad():
                    features = self.wav2vec_model(features).last_hidden_state
            else:
                # Create dummy tensor with expected shape for severe errors
                print("ERROR: Cannot determine feature dimensions. Creating dummy features.")
                seq_len = 100  # Arbitrary sequence length
                features = torch.zeros((batch_size, seq_len, self.wav2vec_feature_dim), device=features.device)
                
        # Now we should have proper 3D tensor
        batch_size = features.shape[0]
        feature_dim = features.shape[2]
        device = features.device
        
        # Create output tensors
        word_features = torch.zeros((batch_size, self.max_words, feature_dim), device=device)
        masks = torch.zeros((batch_size, self.max_words), device=device)
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Get features for this sample
            sample_features = features[b]
            
            # Get word boundaries for this sample
            sample_boundaries = word_boundaries[b] if b < len(word_boundaries) else []
            
            # Print debug info if boundaries are empty
            if len(sample_boundaries) == 0:
                print(f"WARNING: Empty word boundaries for sample {b}")
                continue
                
            # Extract and average features for each word
            word_idx = 0
            
            # Convert time-based boundaries to feature indices
            # Wav2Vec2 produces ~50 frames per second
            feature_rate = 50  # Approximate frame rate for wav2vec features
            
            # Debug the structure of sample_boundaries
            if isinstance(sample_boundaries, dict):
                # Convert dictionary format to list of dictionaries
                boundary_list = []
                for key, value in sample_boundaries.items():
                    if isinstance(value, dict) and 'start' in value and 'end' in value:
                        boundary_list.append({
                            'word': value.get('word', f"word_{key}"),
                            'start': value['start'],
                            'end': value['end']
                        })
                sample_boundaries = boundary_list
            
            for word_info in sample_boundaries:
                if word_idx >= self.max_words:
                    break
                
                # Handle different formats of word_info
                if isinstance(word_info, dict) and 'start' in word_info and 'end' in word_info:
                    # Standard format - convert time to indices
                    start_idx = max(0, int(word_info['start'] * feature_rate))
                    end_idx = min(sample_features.shape[0], int(word_info['end'] * feature_rate))
                else:
                    # Handle unexpected format by creating dummy indices
                    print(f"WARNING: Unexpected word_info format: {type(word_info)}. Creating dummy boundary.")
                    os.exit()
                
                # Ensure valid range - use at least one frame if start=end
                if start_idx == end_idx:
                    end_idx = min(start_idx + 1, sample_features.shape[0])
                    
                # Ensure valid range
                if start_idx < end_idx and start_idx < sample_features.shape[0]:
                    # Extract and average features for this word
                    word_feat = torch.mean(sample_features[start_idx:end_idx], dim=0)
                    word_features[b, word_idx] = word_feat
                    masks[b, word_idx] = 1.0  # Mark as valid
                    word_idx += 1
                else:
                    print(f"WARNING: Invalid boundary indices: {start_idx}-{end_idx} for sample {b}")
        
        return word_features, masks
    
    def forward(self, batch):
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing input data
                - wav2vec_inputs or input_features: Wav2vec inputs
                - word_boundaries: Word boundary information
                - labels_head: Optional emphasis head labels
            
        Returns:
            Dictionary containing:
                - logits: Stress prediction logits [batch_size, max_words, 2]
                - masks: Attention masks [batch_size, max_words]
                - loss: Loss value if labels were provided
        """
        # Extract wav2vec features
        wav2vec_features = self.extract_wav2vec_features(batch)
        
        # Get word boundaries
        word_boundaries = batch.get('word_boundaries', [])
        
        # Average features per word
        word_features, masks = self.average_features_per_word(wav2vec_features, word_boundaries)
        
        # Apply dropout before LSTM
        word_features = self.dropout(word_features)
        
        # Process through LSTM
        # Convert masks to lengths for packing
        lengths = masks.sum(dim=1).long().clamp(min=1)  # Ensure no zero lengths
        
        # Pack sequence
        packed_features = nn.utils.rnn.pack_padded_sequence(
            word_features,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Run LSTM
        packed_lstm_out, _ = self.lstm(packed_features)
        
        # Unpack the sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out,
            batch_first=True,
            total_length=self.max_words
        )
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Classify each word
        logits = self.classifier(lstm_out)
        
        # Calculate loss if labels are provided
        loss = None
        if 'labels_head' in batch:
            try:
                # Create tensor for computing loss
                labels = batch['labels_head'].to(logits.device)
                
                # Check for dimension mismatch and fix
                if labels.shape != logits.shape[:2]:
                    print(f"WARNING: Labels shape {labels.shape} does not match logits shape {logits.shape[:2]}")
                    
                    # Resize labels to match max_words
                    # Method 1: Truncate labels or pad with -100 (ignore index)
                    resized_labels = torch.full((labels.shape[0], self.max_words), -100, 
                                               device=labels.device, dtype=labels.dtype)
                    
                    # Copy valid values up to min length of each dimension
                    min_seq_len = min(labels.shape[1], self.max_words)
                    for i in range(labels.shape[0]):
                        resized_labels[i, :min_seq_len] = labels[i, :min_seq_len]
                    
                    # Use the resized labels
                    labels = resized_labels
                
                # Reshape for loss calculation - ensure proper sizes
                reshaped_logits = logits.reshape(-1, 2)  # [batch*max_words, 2]
                reshaped_labels = labels.reshape(-1)      # [batch*max_words]
                
                # Create mask for valid positions (not -100)
                valid_mask = (reshaped_labels != -100)
                
                # Get only the valid predictions and labels
                valid_logits = reshaped_logits[valid_mask]
                valid_labels = reshaped_labels[valid_mask]
                
                # Calculate cross-entropy loss
                if len(valid_labels) > 0:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(valid_logits, valid_labels)
                else:
                    # Create dummy loss if no valid labels
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                    print("WARNING: No valid labels found for loss calculation")
            
            except Exception as e:
                # Create a dummy loss for robustness
                print(f"ERROR calculating loss: {e}")
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return {
            'logits': logits,
            'masks': masks,
            'loss': loss
        }
    
    def predict(self, audio_features, word_boundaries):
        """
        Predict stress patterns for a single audio sample.
        
        Args:
            audio_features: Audio features from wav2vec model
            word_boundaries: List of word boundary dictionaries
            
        Returns:
            Predicted stress labels (0/1) for each word
        """
        self.eval()
        with torch.no_grad():
            # Create a batch with a single sample
            batch = {
                'wav2vec_inputs': audio_features.unsqueeze(0),
                'word_boundaries': [word_boundaries]
            }
            
            # Forward pass
            outputs = self.forward(batch)
            
            # Get predictions
            logits = outputs['logits']
            masks = outputs['masks']
            
            # Apply softmax and get class with highest probability
            probs = F.softmax(logits[0], dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Only return predictions for actual words (not padding)
            valid_words = masks[0].bool()
            return predictions[valid_words].cpu().numpy()

class StressDetectionDataset(torch.utils.data.Dataset):
    """
    Dataset for word-level stress detection from audio.
    
    Processes audio samples and extracts:
    1. Audio features
    2. Word boundaries
    3. Stress labels (binary: 0=unstressed, 1=stressed)
    
    For training, validation, and evaluation.
    """
    def __init__(self, dataset, max_words=20):
        """
        Initialize the dataset.
        
        Args:
            dataset: HuggingFace dataset with audio, word_boundaries, and emphasis_indices
            max_words: Maximum number of words to keep per sample
        """
        self.dataset = dataset
        self.max_words = max_words
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a dataset sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with:
            - audio: Audio dictionary with array and sampling_rate
            - word_boundaries: List of word boundary dictionaries
            - stress_labels: Binary labels for word stress, padded to max_words
        """
        sample = self.dataset[idx]
        
        # Get audio - keep as dictionary for sampling rate info
        audio = {
            'array': sample['audio']['array'],
            'sampling_rate': sample['audio']['sampling_rate']
        }
        
        # Get word boundaries
        word_boundaries = sample.get('word_boundaries', [])
        
        # If word_boundaries is a dictionary or other structure, convert to list of dicts
        if isinstance(word_boundaries, dict) and 'feature' in word_boundaries:
            # Handle the Hugging Face Dataset Sequence feature
            word_boundaries = [
                {'word': word, 'start': start, 'end': end}
                for word, start, end in zip(
                    word_boundaries.get('word', []),
                    word_boundaries.get('start', []),
                    word_boundaries.get('end', [])
                )
            ]
        
        # Limit to max_words
        word_boundaries = word_boundaries[:self.max_words]
        
        # Get stress labels from emphasis_indices
        stress_labels = self._extract_stress_labels(sample, len(word_boundaries))
        
        # Pad stress labels to max_words with -1 (padding value)
        padded_stress_labels = np.full(self.max_words, -1, dtype=np.int64)
        padded_stress_labels[:len(stress_labels)] = stress_labels
        
        return {
            'audio': audio,
            'word_boundaries': word_boundaries,
            'stress_labels': torch.tensor(padded_stress_labels, dtype=torch.long)
        }
    
    def _extract_stress_labels(self, sample, num_words):
        """
        Extract stress labels from emphasis_indices.
        
        Args:
            sample: Dataset example
            num_words: Number of words to extract labels for
            
        Returns:
            Binary array of stress labels
        """
        # Initialize all words as unstressed
        stress_labels = np.zeros(num_words, dtype=np.int64)
        
        # Check if emphasis_indices exists
        if 'emphasis_indices' in sample:
            emphasis = sample['emphasis_indices']
            
            # Check if it has binary format
            if isinstance(emphasis, dict) and 'binary' in emphasis:
                binary = emphasis['binary']
                # Copy binary values (limiting to num_words)
                binary_len = min(len(binary), num_words)
                stress_labels[:binary_len] = binary[:binary_len]
                
            # Check if it has indices format
            elif isinstance(emphasis, dict) and 'indices' in emphasis:
                indices = emphasis['indices']
                # Mark stressed words
                for idx in indices:
                    if 0 <= idx < num_words:
                        stress_labels[idx] = 1
        
        return stress_labels
