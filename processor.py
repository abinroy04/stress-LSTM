import torch
import re
from datasets import load_dataset
from transformers import pipeline
from scipy.signal import resample
import os

class DSProcessor:
    """
    Dataset processor for speech emphasis detection tasks.
    
    Handles dataset loading, tokenization, and preparing
    training examples with emphasis annotations.
    
    Attributes:
        processor: Simple processor for text tokenization
        ds_name: HuggingFace dataset name
        hyperparameters: Dictionary of configuration parameters
        hf_token: HuggingFace API token for accessing datasets
    """
    def __init__(
        self,
        processor,
        hyperparameters,
        ds_name,
        hf_token
    ):
        self.processor = processor
        self.ds_name = ds_name
        self.hyperparameters = hyperparameters
        self.hf_token = hf_token

    def load_intonation_dataset(self):
        """
        Load a dataset from the Hugging Face Hub.
        
        Returns:
            Dataset object loaded from Hugging Face Hub
        """
        print(f"🔧 DEBUG: Loading dataset: {self.ds_name}")
        
        # Load our dataset from the Hugging Face Hub
        intonation_dataset = load_dataset(
            self.ds_name, 
            trust_remote_code=True, 
            token=self.hf_token
        )
    
        return intonation_dataset

    def map_words_to_tokens(self, example, transcription_column_name):
        """
        Maps words in the transcription to their corresponding token IDs.
        """
        # This function returns a dictionary where keys (words only from alphabet) are mapped to token IDS (values) in the transcription.
        # e.g. map_dict = {'Kitty': [42, 9760], 'smiled': [13541], 'and': [290], 'replied': [8712], 'Thank': [10449], 'you': [345], 'Spot': [15899]} ->
        # returned value:
        # keys: ['Kitty', 'smiled', 'and', 'replied', 'Thank', 'you', 'Spot']
        # values: [[42, 9760], [13541], [290], [8712], [10449], [345], [15899]]
        def contains_no_alpha(s):
            return not re.search(r"[a-zA-Z\']", s)

        def remove_non_alpha(s):
            return re.sub(r"[^a-zA-Z\']", "", s)

        tokens = self.processor.tokenizer.tokenize(example[transcription_column_name])
        tokens_ids = self.processor.tokenizer.convert_tokens_to_ids(tokens)

        map_dict = {}
        current_word = tokens[0]
        current_words_tokens = [tokens_ids[0]]
        dict_elem = 0

        for token_ids, token in zip(tokens_ids[1:], tokens[1:]):
            if token.startswith("Ġ"):
                if current_word:
                    map_dict[f"{remove_non_alpha(current_word)} {dict_elem}"] = (
                        current_words_tokens
                    )
                    current_word = ""
                    current_words_tokens = []
                    dict_elem += 1
                # start a new word (remove the leading 'Ġ')
                current_word = token[1:]
            else:
                # continue the current word
                current_word += token
            # if we came across a token that contains no alphabet characters, we skip it, except for commas
            if not contains_no_alpha(token):
                current_words_tokens.append(token_ids)

        # Add the last word
        if current_word:
            map_dict[f"{remove_non_alpha(current_word)} {dict_elem}"] = (
                current_words_tokens
            )

        correct_map_dict = {
            f"map_dict_{transcription_column_name}": {
                "keys": [str(key).split(" ")[0] for key in map_dict.keys()],
                "values": map_dict.values(),
            }
        }
        example.update(correct_map_dict)
        return example

    def emphasized_tokens(self, example, 
                          transcription_column_name,
                          emphasis_indices_column_name="emphasis_indices"):
        """
        Creates a binary vector marking which tokens are emphasized in the transcription.
        
        Handles different formats of emphasis annotations:
        1. List of indices indicating emphasized word positions
        2. Binary vector with 1s for emphasized words and 0s otherwise
        3. Dictionary with 'binary' key containing binary vector
        
        Args:
            example: Dataset example containing transcription and emphasis indices
            transcription_column_name: Name of the column containing transcription text
            emphasis_indices_column_name: Name of the column containing emphasis annotations
            
        Returns:
            Example with added labels_head_{transcription_column_name} field containing 
            a binary tensor with 1s for emphasized tokens and 0s otherwise
        """
        # This function returns a binary vector with 1 entries for emphasized tokens in the transcription (including special tokens with 0).
        curr_tokenized_sentence = self.processor.tokenizer(
            example[transcription_column_name]
        ).input_ids
        curr_values = example[f"map_dict_{transcription_column_name}"]["values"]
        
        # Handle different emphasis indices formats
        emphasis_data = example[emphasis_indices_column_name]
        
        # Handle different emphasis annotation formats
        if isinstance(emphasis_data, dict) and 'binary' in emphasis_data:
            binary_emphasis = emphasis_data['binary']
        elif isinstance(emphasis_data, list) and len(emphasis_data) > 0:
            # Check if it's a list of indices or a binary vector
            if all(isinstance(x, int) and x < len(curr_values) for x in emphasis_data):
                binary_emphasis = [0] * len(curr_values)
                for idx in emphasis_data:
                    if idx < len(binary_emphasis):
                        binary_emphasis[idx] = 1
            else:
                binary_emphasis = emphasis_data
        else:
            # Default: no emphasis
            binary_emphasis = [0] * len(curr_values)
        
        # Ensure binary_emphasis matches number of words
        if len(binary_emphasis) != len(curr_values):
            # Truncate or pad to match
            if len(binary_emphasis) > len(curr_values):
                binary_emphasis = binary_emphasis[:len(curr_values)]
            else:
                binary_emphasis.extend([0] * (len(curr_values) - len(binary_emphasis)))
            
        # Create emphasized token indices
        concatenated_values = []
        for i, is_emphasized in enumerate(binary_emphasis):
            if is_emphasized == 1 and i < len(curr_values):
                concatenated_values.extend(curr_values[i])
        
        # Map tokens to binary emphasis
        j = 0
        emphasized_words = []
        for token in curr_tokenized_sentence:
            binary = 0
            if j < len(concatenated_values):
                if token == concatenated_values[j]:
                    binary = 1
                    j += 1
            emphasized_words.append(binary)
            
        example[f"labels_head_{transcription_column_name}"] = torch.tensor(emphasized_words)
        return example

    def prepare_dataset(self, example, transcription_column_name):
        """
        Prepares the final dataset example by adding token labels.
        
        Tokenizes the transcription text and adds the resulting token IDs as labels.
        Verifies that the emphasis head labels match the length of token labels.
        
        Args:
            example: Dataset example containing the transcription
            transcription_column_name: Name of the column containing transcription text
            
        Returns:
            Example with added labels_{transcription_column_name} field containing 
            tokenized transcription IDs
        """
        example[f"labels_{transcription_column_name}"] = self.processor.tokenizer(example[transcription_column_name]).input_ids
        assert len(example[f"labels_head_{transcription_column_name}"]) == len(example[f"labels_{transcription_column_name}"])
        return example
    
    def aligned_whisper_transcriptions(self, example, model):
        """
        Prepares audio features and ensures existing word boundaries are used.
        
        Instead of using Whisper for transcription, this function now:
        1. Simply adds wav2vec features to the example
        2. Uses the existing word boundaries from the dataset
        3. Doesn't modify the transcription
        
        Args:
            example: Dataset example containing audio and transcription
            model: Model with wav2vec processor
            
        Returns:
            Example with added wav2vec_inputs and preserved word_boundaries
        """
        # Resample audio to 16kHz if necessary
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        
        if sampling_rate != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio_array) * 16000 / sampling_rate)
            audio_array = resample(audio_array, num_samples)
        
        # Process for wav2vec only
        try:
            if hasattr(model, 'wav2vec_processor'):
                example["wav2vec_inputs"] = model.wav2vec_processor(
                    audio_array,
                    return_tensors="pt",
                    sampling_rate=16000,
                ).input_values
            else:
                # Create placeholder
                example["wav2vec_inputs"] = torch.zeros((1, len(audio_array)))
                
            # Add a dummy input_features field for compatibility
            example["input_features"] = example["wav2vec_inputs"]
        except Exception as e:
            print(f"Warning: Error processing wav2vec features: {e}. Creating placeholder.")
            example["wav2vec_inputs"] = torch.zeros((1, len(audio_array)))
            example["input_features"] = torch.zeros((1, len(audio_array)))
        
        # Preserve existing word boundaries
        if "word_boundaries" not in example:
            print("Warning: No word boundaries found in example. Creating empty boundaries.")
            example["word_boundaries"] = []
            
        # Use the existing transcription as is
        example['aligned_whisper_transcriptions'] = example.get('transcription', '')
        
        return example
    


    def filter_incorrect_transcription(self, example, transcription_column_name):
        """
        Filters out examples where the token mapping doesn't match word count.
        """
        transcription = example[transcription_column_name]
        values = [val for val in example[f"map_dict_{transcription_column_name}"]["values"] if len(val) != 0]
        return len(transcription.split(" ")) == len(values)

    def preprocess(self, 
                   model,
                   transcription_column_name,
                   emphasis_indices_column_name="emphasis_indices"):
        """
        Full preprocessing pipeline for the dataset.
        
        Now simplified to:
        1. Extract wav2vec features directly
        2. Use existing word boundaries from the dataset
        3. Map words to tokens
        4. Create binary emphasis vectors
        5. Prepare final dataset format
        
        Args:
            model: Model with wav2vec processor
            transcription_column_name: Name of the column containing transcription text
            emphasis_indices_column_name: Name of the column containing emphasis annotations
            
        Returns:
            Fully preprocessed dataset ready for training
        """
        proccess_methods = {
            "aligned_whisper_transcriptions": lambda example: self.aligned_whisper_transcriptions(example, model),
            "filter_misaligned_samples": lambda example: self.filter_misaligned_samples(example, transcription_column_name),
            "map_words_to_tokens": lambda example: self.map_words_to_tokens(example, transcription_column_name=transcription_column_name),
            "filter_incorrect_transcription": lambda example: self.filter_incorrect_transcription(example, transcription_column_name=transcription_column_name),
            "emphasized_tokens": lambda example: self.emphasized_tokens(example, emphasis_indices_column_name=emphasis_indices_column_name, transcription_column_name=transcription_column_name),
            "preserve_word_boundaries": lambda example: self.preserve_word_boundaries(example),
            "prepare_dataset": lambda example: self.prepare_dataset(example, transcription_column_name=transcription_column_name)
        }
        
        # Apply aligned transcription first
        print("🔧 DEBUG: Starting preprocessing pipeline...")
        ds1 = self.load_intonation_dataset().map(proccess_methods["aligned_whisper_transcriptions"], num_proc=1)
        
        if emphasis_indices_column_name != 'emphasis_indices' and "emphasis_indices" in ds1.column_names:
            ds1 = ds1.rename_column("emphasis_indices", emphasis_indices_column_name)
        
        # Continue with the rest of the pipeline
        ds3 = ds1.map(proccess_methods["map_words_to_tokens"], num_proc=1)
        
        # Handle the complex nested list structure in map_dict_transcription.values
        def normalize_map_dict_structure(example):
            if f"map_dict_{transcription_column_name}" in example:
                map_dict = example[f"map_dict_{transcription_column_name}"]
                if "values" in map_dict and isinstance(map_dict["values"], list):
                    # Convert any non-list values to lists for consistency
                    for i, val in enumerate(map_dict["values"]):
                        # Ensure each value is properly wrapped as a list
                        if not isinstance(val, list) and not isinstance(val, torch.Tensor):
                            map_dict["values"][i] = [val]
                        # Convert torch.Tensor to list for serialization
                        elif isinstance(val, torch.Tensor):
                            map_dict["values"][i] = val.tolist()
            return example
        
        ds3 = ds3.map(normalize_map_dict_structure, num_proc=1)
        
        ds4 = ds3.map(proccess_methods["emphasized_tokens"], num_proc=1)
        ds5 = ds4.map(proccess_methods["prepare_dataset"], num_proc=1)
        return ds5

    def get_train_dataset(self, 
                      model, 
                      transcription_column_name, 
                      emphasis_indices_column_name="emphasis_indices",
                      columns_to_remove=["filename"]
                     ):
        """
        Get the final preprocessed training dataset.
        """
        preproc_intonation_dataset = self.preprocess(
            model=model,
            emphasis_indices_column_name=emphasis_indices_column_name,
            transcription_column_name=transcription_column_name
        )
        
        # Fix any complex nested structures before saving/loading
        def ensure_serializable_structures(example):
            # Fix map_dict values which might be tensor sequences
            if f"map_dict_{transcription_column_name}" in example:
                map_dict = example[f"map_dict_{transcription_column_name}"]
                if "values" in map_dict and isinstance(map_dict["values"], list):
                    # Make sure all values are Python lists, not tensors or other complex types
                    map_dict["values"] = [
                        v.tolist() if hasattr(v, 'tolist') else list(v) if hasattr(v, '__iter__') else [v] 
                        for v in map_dict["values"]
                    ]
            return example
        
        preproc_intonation_dataset = preproc_intonation_dataset.map(
            ensure_serializable_structures, 
            num_proc=1,
            desc="Ensuring serializable data structures"
        )
        
        # Convert any remaining problematic tensor sequences to regular sequences
        from datasets import Sequence, Value
        for col in preproc_intonation_dataset.column_names:
            if col.startswith("map_dict_"):
                # Create a temporary column with properly structured data
                preproc_intonation_dataset = preproc_intonation_dataset.add_column(
                    f"{col}_temp", preproc_intonation_dataset[col]
                )
                # Remove the original column and rename the temp column
                preproc_intonation_dataset = preproc_intonation_dataset.remove_columns([col])
                preproc_intonation_dataset = preproc_intonation_dataset.rename_column(
                    f"{col}_temp", col
                )
        
        if columns_to_remove:
            # Filter out word_boundaries from columns to remove
            safe_columns_to_remove = [col for col in columns_to_remove if col != "word_boundaries"]
            existing_cols = [col for col in safe_columns_to_remove if col in preproc_intonation_dataset.column_names]
            if existing_cols:
                preproc_intonation_dataset = preproc_intonation_dataset.remove_columns(existing_cols)
        
        preproc_intonation_dataset.set_format("torch")
        return preproc_intonation_dataset

def preserve_word_boundaries(self, example):
    """
    Explicitly preserve word boundaries in the processed example.
    """
    # If word_boundaries exist, ensure they are properly structured
    if "word_boundaries" in example:
        # Just accessing it is enough to ensure it's included in the output
        word_boundaries = example["word_boundaries"]
            
            # Check if we need to restructure the boundaries
        if isinstance(word_boundaries, dict) and not any(isinstance(k, str) and k in ['word', 'start', 'end'] for k in word_boundaries.keys()):
                # Convert dict-of-dicts format to list-of-dicts format if needed
            structured_boundaries = []
            for key in sorted(word_boundaries.keys()):
                if isinstance(word_boundaries[key], dict) and 'word' in word_boundaries[key]:
                    structured_boundaries.append(word_boundaries[key])
                
                # Only replace if we successfully restructured
            if structured_boundaries:
                example["word_boundaries"] = structured_boundaries
                print(f"Restructured word_boundaries from dict to list format: {len(structured_boundaries)} items")
        else:
            # If missing, add an empty list placeholder
            example["word_boundaries"] = []
            print("Added empty word_boundaries placeholder")
        
    return example
        
def extract_word_boundaries(self, example, model, transcription_column_name):
        """
        Extract word boundaries by running ASR pipeline with word timestamps.
        """
        # Prepare and resample audio
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        if sr != 16000:
            from scipy.signal import resample
            num = int(len(audio) * 16000 / sr)
            audio = resample(audio, num)
        
        # Ensure audio is a numpy array, not a list
        if isinstance(audio, list):
            import numpy as np
            audio = np.array(audio, dtype=np.float32)
        
        # Run ASR pipeline with word timestamps
        asr = pipeline(
            "automatic-speech-recognition",
            model=model.whisper_backbone_name,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            return_timestamps="word"
        )
        
        try:
            # Try direct processing first
            result = asr(audio, chunk_length_s=30)
        except (TypeError, ValueError) as e:
            print(f"Error in ASR processing: {str(e)}")
            print(f"Audio type: {type(audio)}, shape: {getattr(audio, 'shape', 'unknown')}")
            
            # Skip word boundaries extraction for this example
            example["word_boundaries"] = {"words": []}
            return example
        
        # Build boundaries in raw audio frames (16 kHz)
        boundaries = []
        for chunk in result.get("chunks", []):
            if "start" in chunk and "end" in chunk:
                s = chunk["start"]
                e = chunk["end"]
            elif "timestamp" in chunk:
                # timestamp is a tuple or list [start, end]
                s, e = chunk["timestamp"]
            else:
                # unexpected format, skip
                continue
            boundaries.append({
                "start_frame": int(s * 16000),
                "end_frame":   int(e * 16000),
            })
        
        example["word_boundaries"] = {"words": boundaries}
        return example

def preserve_word_boundaries(self, example):
    """
        Explicitly preserve word boundaries in the processed example.
        
        Some dataset operations may inadvertently drop columns. This method
        ensures word_boundaries are kept and properly formatted.
        
        Args:
            example: Dataset example, potentially containing word_boundaries
            
        Returns:
            Example with preserved word_boundaries
    """
    # If word_boundaries exist, ensure they are properly structured
    if "word_boundaries" in example:
        # Just accessing it is enough to ensure it's included in the output
        word_boundaries = example["word_boundaries"]
            
            # Check if we need to restructure the boundaries
        if isinstance(word_boundaries, dict) and not any(isinstance(k, str) and k in ['word', 'start', 'end'] for k in word_boundaries.keys()):
                # Convert dict-of-dicts format to list-of-dicts format if needed
            structured_boundaries = []
            for key in sorted(word_boundaries.keys()):
                if isinstance(word_boundaries[key], dict) and 'word' in word_boundaries[key]:
                    structured_boundaries.append(word_boundaries[key])
                
                # Only replace if we successfully restructured
            if structured_boundaries:
                example["word_boundaries"] = structured_boundaries
                print(f"Restructured word_boundaries from dict to list format: {len(structured_boundaries)} items")
        else:
            # If missing, add an empty list placeholder
            example["word_boundaries"] = []
            print("Added empty word_boundaries placeholder")
        
    return example
