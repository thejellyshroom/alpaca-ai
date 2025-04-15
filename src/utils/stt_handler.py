import torch
import os
from faster_whisper import WhisperModel
import numpy as np
import time

class Transcriber:
    def __init__(self, config):

        config = config.get("faster-whisper", {})
        self.model_id = config.get("model_id", "Systran/faster-whisper-small")
        
        # Extract parameters from kwargs with defaults
        self.beam_size = config.get("beam_size", 5)
        self.compute_type = config.get("compute_type", "int8")  # Changed to int8 for better compatibility
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Set download_root for model files
        self.download_root = config.get("download_root")
        
        # Initialize faster-whisper model
        try:
            print(f"Initializing faster-whisper with model={self.model_id}, device={self.device}, compute_type={self.compute_type}, beam_size={self.beam_size}")
            
            self.model = WhisperModel(
                self.model_id,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root
            )
            print(f"Successfully loaded model: {self.model_id}")
            
            # Store beam size for transcription
            self.beam_size = self.beam_size
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        self.pipe = None  # Not used with faster-whisper

    def transcribe(self, audio_file):
        """Transcribe audio file to text.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Validate audio file
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                print(f"Warning: Audio file {audio_file} is empty or does not exist")
                return ""
            
            print(f"🎤 Starting transcription with model: {self.model_id.split('/')[-1]}...")
            
            start_time = time.time()

            segments, _ = self.model.transcribe(audio_file, beam_size=self.beam_size)
            text = " ".join([segment.text for segment in segments])
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"✓ Transcription complete in {duration:.2f} seconds: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            
            return text
                
        except TypeError as e:
            if "unsupported operand type(s) for *: 'NoneType'" in str(e):
                print(f"Caught NoneType error in Whisper model. This may indicate an issue with the audio input.")
                return ""
            raise
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return "" 