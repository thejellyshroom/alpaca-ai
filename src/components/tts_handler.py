import torch
import numpy as np
from kokoro import KPipeline

import logging
import os
import re
import random
import time


class TTSHandler:
    def __init__(self, config=None):
        config = config.get("kokoro", {})
        self.voice = config.get("voice")
        self.speed = config.get('speed')
        self.sample_rate = config.get('sample_rate', 24000)
        self.device = config.get('device', 'cpu')
        
        # Voice characteristics
        self.speech_characteristics = {
            "expressiveness": 1.0,  # 0.0-2.0, how expressive the voice is
            "variability": 0.2,     # 0.0-1.0, how much the speech speed varies
            "character": self.voice      # Voice character/persona
        }
        
        print(f"Initializing Kokoro TTS...")
        
        # Determine language code from voice prefix
        lang_code = self.voice[0]  # First letter of voice ID determines language
        self.kokoro_pipeline = KPipeline(lang_code=lang_code)
        
    def set_characteristics(self, **kwargs):
        """Update speech characteristics.
        
        Args:
            **kwargs: Characteristics to update
                - expressiveness (float): 0.0-2.0
                - variability (float): 0.0-1.0
                - character (str): Voice ID
        """
        for key, value in kwargs.items():
            if key in self.speech_characteristics:
                # Validate values
                if key == "expressiveness":
                    value = max(0.0, min(2.0, value))
                elif key == "variability":
                    value = max(0.0, min(1.0, value))
                elif key == "character" and value not in self.available_voices:
                    print(f"Warning: Invalid voice '{value}'. Using '{self.voice}'.")
                    value = self.voice
                
                # Update the characteristic
                self.speech_characteristics[key] = value
                
                # Special handling for character/voice change
                if key == "character" and value != self.voice:
                    self.voice = value
                    # Update language code if needed
                    lang_code = value[0]
                    self.kokoro_pipeline = KPipeline(lang_code=lang_code)
        
        print(f"Updated speech characteristics: {self.speech_characteristics}")

        
    def synthesize(self, text, **kwargs):
        """Convert text to speech with optional characteristics override.
        
        Args:
            text (str): Text to convert to speech
            **kwargs: Optional characteristic overrides for this utterance
                - expressiveness, variability, character
                
        Returns:
            tuple: (audio_array, sample_rate)
        """
        # Apply any temporary characteristic overrides
        if kwargs:
            temp_characteristics = self.speech_characteristics.copy()
            self.set_characteristics(**kwargs)
        
        try:
            if not text:
                return np.zeros(0, dtype=np.float32), self.sample_rate
            
            if len(text) > 200:
                sentences = self._split_into_sentences(text)
                audio_segments = []
                sample_rate = self.sample_rate
                silence_duration = kwargs.get('sentence_silence', 0.2)  # Get from kwargs or use default
                
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    try:
                        # Generate speech for each sentence
                        audio_segment = self._synthesize_single(sentence)
                        
                        if audio_segment is None:
                            print(f"Warning: Got None audio segment for sentence: {sentence}")
                            continue
                            
                        if len(audio_segment) > 0:
                            audio_segments.append(audio_segment)
                            # Add a small silence between sentences
                            silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
                            audio_segments.append(silence)
                    except Exception as e:
                        print(f"Error synthesizing sentence: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Combine all audio segments
                if audio_segments:
                    try:
                        combined_audio = np.concatenate(audio_segments)
                        return combined_audio, sample_rate
                    except Exception as e:
                        print(f"Error combining audio segments: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return np.zeros(0, dtype=np.float32), sample_rate
                else:
                    print("Warning: No audio segments were generated")
                    return np.zeros(0, dtype=np.float32), sample_rate
            else:
                audio = self._synthesize_single(text)
                if audio is None:
                    print(f"Warning: Got None audio for text: {text}")
                    return np.zeros(0, dtype=np.float32), self.sample_rate
                return audio, self.sample_rate
        except Exception as e:
            print(f"Unexpected error in speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32), self.sample_rate
        finally:
            if kwargs:
                self.speech_characteristics = temp_characteristics
    
    def _synthesize_single(self, text):
        try:
            generator = self.kokoro_pipeline(
                text,
                voice=self.voice,
                speed=self.speed,
                split_pattern=r'\n+'
            )
            
            audio_segments = []
            for _, _, audio in generator:
                if audio is None:
                    print("Warning: Received None audio from Kokoro pipeline")
                    continue
                
                if hasattr(audio, 'numpy'):
                    audio_segments.append(audio.numpy())
                elif isinstance(audio, np.ndarray):
                    audio_segments.append(audio)
                else:
                    try:
                        audio_segments.append(np.array(audio, dtype=np.float32))
                    except Exception as e:
                        print(f"Error converting audio to numpy array: {str(e)}")
                        continue
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                return combined_audio
            else:
                return np.zeros(0, dtype=np.float32)
        except Exception as e:
            print(f"Error in Kokoro speech synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(0, dtype=np.float32)
    
    def _split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences 