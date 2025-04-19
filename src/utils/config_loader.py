# src/core/config_loader.py

import json
import os
from src.utils.override_maps import ASR_OVERRIDE_MAP, TTS_OVERRIDE_MAP, LLM_OVERRIDE_MAP
from src.utils.override_maps import apply_overrides
class ConfigLoader:
    def __init__(self):
        """Initialize the config loader."""
        self.asr_config = None
        self.tts_config = None
        self.llm_config = None
        self.assistant_params = None
        print("ConfigLoader initialized.")

    def load_config_file(self, config_file):
        try:
            if not os.path.exists(config_file):
                 print(f"Warning: Config file not found: {config_file}")
                 return {}
            with open(config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from file: {config_file}")
            return {}
        except Exception as e:
             print(f"Warning: Unexpected error loading config file {config_file}: {e}")
             return {}

    def get_default_config_paths(self):
        """Gets default paths for config files relative to the project root."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(project_root, 'src', 'config')
        return {
            'asr': os.path.join(config_dir, 'conf_asr.json'),
            'tts': os.path.join(config_dir, 'conf_tts.json'),
            'llm': os.path.join(config_dir, 'conf_llm.json')
        }

    def _clean_env_var(self, value_str, remove_comments=False):
        """Cleans environment variable string: strips whitespace, quotes, and optionally comments."""
        if not isinstance(value_str, str):
            return value_str # Return original if not a string
        
        cleaned = value_str
        if remove_comments:
            cleaned = cleaned.split('#')[0]
            
        # Strip whitespace first, then quotes
        cleaned = cleaned.strip().strip('"').strip("'")
        return cleaned

    def _load_preset_config(self, component_type, preset_name):
        default_paths = self.get_default_config_paths()
        config_path = default_paths.get(component_type)
        
        conf_all = self.load_config_file(config_path)
        config = conf_all.get(preset_name, {})
        return config



    def load_configs_from_env(self):
        print("Loading configurations and applying environment overrides...")
        
        # 1. Get presets from environment
        asr_preset = os.getenv('ASR_PRESET', 'default')
        tts_preset = os.getenv('TTS_PRESET', 'default')
        llm_preset = os.getenv('LLM_PRESET', 'default')
        print(f"Using presets - ASR: {asr_preset}, TTS: {tts_preset}, LLM: {llm_preset}")

        # 2. Load base configs from JSON using presets
        self.asr_config = self._load_preset_config('asr', asr_preset)
        self.tts_config = self._load_preset_config('tts', tts_preset)
        self.llm_config = self._load_preset_config('llm', llm_preset)
        
        # 4. Apply overrides using the helper method
        apply_overrides(self, self.asr_config, ASR_OVERRIDE_MAP)
        apply_overrides(self, self.tts_config, TTS_OVERRIDE_MAP)
        apply_overrides(self, self.llm_config, LLM_OVERRIDE_MAP)

        # 5. Handle Special Cases (after general overrides)
        # ASR Energy Threshold link
        if 'ASR_ENERGY_THRESHOLD' in os.environ:
            try:
                energy_val = int(os.environ['ASR_ENERGY_THRESHOLD']) # Already validated by _apply_overrides
                recognizer_section = self.asr_config.get('recognizer')
                if isinstance(recognizer_section, dict):
                    print(f"Applying linked ASR_ENERGY_THRESHOLD override to ['recognizer']['energy_threshold']: {energy_val}")
                    recognizer_section['energy_threshold'] = energy_val
                else:
                    pass
            except ValueError:
                 pass 
            except Exception as e:
                 print(f"Error applying linked ASR_ENERGY_THRESHOLD to recognizer section: {e}")

        print("Configurations loaded successfully.")
        return True
        
    def get_assistant_parameters(self):
        """Prepare parameters needed by Alpaca based on loaded configs and environment variables."""
        if self.asr_config is None or self.tts_config is None or self.llm_config is None:
             print("Error: Configs not loaded properly. Call load_configs_from_env() first.")
             return None

        # Load assistant behavior parameters from environment variables with defaults
        try:
            duration_str = self._clean_env_var(os.getenv('FIXED_DURATION'), remove_comments=True)
            timeout_str = self._clean_env_var(os.getenv('TIMEOUT', '5'), remove_comments=True)
            phrase_limit_str = self._clean_env_var(os.getenv('PHRASE_LIMIT', '10'), remove_comments=True)
            
            duration = int(duration_str) if duration_str else None # None means dynamic duration
            timeout = int(timeout_str) # Default 5 seconds
            phrase_limit = int(phrase_limit_str) # Default 10 seconds
            
            print(f"Assistant parameters - Duration: {duration}, Timeout: {timeout}, Phrase Limit: {phrase_limit}")
        except ValueError as e:
            print(f"Warning: Invalid integer value in environment for DURATION, TIMEOUT or PHRASE_LIMIT after cleaning. Using defaults. Error: {e}")
            duration = None # Default to dynamic on error too
            timeout = 5
            phrase_limit = 10
        except Exception as e:
             print(f"Warning: Unexpected error reading assistant behavior env vars. Using defaults. Error: {e}")
             duration = None
             timeout = 5
             phrase_limit = 10

        # Parameters for Alpaca __init__
        params = {
             'asr_config': self.asr_config,
             'tts_config': self.tts_config,
             'llm_config': self.llm_config,
             'duration': duration,
             'timeout': timeout,
             'phrase_limit': phrase_limit
        }
        self.assistant_params = params
        return params

    def load_all(self):
        """Load all configurations from env/JSON. Returns assistant parameters."""
        if not self.load_configs_from_env():
             return None # Config loading failed
        return self.get_assistant_parameters() 