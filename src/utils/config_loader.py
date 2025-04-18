# src/core/config_loader.py

import json
import os

class ConfigLoader:
    def __init__(self):
        """Initialize the config loader."""
        self.asr_config = None
        self.tts_config = None
        self.llm_config = None
        self.assistant_params = None
        print("ConfigLoader initialized.")

    def load_config_file(self, config_file):
        """Load a JSON configuration file with error handling."""
        try:
            # Ensure path exists before opening
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
        """Get default paths for config files relative to the project root."""
        # Assume this file is in src/core/. Go up two levels for project root.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_dir = os.path.join(project_root, 'src', 'config')
        return {
            'asr': os.path.join(config_dir, 'conf_asr.json'),
            'tts': os.path.join(config_dir, 'conf_tts.json'),
            'llm': os.path.join(config_dir, 'conf_llm.json')
        }

    def load_configs_from_env(self):
        """Load specific preset configs based on environment variables and apply overrides."""
        print("Loading configurations based on environment variables...")
        default_paths = self.get_default_config_paths()

        # Determine presets from environment variables (default to 'default')
        asr_preset = os.getenv('ASR_PRESET', 'default')
        tts_preset = os.getenv('TTS_PRESET', 'default')
        llm_preset = os.getenv('LLM_PRESET', 'default')
        print(f"Using presets - ASR: {asr_preset}, TTS: {tts_preset}, LLM: {llm_preset}")

        # --- Load ASR config ---
        asr_config_path = default_paths.get('asr', '')
        asr_conf_all = self.load_config_file(asr_config_path)
        self.asr_config = asr_conf_all.get(asr_preset, {})
        if not self.asr_config:
             print(f"Warning: ASR preset '{asr_preset}' not found in {asr_config_path}. Using empty config.")
        # Apply ASR environment variable overrides
        if 'ASR_MODEL' in os.environ:
            print(f"Overriding ASR model from environment: {os.environ['ASR_MODEL']}")
            self.asr_config['model'] = os.environ['ASR_MODEL']

        # --- Specific ASR Parameter Overrides ---
        asr_param_map = {
            'ASR_ENERGY_THRESHOLD': ('audio_validation', 'energy_threshold', int),
            'ASR_DEVICE': ('faster-whisper', 'device', str),
            'ASR_COMPUTE_TYPE': ('faster-whisper', 'compute_type', str),
            'ASR_MAX_RETRIES': ('audio_validation', 'max_retries', int),
            'ASR_TIMEOUT': ('audio_validation', 'timeout', int)
        }
        for env_var, (section, key, type_converter) in asr_param_map.items():
            if env_var in os.environ:
                value_str = os.environ[env_var]
                try:
                    value = type_converter(value_str)
                    if section in self.asr_config and isinstance(self.asr_config[section], dict):
                        print(f"Overriding ASR config ['{section}']['{key}'] from env var {env_var}: {value}")
                        self.asr_config[section][key] = value
                        # Special case for energy_threshold: apply to recognizer too
                        if env_var == 'ASR_ENERGY_THRESHOLD' and 'recognizer' in self.asr_config and isinstance(self.asr_config['recognizer'], dict):
                             print(f"Applying ASR_ENERGY_THRESHOLD override to ['recognizer']['energy_threshold'] as well: {value}")
                             self.asr_config['recognizer']['energy_threshold'] = value
                    else:
                        print(f"Warning: Cannot apply env var {env_var}. Section '{section}' not found or not a dictionary in ASR config.")
                except ValueError:
                    print(f"Warning: Invalid value '{value_str}' for env var {env_var}. Expected type {type_converter.__name__}. Ignoring override.")


        # --- Load TTS config ---
        tts_config_path = default_paths.get('tts', '')
        tts_conf_all = self.load_config_file(tts_config_path)
        self.tts_config = tts_conf_all.get(tts_preset, {})
        if not self.tts_config:
             print(f"Warning: TTS preset '{tts_preset}' not found in {tts_config_path}. Using empty config.")
        # Apply TTS environment variable overrides
        if 'TTS_MODEL' in os.environ:
             print(f"Overriding TTS model from environment: {os.environ['TTS_MODEL']}")
             self.tts_config['model'] = os.environ['TTS_MODEL']

        # --- Specific TTS Parameter Overrides (within 'kokoro' section) ---
        if 'kokoro' in self.tts_config and isinstance(self.tts_config['kokoro'], dict):
            tts_param_map = {
                'TTS_SPEED': ('speed', float),
                'TTS_EXPRESSIVENESS': ('expressiveness', float),
                'TTS_VARIABILITY': ('variability', float),
                'TTS_VOICE': ('voice', str)
            }
            for env_var, (key, type_converter) in tts_param_map.items():
                if env_var in os.environ:
                    value_str = os.environ[env_var]
                    try:
                        value = type_converter(value_str)
                        print(f"Overriding TTS config ['kokoro']['{key}'] from env var {env_var}: {value}")
                        self.tts_config['kokoro'][key] = value
                    except ValueError:
                         print(f"Warning: Invalid value '{value_str}' for env var {env_var}. Expected type {type_converter.__name__}. Ignoring override.")
        elif any(env_var in os.environ for env_var in ['TTS_SPEED', 'TTS_EXPRESSIVENESS', 'TTS_VARIABILITY', 'TTS_VOICE']):
             print("Warning: Cannot apply TTS parameter overrides. Section 'kokoro' not found or not a dictionary in TTS config.")

        # --- Load LLM config ---
        llm_config_path = default_paths.get('llm', '')
        llm_conf_all = self.load_config_file(llm_config_path)
        self.llm_config = llm_conf_all.get(llm_preset, {})
        if not self.llm_config:
             print(f"Warning: LLM preset '{llm_preset}' not found in {llm_config_path}. Using empty config.")
        # Apply LLM environment variable overrides (Model, API Base, System Prompt)
        if 'QUERY_LLM_MODEL' in os.environ:
             raw_model_name = os.environ['QUERY_LLM_MODEL']
             # Clean the model name: strip whitespace, remove quotes, remove comments
             cleaned_model_name = raw_model_name.split('#')[0].strip().strip('"').strip("'")
             print(f"Overriding LLM model from environment (Raw: '{raw_model_name}', Cleaned: '{cleaned_model_name}')")
             self.llm_config['model'] = cleaned_model_name
        if 'LLM_API_BASE' in os.environ:
             # Assuming API base doesn't typically have comments/quotes, but strip whitespace
             cleaned_api_base = os.environ['LLM_API_BASE'].strip()
             print(f"Overriding LLM API base from environment: {cleaned_api_base}")
             self.llm_config['api_base'] = cleaned_api_base
        # SYSTEM_PROMPT env var override (takes priority over JSON)
        if 'SYSTEM_PROMPT' in os.environ:
             raw_system_prompt = os.environ['SYSTEM_PROMPT']
             # Clean the system prompt: strip whitespace, remove quotes
             cleaned_system_prompt = raw_system_prompt.strip().strip('"').strip("'")
             print(f"Overriding system prompt from environment (Raw: '{raw_system_prompt[:50]}...', Cleaned: '{cleaned_system_prompt[:50]}...')")
             # Ensure llm_config exists before modifying
             if not isinstance(self.llm_config, dict):
                 self.llm_config = {}
             self.llm_config['system_prompt'] = cleaned_system_prompt


        print("Configurations loaded and environment overrides applied.")
        return True

    def get_assistant_parameters(self):
        """Prepare parameters needed by Alpaca based on loaded configs and environment variables."""
        if self.asr_config is None or self.tts_config is None or self.llm_config is None:
             print("Error: Configs not loaded properly. Call load_configs_from_env() first.")
             return None

        # Load assistant behavior parameters from environment variables with defaults
        try:
            duration_str = os.getenv('FIXED_DURATION')
            duration = int(duration_str) if duration_str else None # None means dynamic duration
            timeout = int(os.getenv('TIMEOUT', '5')) # Default 5 seconds
            phrase_limit = int(os.getenv('PHRASE_LIMIT', '10')) # Default 10 seconds
            print(f"Assistant parameters - Duration: {duration}, Timeout: {timeout}, Phrase Limit: {phrase_limit}")
        except ValueError as e:
            print(f"Warning: Invalid integer value in environment for TIMEOUT or PHRASE_LIMIT: {e}. Using defaults.")
            duration = None # Default to dynamic on error too
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