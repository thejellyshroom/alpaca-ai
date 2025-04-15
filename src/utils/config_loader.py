# src/core/config_loader.py

import json
import os
import argparse

class ConfigLoader:
    def __init__(self):
        """Initialize the config loader."""
        self.args = None
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

    def parse_arguments(self):
        """Parse command line arguments using argparse."""
        default_paths = self.get_default_config_paths()
        parser = argparse.ArgumentParser(description='AI Voice Assistant')
        
        # Assistant behavior args
        parser.add_argument('--fixed-duration', type=int, help='Use fixed duration recording instead of dynamic listening')
        parser.add_argument('--timeout', type=int, default=5, help='Maximum seconds to wait for speech before giving up')
        parser.add_argument('--phrase-limit', type=int, default=10, help='Maximum seconds for a single phrase')

        # Configuration file paths
        parser.add_argument('--asr-config', type=str, default=default_paths.get('asr', ''), help='Path to ASR configuration file')
        parser.add_argument('--tts-config', type=str, default=default_paths.get('tts', ''), help='Path to TTS configuration file')
        parser.add_argument('--llm-config', type=str, default=default_paths.get('llm', ''), help='Path to LLM configuration file')

        # Preset selection
        parser.add_argument('--asr-preset', type=str, default='default', help='ASR preset to use')
        parser.add_argument('--tts-preset', type=str, default='default', help='TTS preset to use')
        parser.add_argument('--llm-preset', type=str, default='default', help='LLM preset to use')

        # Add system prompt argument
        parser.add_argument('--system-prompt', type=str, default=None, help='Override the default system prompt for the LLM')

        self.args = parser.parse_args()
        print(f"Arguments parsed: {self.args}")

    def load_configs_from_args(self):
        """Load specific preset configs based on parsed arguments."""
        if not self.args:
            print("Error: Arguments not parsed yet. Call parse_arguments() first.")
            return False
        
        print("Loading configurations based on arguments...")
        
        # Load ASR config
        asr_conf_all = self.load_config_file(self.args.asr_config)
        self.asr_config = asr_conf_all.get(self.args.asr_preset, {})
        if not self.asr_config:
             print(f"Warning: ASR preset '{self.args.asr_preset}' not found in {self.args.asr_config}. Using empty config.")

        # Load TTS config
        tts_conf_all = self.load_config_file(self.args.tts_config)
        self.tts_config = tts_conf_all.get(self.args.tts_preset, {})
        if not self.tts_config:
             print(f"Warning: TTS preset '{self.args.tts_preset}' not found in {self.args.tts_config}. Using empty config.")

        # Load LLM config
        llm_conf_all = self.load_config_file(self.args.llm_config)
        self.llm_config = llm_conf_all.get(self.args.llm_preset, {})
        if not self.llm_config:
             print(f"Warning: LLM preset '{self.args.llm_preset}' not found in {self.args.llm_config}. Using empty config.")
             
        # Override system prompt if provided via args
        if self.args.system_prompt:
             print(f"Overriding system prompt from args: '{self.args.system_prompt[:50]}...'")
             # Ensure llm_config exists before modifying
             if not isinstance(self.llm_config, dict):
                 self.llm_config = {}
             self.llm_config['system_prompt'] = self.args.system_prompt
             
        print("Configurations loaded.")
        return True
        
    def get_assistant_parameters(self):
        """Prepare parameters needed by VoiceAssistant based on loaded configs/args."""
        if not self.args or self.asr_config is None or self.tts_config is None or self.llm_config is None:
             print("Error: Arguments/Configs not loaded properly.")
             return None
         
        # Parameters for VoiceAssistant __init__ and main_loop
        params = {
             'asr_config': self.asr_config,
             'tts_config': self.tts_config,
             'llm_config': self.llm_config,
             'duration': self.args.fixed_duration,
             'timeout': self.args.timeout,
             'phrase_limit': self.args.phrase_limit
        }
        self.assistant_params = params
        return params

    def load_all(self):
        """Parse args and load all configurations. Returns assistant parameters."""
        self.parse_arguments()
        if not self.load_configs_from_args():
             return None # Config loading failed
        return self.get_assistant_parameters() 