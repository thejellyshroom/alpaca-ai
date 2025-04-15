from src.core.voice_assistant import VoiceAssistant
from src.rag.importdocs import *
import argparse
import json
import os


def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def get_default_config_paths():
    """Get default paths for config files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(script_dir, 'src')

    return {
        'asr': os.path.join(src_dir, 'config/conf_asr.json'),
        'tts': os.path.join(src_dir, 'config/conf_tts.json'),
        'llm': os.path.join(src_dir, 'config/conf_llm.json')
    }

def main():
    # Get default config paths
    default_config_paths = get_default_config_paths()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Voice Assistant')
    parser.add_argument('--fixed-duration', type=int, help='Use fixed duration recording instead of dynamic listening')
    parser.add_argument('--timeout', type=int, default=5, help='Maximum seconds to wait for speech before giving up')
    parser.add_argument('--phrase-limit', type=int, default=10, help='Maximum seconds for a single phrase')

    # Configuration file options
    parser.add_argument('--config', type=str, help='Path to global configuration file')
    parser.add_argument('--asr-config', type=str, default=default_config_paths['asr'], help='Path to ASR configuration file')
    parser.add_argument('--tts-config', type=str, default=default_config_paths['tts'], help='Path to TTS configuration file')
    parser.add_argument('--llm-config', type=str, default=default_config_paths['llm'], help='Path to LLM configuration file')
    
    # Preset selection options
    parser.add_argument('--asr-preset', type=str, default='default', help='ASR preset to use')
    parser.add_argument('--tts-preset', type=str, default='default', help='TTS preset to use')
    parser.add_argument('--llm-preset', type=str, default='default', help='LLM preset to use')

    args = parser.parse_args()
    
    # run importdocs.py
    importdocs()
    
    # Load configurations
    asr_config = {}
    tts_config = {}
    llm_config = {}
    
    # Load from config files
    asr_conf_all = load_config(args.asr_config)
    asr_config = asr_conf_all[args.asr_preset]
    
    tts_conf_all = load_config(args.tts_config)
    tts_config = tts_conf_all[args.tts_preset]

    llm_conf_all = load_config(args.llm_config)
    llm_config = llm_conf_all[args.llm_preset]
    
    # Prepare parameters for VoiceAssistant
    assistant_params = {}
    
    # Provide configs to VoiceAssistant
    assistant_params['asr_config'] = asr_config
    assistant_params['tts_config'] = tts_config
    assistant_params['llm_config'] = llm_config
    
    # Initialize the voice assistant
    assistant = VoiceAssistant(**assistant_params)

    # Start the interaction loop using the new main_loop method
    assistant.main_loop(
        duration=args.fixed_duration,
        timeout=args.timeout,
        phrase_limit=args.phrase_limit,
    )

if __name__ == "__main__":
    main()