from src.core.voice_assistant import VoiceAssistant
from src.rag.importdocs import *
import argparse
import json
import os


def load_config(config_file):
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {str(e)}")
        return {}

def get_default_config_paths():
    """Get default paths for config files."""
    # Get the directory of this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the src directory (assuming it's at the same level as main.py or one level up if main.py is inside src)
    # A more robust approach might be needed depending on project structure, but this covers common cases.
    src_dir = os.path.join(script_dir, 'src')
    if not os.path.isdir(src_dir):
         # If main.py is inside src, go one level up
         src_dir = os.path.join(os.path.dirname(script_dir), 'src')
         if not os.path.isdir(src_dir):
              # Fallback or raise error if src isn't found nearby
              print("Warning: Could not reliably determine 'src' directory location relative to main.py. Using script directory.")
              src_dir = script_dir # Fallback to script dir if src isn't found

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
    # Start with empty configs
    asr_config = {}
    tts_config = {}
    llm_config = {}
    
    # Load from config files
    if os.path.exists(args.asr_config):
        asr_conf_all = load_config(args.asr_config)
        if args.asr_preset in asr_conf_all:
            asr_config = asr_conf_all[args.asr_preset]
        else:
            print(f"Warning: ASR preset '{args.asr_preset}' not found. Using 'default' preset.")
            asr_config = asr_conf_all.get('default', {})
    
    if os.path.exists(args.tts_config):
        tts_conf_all = load_config(args.tts_config)
        if args.tts_preset in tts_conf_all:
            tts_config = tts_conf_all[args.tts_preset]
        else:
            print(f"Warning: TTS preset '{args.tts_preset}' not found. Using 'default' preset.")
            tts_config = tts_conf_all.get('default', {})
    
    if os.path.exists(args.llm_config):
        llm_conf_all = load_config(args.llm_config)
        if args.llm_preset in llm_conf_all:
            llm_config = llm_conf_all[args.llm_preset]
        else:
            print(f"Warning: LLM preset '{args.llm_preset}' not found. Using 'default' preset.")
            llm_config = llm_conf_all.get('default', {})
    
    # Prepare parameters for VoiceAssistant
    assistant_params = {}
    
    # Provide configs to VoiceAssistant
    assistant_params['asr_config'] = asr_config
    assistant_params['tts_config'] = tts_config
    assistant_params['llm_config'] = llm_config
    
    # Initialize the voice assistant
    assistant = VoiceAssistant(**assistant_params)

    # Start the interaction loop
    try:
        while True:
            assistant.interaction_loop(
                duration=args.fixed_duration, 
                timeout=args.timeout,
                phrase_limit=args.phrase_limit,
            )
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()