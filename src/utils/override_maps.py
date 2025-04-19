import os


ASR_OVERRIDE_MAP = {
    'ASR_MODEL':          {'key': 'model', 'type': str, 'clean': True, 'clean_comments': True},
    'ASR_DEVICE':         {'key': ('faster-whisper', 'device'), 'type': str, 'clean': True},
    'ASR_COMPUTE_TYPE':   {'key': ('faster-whisper', 'compute_type'), 'type': str, 'clean': True},
    'ASR_ENERGY_THRESHOLD': {'key': ('audio_validation', 'energy_threshold'), 'type': int},
    'ASR_MAX_RETRIES':    {'key': ('audio_validation', 'max_retries'), 'type': int},
    'ASR_TIMEOUT':        {'key': ('audio_validation', 'timeout'), 'type': int},
}
# Note: TTS overrides assume 'kokoro' section exists from preset
TTS_OVERRIDE_MAP = {
    'TTS_MODEL':          {'key': 'model', 'type': str, 'clean': True, 'clean_comments': True},
    'TTS_SPEED':          {'key': ('kokoro', 'speed'), 'type': float},
    'TTS_EXPRESSIVENESS': {'key': ('kokoro', 'expressiveness'), 'type': float},
    'TTS_VARIABILITY':    {'key': ('kokoro', 'variability'), 'type': float},
    'TTS_VOICE':          {'key': ('kokoro', 'voice'), 'type': str, 'clean': True},
}
LLM_OVERRIDE_MAP = {
    'QUERY_LLM_MODEL':    {'key': 'model', 'type': str, 'clean': True, 'clean_comments': True},
    'LLM_API_BASE':       {'key': 'api_base', 'type': str, 'clean': True},
    'SYSTEM_PROMPT':      {'key': 'system_prompt', 'type': str, 'clean': True},
}

def apply_overrides(loader_instance, config_dict, override_map):
    """
    Applies environment variable overrides to a configuration dictionary.

    Args:
        loader_instance: The ConfigLoader instance (to access _clean_env_var).
        config_dict (dict): The dictionary to modify.
        override_map (dict): Map from env var name to target details.
    """
    if not isinstance(config_dict, dict):
        print(f"Warning: Cannot apply overrides to non-dictionary: {type(config_dict)}")
        return

    for env_var, details in override_map.items():
        if env_var in os.environ:
            value_str = os.environ[env_var]
            key_path = details['key'] # Can be string or tuple
            type_converter = details.get('type', str)
            # Default to cleaning whitespace/quotes AND comments for all overrides
            clean = details.get('clean', True) 
            clean_comments = details.get('clean_comments', True)

            value_to_set = value_str
            if clean:
                # Use the passed ConfigLoader instance to call its cleaning method
                value_to_set = loader_instance._clean_env_var(value_to_set, remove_comments=clean_comments)

            try:
                final_value = type_converter(value_to_set)
                
                # Navigate path and set value
                target_dict = config_dict
                is_nested = isinstance(key_path, tuple)
                
                if is_nested:
                    key = key_path[-1]
                    path_keys = key_path[:-1]
                    valid_path = True
                    current_path_str = ""
                    for section_key in path_keys:
                         current_path_str += f"['{section_key}']"
                         section = target_dict.get(section_key)
                         if isinstance(section, dict):
                             target_dict = section
                         else:
                             print(f"Warning: Cannot apply env var {env_var}. Path {current_path_str} invalid or not dict in config.")
                             valid_path = False
                             break
                    if valid_path:
                         target_dict[key] = final_value
                else: # Top level key
                     key = key_path
                     config_dict[key] = final_value

            except ValueError:
                # Error during type conversion
                print(f"Warning: Invalid value format '{value_str}' for env var {env_var}. Expected type {type_converter.__name__}. Raw value after cleaning: '{value_to_set}'. Ignoring override.")
            except Exception as e:
                # Catch other potential errors
                print(f"Warning: Error applying override for env var {env_var} to path {key_path}: {e}")
