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

def apply_overrides(self, config_dict, override_map):
    for env_var, details in override_map.items():
        if env_var in os.environ:
            value_str = os.environ[env_var]
            key_path = details['key'] # Can be string or tuple
            type_converter = details.get('type', str)
            clean = details.get('clean', False)
            clean_comments = details.get('clean_comments', False)

            value_to_set = value_str
            if clean:
                value_to_set = self._clean_env_var(value_to_set, remove_comments=clean_comments)

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
                                valid_path = False
                                break
                    if valid_path:
                            target_dict[key] = final_value
                else: # Top level key
                        key = key_path
                        config_dict[key] = final_value

            except Exception as e:
                print(f"Warning: Error applying override for env var {env_var} to path {key_path}: {e}")
