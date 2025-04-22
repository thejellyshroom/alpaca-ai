from ..utils.conversation_manager import ConversationManager
from ..utils.component_manager import ComponentManager
from .alpaca_interaction import AlpacaInteraction
import time

class Alpaca:
    """Container class to initialize and hold the core components."""
    def __init__(self, 
                 asr_config=None,
                 tts_config=None,
                 llm_config=None,
                 duration=None, 
                 timeout=5, 
                 phrase_limit=10,
                 mode='voice'):
        
        # Configs are passed directly to ComponentManager
        asr_config = asr_config or {}
        tts_config = tts_config or {}
        llm_config = llm_config or {}
        
        system_prompt = llm_config.get('system_prompt', "You are a helpful assistant.")
        self.conversation_manager = ConversationManager(system_prompt=system_prompt)
        self.component_manager = ComponentManager(asr_config, tts_config, llm_config, mode=mode)
        self.interaction_handler = AlpacaInteraction(self.component_manager, self.conversation_manager)

        self.duration_arg = duration
        self.timeout_arg = timeout
        self.phrase_limit_arg = phrase_limit
        