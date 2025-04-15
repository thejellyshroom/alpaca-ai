from ..utils.helper_functions import *
from ..utils.conversation_manager import ConversationManager
from ..utils.component_manager import ComponentManager
from .alpaca_interaction import InteractionHandler
import time
import gc # Needed for garbage collection

class Alpaca:
    """Container class to initialize and hold the core components."""
    def __init__(self, 
                 asr_config=None,
                 tts_config=None,
                 llm_config=None,
                 duration=None, 
                 timeout=5, 
                 phrase_limit=10):
        
        # Configs are passed directly to ComponentManager
        asr_config = asr_config or {}
        tts_config = tts_config or {}
        llm_config = llm_config or {}
        
        # Extract system prompt (could move to ConfigLoader)
        system_prompt = llm_config.get('system_prompt', "You are a helpful assistant.")
        self.conversation_manager = ConversationManager(system_prompt=system_prompt)

        # Initialize Component Manager (which loads handlers)
        self.component_manager = ComponentManager(asr_config, tts_config, llm_config)
        
        # Initialize Interaction Handler, passing the managers
        self.interaction_handler = InteractionHandler(self.component_manager, self.conversation_manager)

        # Store loop parameters (originating from ConfigLoader -> main.py)
        # These might be passed directly to interaction_handler.run_single_interaction in main.py
        self.duration_arg = duration
        self.timeout_arg = timeout
        self.phrase_limit_arg = phrase_limit
        
        print("\nAI Voice assistant Core Initialized and Ready!")
        # Summary is printed by ComponentManager
        
    # --- All functional methods (listen, process, speak, loops) are removed --- 
    # --- Logic is now in InteractionHandler or main.py --- 

    # Potential future methods: 
    # - Methods to reload specific components? (e.g., reload_tts())
    # - Methods to get status?

    def main_loop(self):
        """The main loop calling interaction_loop and handling cleanup via ComponentManager."""
        print(f"Starting main loop with timeout={self.timeout_arg}, phrase_limit={self.phrase_limit_arg}, duration={self.duration_arg}")
        try:
            while True:
                user_input_status, assistant_output = self.interaction_handler.run_single_interaction(
                    duration=self.duration_arg,
                    timeout=self.timeout_arg,
                    phrase_limit=self.phrase_limit_arg
                )
                if user_input_status == "ERROR":
                    print(f"Recovering from interaction error: {assistant_output}")
                    time.sleep(2)
                elif user_input_status == "INTERRUPTED":
                     print("Interaction interrupted, starting new loop.")

        except KeyboardInterrupt:
             print("\nExiting Voice assistant...")
        finally:
            print("Performing final cleanup via ComponentManager...")
            if hasattr(self, 'component_manager') and self.component_manager:
                 self.component_manager.cleanup()
            # Also cleanup conversation manager if needed (though less critical)
            if hasattr(self, 'conversation_manager'):
                 del self.conversation_manager
                 self.conversation_manager = None
            gc.collect()
            print("Cleanup complete.")
            