# src/core/conversation_manager.py

from datetime import timedelta


class ConversationManager:
    def __init__(self, system_prompt="You are a helpful assistant."):
        """Initialize the conversation history."""
        self.system_prompt = {"role": "system", "content": system_prompt}
        self.conversation_history = [self.system_prompt]
        
    def format_timedelta(delta: timedelta) -> str:
        """Formats timedelta into human-readable string (e.g., '5 minutes ago')."""
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds} seconds ago"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        else:
            days = seconds // 86400
            return f"{days} day{'s' if days > 1 else ''} ago"

    def add_user_message(self, text):
        """Add a user message to the history."""
        if text:
            self.conversation_history.append({"role": "user", "content": text})
        else:
            print("Warning: Attempted to add empty user message.")

    def add_assistant_message(self, text):
        """Add an assistant message to the history."""
        if text:
             self.conversation_history.append({"role": "assistant", "content": text})
        else:
             print("Warning: Attempted to add empty assistant message.")

    def get_history(self):
        """Return the current conversation history."""
        return self.conversation_history

    def clear_history(self, keep_system_prompt=True):
        """Clear the conversation history, optionally keeping the system prompt."""
        if keep_system_prompt:
            self.conversation_history = [self.system_prompt]
            print("Conversation history cleared (system prompt kept).")
        else:
            self.conversation_history = []
            print("Conversation history cleared completely.") 