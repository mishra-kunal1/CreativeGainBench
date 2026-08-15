from creativegainbench.receivers.hash_receiver import HashReceiverAgent

try:
    from creativegainbench.receivers.ollama_receiver import OllamaReceiverAgent
except ImportError:  # optional openai client
    OllamaReceiverAgent = None  # type: ignore[misc, assignment]

__all__ = ["HashReceiverAgent", "OllamaReceiverAgent"]
