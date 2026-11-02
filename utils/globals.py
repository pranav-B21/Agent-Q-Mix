class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self):
        self.value = 0.0


class Cost(Singleton):
    def __init__(self):
        self.value = 0.0


class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0


class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0


class LLMCalls(Singleton):
    """Per-call records for the current episode: tokens, finish_reason, reasoning fallback."""

    def __init__(self):
        self.value = []

    def reset(self):
        self.value = []


class Time(Singleton):
    def __init__(self):
        self.value = ""


class Mode(Singleton):
    def __init__(self):
        self.value = ""
