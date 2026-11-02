from utils.globals import Cost, PromptTokens, CompletionTokens, LLMCalls

TOKEN_COSTS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "deepseek-chat": {"input": 0.00014, "output": 0.00028},
    "deepseek-reasoner": {"input": 0.00055, "output": 0.0022},
    "qwen-turbo": {"input": 0.0003, "output": 0.0006},
    "qwen-plus": {"input": 0.0008, "output": 0.002},
    "qwen-max": {"input": 0.002, "output": 0.006},
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def cost_count(prompt: str, response: str, model_name: str, usage: dict = None, **call_info):
    """Count the server's reported usage when it gives one, else estimate from visible text.

    The estimate misses hidden reasoning (a ~1.7x undercount on gpt-oss-120b), so each call
    records whether it was estimated.
    """
    if usage and usage.get("prompt_tokens") is not None:
        prompt_len = usage["prompt_tokens"]
        completion_len = usage.get("completion_tokens") or 0
    else:
        prompt_len = estimate_tokens(prompt)
        completion_len = estimate_tokens(response)
    LLMCalls.instance().value.append({
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_len,
        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "estimated": not usage,
        **call_info,
    })

    costs = TOKEN_COSTS.get(model_name, {"input": 0.0, "output": 0.0})
    price = prompt_len * costs["input"] / 1000 + completion_len * costs["output"] / 1000

    Cost.instance().value += price
    PromptTokens.instance().value += prompt_len
    CompletionTokens.instance().value += completion_len

    return price, prompt_len, completion_len
