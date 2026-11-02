"""
Single-agent baseline (the paper's "Base", vanilla direct prompting): one LLM call per question,
graded by the same dataset evaluator and saved in the same format as run_qmix_eval.py.

The paper doesn't give Base's exact prompt. This uses the domain's decision constraint as the system
prompt, so answers come out in the format the grader expects, and the task as the user message.

Usage:
  python -m experiments.run_base_eval --dataset mmlu_pro --split validation \
      --llm_name openai/gpt-oss-120b --output_path out/mmlu_pro_base.json
"""

import os
import sys
import argparse
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import get_dataset
from experiments.accuracy import save_results
from experiments.run_qmix_train import DOMAIN_MAP
from llm.llm_registry import LLMRegistry
from prompt.prompt_set_registry import PromptSetRegistry
from utils.globals import PromptTokens, CompletionTokens, LLMCalls


async def evaluate(args):
    dataset = get_dataset(args.dataset, split=args.split, limit=args.limit)
    if len(dataset) == 0:
        raise SystemExit(f"No samples for {args.dataset}")
    llm = LLMRegistry.get(args.llm_name, model_name=args.llm_name)
    constraint = PromptSetRegistry.get(DOMAIN_MAP[args.dataset]).get_decision_constraint().strip()

    results = []
    for sample in dataset.samples:
        PromptTokens.instance().reset()
        CompletionTokens.instance().reset()
        LLMCalls.instance().reset()
        messages = [{"role": "system", "content": constraint}, {"role": "user", "content": sample.task}]
        try:
            answer = await llm.agen(messages)
        except Exception as e:
            print(f"[error] {sample.task_id}: {type(e).__name__}: {e}", flush=True)
            answer = ""
        results.append({
            "task_id": sample.task_id,
            "score": dataset.evaluate(answer, sample.ground_truth),
            "tokens_used": int(PromptTokens.instance().value + CompletionTokens.instance().value),
            "prediction": answer[:500],
            "full_prediction": answer,
            "ground_truth": sample.ground_truth[:200],
            "truncated_calls": sum(c.get("finish_reason") == "length" for c in LLMCalls.instance().value),
            "reasoning_fallback_calls": sum(bool(c.get("reasoning_fallback")) for c in LLMCalls.instance().value),
            "cost": 0.0,
        })
    save_results(results, args.output_path, metadata={"dataset": args.dataset, "llm": args.llm_name, "method": "base"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--llm_name", default="openai/gpt-oss-120b")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_path", required=True)
    asyncio.run(evaluate(p.parse_args()))


if __name__ == "__main__":
    main()
