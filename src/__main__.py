"""Entry point for running constrained tool calling inference."""

import argparse
import json
from pathlib import Path
import sys
from typing import List

from llm_sdk import Small_LLM_Model
from src.engine import ConstrainedEngine
from src.models import FunctionDefinition, ResultEntry, TestCase, ToolCall
from src.vocab import VocabManager


def parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments matching Section IV.3.2 of the subject."""
    parser = argparse.ArgumentParser(
        description="Run constrained tool-calling with deterministic decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to save the generated function calling results.",
    )
    return parser.parse_args()


def load_tools(path: str) -> List[FunctionDefinition]:
    """Loads and validates function definitions from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Functions file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Functions definition file must contain a JSON list.")
    return [FunctionDefinition.model_validate(item) for item in data]


def load_tests(path: str) -> List[TestCase]:
    """Loads and validates test prompts from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input test file not found: {path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input test file must contain a JSON list.")
    return [TestCase.model_validate(item) for item in data]


def main() -> None:
    """Initializes components and triggers the generation pipeline."""
    args = parse_arguments()

    tools: List[FunctionDefinition] = load_tools(args.functions_definition)
    tests: List[TestCase] = load_tests(args.input)

    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab_mgr = VocabManager(vocab_path)

    engine = ConstrainedEngine(model=model, vocab_manager=vocab_mgr)
    results: List[ResultEntry] = []
    errors: List[str] = []

    for index, test in enumerate(tests, start=1):
        try:
            call: ToolCall = engine.generate(
                user_prompt=test.prompt, function_definitions=tools
            )
            entry = ResultEntry(
                prompt=test.prompt,
                name=call.name,
                parameters=call.parameters,
            )
        except Exception as err:
            errors.append(
                f"prompt {index} ({test.prompt!r}): {err}"
            )
            continue

        results.append(entry)
        print(entry.model_dump_json(indent=2))

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(
            "Generation failed for one or more prompts:\n"
            f"{details}"
        )

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    output_data = [item.model_dump() for item in results]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)
