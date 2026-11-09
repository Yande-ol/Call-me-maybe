*This project has been created as part of the 42 curriculum by Yande-ol.*

# Call Me Maybe

## Description

Call Me Maybe is a Python implementation of function calling for a small language
model. It receives natural-language prompts and a list of available function
definitions, then produces a structured JSON tool call instead of a conversational
answer.

For example, the prompt `What is the sum of 2 and 3?` can be converted into:

```json
{
	"name": "fn_add_numbers",
	"parameters": {
		"a": 2.0,
		"b": 3.0
	}
}
```

The project uses the provided `Small_LLM_Model` SDK with the default
`Qwen/Qwen3-0.6B` model. The model chooses the function and extracts the argument
values. The program does not use a list of hardcoded prompt-to-function rules.

The main technical goal is reliable structured generation. Instead of asking the
model to produce JSON and trusting the result, the project applies constrained
decoding. A grammar state machine describes the valid JSON structure, and the
generation loop masks the model logits so that only valid next tokens can be
selected. The generated call is then validated against the function schema using
Pydantic.

## Features

- Function selection performed by the language model.
- Constrained generation of JSON objects with `name` and `parameters` fields.
- Support for string, number, integer, and boolean parameters.
- Support for negative numbers, decimal numbers, empty strings, escaped quotes,
	backslashes, regular expressions, and punctuation inside string values.
- Pydantic validation of function definitions, test cases, tool calls, and output
	entries.
- Clear errors for missing files, malformed JSON, invalid schemas, and generation
	failures.
- Configurable input, function-definition, and output paths.
- Deterministic greedy token selection after applying the grammar mask.

## Project Structure

```text
.
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk/
│   └── __init__.py
├── src/
│   ├── __main__.py
│   ├── engine.py
│   ├── grammar.py
│   ├── models.py
│   └── vocab.py
├── Makefile
├── pyproject.toml
└── uv.lock
```

### Main modules

- `src/__main__.py` parses command-line arguments, loads JSON files, initializes
	the model and vocabulary, runs every test prompt, and writes the result file.
- `src/engine.py` implements the token-by-token generation loop. It obtains model
	logits, applies the allowed-token mask, selects the best valid token, and returns
	a validated tool call.
- `src/grammar.py` contains the JSON grammar state machine. It tracks the selected
	function, required arguments, current value type, string escaping, and valid
	transitions.
- `src/vocab.py` loads the model vocabulary and precomputes token groups used for
	string, number, quote, and empty-string generation.
- `src/models.py` defines the Pydantic schemas and validates that generated
	parameters match the selected function definition.
- `llm_sdk/` contains the SDK supplied with the project. It loads the model,
	encodes text, decodes token IDs, and exposes next-token logits.

## Algorithm

The generation pipeline is:

```text
Input files
		-> Pydantic validation
		-> System prompt containing the available tools
		-> Tokenization with the SDK
		-> Model logits for the next token
		-> Grammar-based allowed-token calculation
		-> Logit masking
		-> Greedy selection of the best valid token
		-> JSON parsing and schema validation
		-> Output file
```

### 1. Schema and prompt preparation

The function definition file is loaded into `FunctionDefinition` models. Each
function contains a name, description, parameter definitions, and an optional
return type. The available functions and the user's prompt are included in the
model context so that the LLM can make the semantic decision.

### 2. Grammar states

`JSONGrammar` controls the expected structure through states such as:

```text
START -> FUNCTION_NAME -> AFTER_NAME -> ARG_KEY -> COLON -> ARG_VALUE
			 -> NEXT_OR_CLOSE_ARGS -> CLOSE_JSON -> DONE
```

At each state, the grammar returns only valid literal continuations. For example,
after an argument name it allows `: `, while a number can only receive characters
that keep it a valid JSON number. Required argument names come from the selected
function definition, so missing and unexpected parameters cannot be accepted.

### 3. Constrained decoding

The model returns a logit for every vocabulary token. The engine creates a mask
filled with negative infinity and copies logits only for tokens accepted by the
current grammar state. The next token is selected with `argmax` from this masked
array.

This process is repeated until the grammar reaches `DONE`. Invalid JSON punctuation,
unknown function names, invalid parameter keys, malformed numbers, and incorrect
structural transitions are therefore excluded during generation rather than
detected only after generation has finished.

String values receive additional handling for JSON escapes and internal quotes.
The vocabulary manager accepts printable ASCII characters commonly needed in
paths, SQL, templates, and regular expressions. A small lookahead is used when a
quote may be either an escaped character inside a value or the closing quote of
that value.

### 4. Validation

After generation, the result is parsed as a `ToolCall`. The selected function must
exist, every required parameter must be present, no extra parameters are allowed,
and every value must match its declared type. Integer values are normalized to
floating-point values when the schema declares the generic `number` type.

## Input Format

### Function definitions

`data/input/functions_definition.json` is a JSON array. Each item describes one
available function:

```json
[
	{
		"name": "fn_add_numbers",
		"description": "Add two numbers together and return their sum.",
		"parameters": {
			"a": {"type": "number"},
			"b": {"type": "number"}
		},
		"returns": {"type": "number"}
	}
]
```

Supported parameter types are `string`, `number`, `integer`, and `boolean`.

### Test prompts

`data/input/function_calling_tests.json` is a JSON array of objects containing a
`prompt` string:

```json
[
	{"prompt": "What is the sum of 2 and 3?"},
	{"prompt": "Reverse the string 'hello'"}
]
```

## Output Format

The default output is `data/output/function_calling_results.json`. It is a JSON
array with exactly three keys per entry:

```json
[
	{
		"prompt": "What is the sum of 2 and 3?",
		"name": "fn_add_numbers",
		"parameters": {
			"a": 2.0,
			"b": 3.0
		}
	}
]
```

The output contains no assistant prose, extra keys, comments, or trailing commas.
The output directory is created automatically and is excluded from Git because it
is generated data.

## Instructions

### Requirements

- Python 3.10 or later.
- `uv` for dependency and environment management.
- Network access on the first run so the model and tokenizer can be downloaded by
	the supplied SDK, unless they are already cached locally.

### Installation

From the repository root:

```bash
uv sync
```

### Run with default paths

```bash
uv run python -m src
```

This reads the two files in `data/input/` and writes:

```text
data/output/function_calling_results.json
```

### Run with custom paths

```bash
uv run python -m src \
	--functions_definition data/input/functions_definition.json \
	--input data/input/function_calling_tests.json \
	--output data/output/function_calls.json
```

### Makefile commands

```bash
make install      # Run uv sync
make run          # Run the application
make debug        # Run through Python's debugger
make lint         # Run flake8 and configured mypy checks
make lint-strict  # Run flake8 and mypy --strict
make clean        # Remove Python caches
```

## Testing Strategy

The implementation was validated at several levels:

1. The default public prompts check function selection, simple argument
	 extraction, arithmetic values, strings, and regular expressions.
2. Edge cases check negative and decimal numbers, zero, empty strings, spaces in
	 strings, and replacements represented by one character.
3. Private moulinette prompts check functions not present in the public examples,
	 including multiplication, booleans, compound interest, SQL queries, file
	 paths, and templates with escaped quotes.
4. Output files are parsed again as JSON and checked for the exact required keys.
5. `flake8` and strict `mypy` checks are used to detect style and typing issues.
6. Invalid JSON, missing files, unknown semantic requests, and malformed model
	 generations are exercised to verify clear error handling.

The project has no dependency on heuristic function-name selection. Accuracy is
measured by comparing the generated function name and parameters with the expected
correction for each test prompt.

## Performance and Reliability

The model is loaded once per program execution and reused for all prompts. The
generation loop is deterministic and bounded by a maximum token count, preventing
unbounded output. Constrained decoding guarantees that successful generated calls
are structurally valid JSON, while Pydantic performs the final schema check.

Runtime depends mainly on the hardware used by the SDK and whether the model is
already cached. On the development machine, the public and private suites complete
within the project's expected evaluation time.

## Design Decisions

- A finite-state grammar was chosen because function calls have a small, explicit
	JSON structure and require strict control over parameter names and types.
- NumPy is used for the vocabulary mask, keeping the constrained-decoding logic
	independent from model internals.
- Pydantic is used at input and output boundaries so malformed data fails with
	explicit validation errors.
- Greedy selection was chosen for reproducibility and stable evaluation results.
- The SDK's public methods are used for encoding, decoding, logits, and vocabulary
	access. Private SDK attributes and methods are not accessed by the application.

## Challenges

The main challenges were aligning grammar transitions with byte-pair tokenizer
boundaries and preserving string content exactly. A token may contain a leading
space, multiple characters, JSON punctuation, or an escaped quote. The solution
tracks the generated buffer, sanitizes only invalid string escapes, uses vocabulary
categories for different value types, and applies lookahead when an internal quote
could be confused with the closing quote.

Another challenge was distinguishing the schema type `number` from `integer`.
The implementation accepts integer-looking numeric text for a `number` parameter
and normalizes the validated value to `float`, while preserving strict integer
validation for `integer` parameters.

## Resources

- [Python documentation](https://docs.python.org/3/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [JSON standard](https://www.json.org/json-en.html)
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/)
- [Qwen model documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained decoding overview](https://huggingface.co/docs/transformers/main/en/internal/generation_utils)

AI tools were used as development assistants for researching constrained decoding,
reviewing tokenizer and JSON edge cases, suggesting test cases, and checking the
clarity of documentation. AI-generated suggestions were manually reviewed,
implemented, and validated with the public tests, private moulinette tests, JSON
validation, flake8, and mypy. The final implementation and its behavior remain the
responsibility of the project author.
