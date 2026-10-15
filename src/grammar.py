"""Grammar state machine for constrained JSON tool call generation."""

from enum import Enum, auto
from typing import Dict, List, Optional, Set
from src.models import FunctionDefinition


class GrammarState(Enum):
    """Discrete states of the JSON tool call generation lifecycle."""

    START = auto()
    FUNCTION_NAME = auto()
    AFTER_NAME = auto()
    ARG_KEY = auto()
    COLON = auto()
    ARG_VALUE = auto()
    NEXT_OR_CLOSE_ARGS = auto()
    CLOSE_JSON = auto()
    DONE = auto()


class JSONGrammar:
    """Tracks generation progress and determines valid text continuations.

    Attributes
    ----------
    functions: Dict[str, FunctionDefinition]
        Available function definitions indexed by name.
    """

    def __init__(
        self, function_definitions: List[FunctionDefinition]
    ) -> None:
        self.functions: Dict[str, FunctionDefinition] = {
            fn.name: fn for fn in function_definitions
        }
        self.reset()

    def reset(self) -> None:
        """Resets the state machine for a new generation session."""
        self.state: GrammarState = GrammarState.START
        self.buffer: str = ""
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_arg_name: Optional[str] = None
        self.remaining_args: List[str] = []
        self.in_string_value: bool = False

    def get_allowed_literals_or_chars(self) -> Set[str]:
        """Returns the set of strings/characters allowed at current state."""
        if self.state == GrammarState.START:
            return {'{"name": "'}

        if self.state == GrammarState.FUNCTION_NAME:
            if not self.selected_function:
                return set(self.functions.keys())
            return set()

        if self.state == GrammarState.AFTER_NAME:
            return {'", "arguments": {'}

        if self.state == GrammarState.ARG_KEY:
            if self.remaining_args:
                return {f'"{arg}"' for arg in self.remaining_args}
            return set()

        if self.state == GrammarState.COLON:
            return {": "}

        if self.state == GrammarState.ARG_VALUE:
            if not self.selected_function or not self.current_arg_name:
                return set()

            param_def = self.selected_function.parameters[
                self.current_arg_name
            ]

            if param_def.type in ("number", "integer"):
                return set("0123456789.-,}")

            if param_def.type == "string":
                if not self.in_string_value:
                    return {'"'}
                allowed_chars = (
                    "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789 _-./:'\""
                )
                return set(allowed_chars)

        if self.state == GrammarState.NEXT_OR_CLOSE_ARGS:
            if self.remaining_args:
                return {", "}
            return {"}"}

        if self.state == GrammarState.CLOSE_JSON:
            return {"}"}

        return set()

    def feed(self, text: str) -> None:
        """Updates internal state based on newly generated text."""
        self.buffer += text

        if self.state == GrammarState.START:
            if self.buffer.startswith('{"name": "'):
                self.state = GrammarState.FUNCTION_NAME

        elif self.state == GrammarState.FUNCTION_NAME:
            for name, fn in self.functions.items():
                target = f'{{"name": "{name}'
                if self.buffer == target or self.buffer.startswith(
                    target + '"'
                ):
                    self.selected_function = fn
                    self.remaining_args = list(fn.parameters.keys())
                    self.state = GrammarState.AFTER_NAME
                    break

        elif self.state == GrammarState.AFTER_NAME:
            name = self.selected_function.name
            expected_prefix = f'{{"name": "{name}", "arguments": {{'
            if self.buffer.startswith(expected_prefix):
                if self.remaining_args:
                    self.state = GrammarState.ARG_KEY
                else:
                    self.state = GrammarState.NEXT_OR_CLOSE_ARGS

        elif self.state == GrammarState.ARG_KEY:
            for arg in list(self.remaining_args):
                if f'"{arg}"' in self.buffer:
                    self.current_arg_name = arg
                    self.remaining_args.remove(arg)
                    self.state = GrammarState.COLON
                    break

        elif self.state == GrammarState.COLON:
            if self.buffer.endswith(": "):
                self.state = GrammarState.ARG_VALUE
                self.in_string_value = False

        elif self.state == GrammarState.ARG_VALUE:
            if self.selected_function and self.current_arg_name:
                param = self.selected_function.parameters[
                    self.current_arg_name
                ]

                if param.type == "string":
                    if not self.in_string_value and text == '"':
                        self.in_string_value = True
                    elif self.in_string_value and text.endswith('"'):
                        self.in_string_value = False
                        self.current_arg_name = None
                        self.state = GrammarState.NEXT_OR_CLOSE_ARGS
                else:
                    if text.endswith(",") or text.endswith("}"):
                        self.current_arg_name = None
                        if text.endswith(","):
                            self.state = GrammarState.ARG_KEY
                        else:
                            self.state = GrammarState.CLOSE_JSON

        elif self.state == GrammarState.NEXT_OR_CLOSE_ARGS:
            if text.strip() == ",":
                self.state = GrammarState.ARG_KEY
            elif text.strip() == "}":
                self.state = GrammarState.CLOSE_JSON

        elif self.state == GrammarState.CLOSE_JSON:
            if text.strip() == "}":
                self.state = GrammarState.DONE
