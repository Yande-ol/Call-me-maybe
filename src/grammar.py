"""Grammar state machine for constrained JSON tool call generation."""

from enum import Enum, auto
import re
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
        self.state_start_idx: int = 0
        self.selected_function: Optional[FunctionDefinition] = None
        self.current_arg_name: Optional[str] = None
        self.remaining_args: List[str] = []
        self.in_string_value: bool = False

    def _transition_to(self, new_state: GrammarState) -> None:
        """Transitions to a new state and marks buffer position."""
        self.state = new_state
        self.state_start_idx = len(self.buffer)

    def current_param_type(self) -> Optional[str]:
        """Returns the parameter type currently being generated, if any."""
        if self.selected_function and self.current_arg_name:
            return self.selected_function.parameters[
                self.current_arg_name
            ].type
        return None

    def is_valid_number_prefix(self, value: str) -> bool:
        """Returns whether value can be extended into a valid JSON number."""
        pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
        if self.current_param_type() == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)"
        return (
            value in ("", "-")
            or re.fullmatch(pattern, value) is not None
        )

    def is_complete_number(self, value: str) -> bool:
        """Returns whether value is a complete valid JSON number."""
        pattern = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
        if self.current_param_type() == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)"
        return re.fullmatch(pattern, value) is not None

    def get_allowed_literals_or_chars(self) -> Set[str]:
        """Returns the set of strings/characters allowed at current state."""
        partial = self.buffer[self.state_start_idx:]

        if self.state == GrammarState.START:
            return {'{"name": "'}

        if self.state == GrammarState.FUNCTION_NAME:
            allowed: Set[str] = set()
            for name in self.functions:
                if name.startswith(partial) and len(partial) < len(name):
                    allowed.add(name[len(partial):])
            return allowed

        if self.state == GrammarState.AFTER_NAME:
            return {'", "parameters": {'}

        if self.state == GrammarState.ARG_KEY:
            allowed_keys: Set[str] = set()
            for arg in self.remaining_args:
                target = f'"{arg}"'
                if target.startswith(partial) and len(partial) < len(target):
                    allowed_keys.add(target[len(partial):])
            return allowed_keys

        if self.state == GrammarState.COLON:
            return {": "}

        if self.state == GrammarState.ARG_VALUE:
            ptype = self.current_param_type()
            if not ptype:
                return set()

            if ptype == "boolean":
                allowed_bools: Set[str] = set()
                for b_val in ("true", "false"):
                    if b_val.startswith(partial) and len(partial) < len(b_val):
                        allowed_bools.add(b_val[len(partial):])
                return allowed_bools

            if ptype in ("number", "integer"):
                chars = set("0123456789.-eE+")
                chars = {
                    char
                    for char in chars
                    if self.is_valid_number_prefix(partial + char)
                }
                if self.is_complete_number(partial):
                    chars.update({","} if self.remaining_args else {"}"})
                return chars

            if ptype == "string":
                return {'"'}

        if self.state == GrammarState.NEXT_OR_CLOSE_ARGS:
            if self.remaining_args:
                return {", "}
            return {"}"}

        if self.state == GrammarState.CLOSE_JSON:
            return {"}"}

        return set()

    def feed(self, text: str) -> None:
        """Updates internal state based on newly generated text."""
        if (
            self.state == GrammarState.ARG_VALUE
            and self.current_param_type() == "string"
            and self.in_string_value
        ):
            if '"' in text:
                q_idx = text.index('"')
                content = text[:q_idx].replace("\\", "\\\\")
                text = content + '"'
            else:
                text = text.replace("\\", "\\\\")

        self.buffer += text
        partial = self.buffer[self.state_start_idx:]

        if self.state == GrammarState.START:
            if self.buffer.endswith('{"name": "'):
                self._transition_to(GrammarState.FUNCTION_NAME)

        elif self.state == GrammarState.FUNCTION_NAME:
            if partial in self.functions:
                fn = self.functions[partial]
                self.selected_function = fn
                self.remaining_args = list(fn.parameters.keys())
                self._transition_to(GrammarState.AFTER_NAME)

        elif self.state == GrammarState.AFTER_NAME:
            if partial.endswith('", "parameters": {'):
                if self.remaining_args:
                    self._transition_to(GrammarState.ARG_KEY)
                else:
                    self._transition_to(GrammarState.NEXT_OR_CLOSE_ARGS)

        elif self.state == GrammarState.ARG_KEY:
            for arg in list(self.remaining_args):
                if partial == f'"{arg}"':
                    self.current_arg_name = arg
                    self.remaining_args.remove(arg)
                    self._transition_to(GrammarState.COLON)
                    break

        elif self.state == GrammarState.COLON:
            if partial.endswith(": "):
                self.in_string_value = False
                self._transition_to(GrammarState.ARG_VALUE)

        elif self.state == GrammarState.ARG_VALUE:
            ptype = self.current_param_type()
            if not ptype:
                return

            if ptype == "string":
                if not self.in_string_value and text == '"':
                    self.in_string_value = True
                elif self.in_string_value and text.endswith('"'):
                    self.in_string_value = False
                    self.current_arg_name = None
                    self._transition_to(GrammarState.NEXT_OR_CLOSE_ARGS)
            elif ptype == "boolean":
                if partial in ("true", "false"):
                    self.current_arg_name = None
                    self._transition_to(GrammarState.NEXT_OR_CLOSE_ARGS)
            else:
                if text.endswith(",") and self.is_complete_number(
                    partial[:-1]
                ):
                    self.current_arg_name = None
                    if not self.buffer.endswith(", "):
                        self.buffer = self.buffer[:-1] + ", "
                    self._transition_to(GrammarState.ARG_KEY)
                elif text.endswith("}") and self.is_complete_number(
                    partial[:-1]
                ):
                    self.current_arg_name = None
                    self._transition_to(GrammarState.CLOSE_JSON)

        elif self.state == GrammarState.NEXT_OR_CLOSE_ARGS:
            if partial.strip() == ",":
                self._transition_to(GrammarState.ARG_KEY)
            elif partial.strip() == "}":
                self._transition_to(GrammarState.CLOSE_JSON)

        elif self.state == GrammarState.CLOSE_JSON:
            if partial.strip() == "}":
                self._transition_to(GrammarState.DONE)
