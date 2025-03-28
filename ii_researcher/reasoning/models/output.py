from typing import List, Optional

from pydantic import BaseModel

from ii_researcher.reasoning.config import ConfigConstants
from ii_researcher.reasoning.models.action import Action
from ii_researcher.reasoning.utils import parse_code_blobs


class ModelOutput(BaseModel):
    """A model representing the output from the language model."""

    action: Optional[Action] = None
    raw: str
    is_last: bool = False

    @classmethod
    def from_string(cls, string: str, tool_names: List[str] = None) -> "ModelOutput":
        """
        Parse a string into a ModelOutput instance.

        Args:
            string: A string containing the raw model output

        Returns:
            ModelOutput instance
        """
        raw = string
        action_string = parse_code_blobs(raw, tool_names)

        if action_string:
            try:
                action = Action.from_string(action_string)
                return cls(action=action, raw=raw)
            except ValueError:
                # If parsing as an action fails, treat it as regular output
                return cls(action=None, raw=raw)
        else:
            return cls(action=None, raw=raw)

    def short_format(self) -> str:
        """Format the output in a short format, showing only the action."""
        if self.action:
            return f"Truncated reasoning ....\nAction:\n```py\n{self.action.to_string()}\n```{ConfigConstants.END_CODE}"
        return self.raw

    def full_format(self) -> str:
        """Format the output in its full format."""
        return clean_streamed_output(self.raw.strip()) + ConfigConstants.END_CODE

    def to_string(self) -> str:
        """Convert the output to a string."""
        if ConfigConstants.THINK_TAG_CLOSE in self.raw:
            self.raw = self.raw.replace(ConfigConstants.THINK_TAG_CLOSE, "")
        return self.full_format()


def clean_streamed_output(text: str) -> str:
    """
    Clean the streamed output by removing the end_code marker.

    Args:
        text (str): The streamed output text.

    Returns:
        str: The cleaned text.
    """
    text = text.strip()
    for i in range(len(ConfigConstants.END_CODE), 0, -1):
        if text.endswith(ConfigConstants.END_CODE[:i]):
            return text[:-i]
    return text
