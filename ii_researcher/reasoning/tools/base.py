import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class BaseTool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    argument_schema: Dict[str, Dict[str, Any]]
    return_type: str
    suffix: str

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """Execute the tool with the given arguments."""

    async def execute_stream(
        self, stream_event: Callable[[str, Dict[str, Any]], None], **kwargs
    ) -> str:
        """Execute the tool with the given arguments."""
        await stream_event("tool", {"name": self.name, "arguments": kwargs})
        await asyncio.sleep(0)
        result = await self.execute(**kwargs)
        return result

    def format_description(self) -> str:
        """Format the tool description for the LLM."""
        return f"- {self.name}: {self.description}\n    Takes inputs: {self.argument_schema}\n    Returns an output of type: {self.return_type}"
