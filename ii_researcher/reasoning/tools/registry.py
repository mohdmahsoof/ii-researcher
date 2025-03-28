import logging
from typing import Dict, List, Optional, Type

from ii_researcher.reasoning.tools.base import BaseTool


class ToolRegistry:
    """Registry for tools."""

    _instance = None
    _tools: Dict[str, Type[BaseTool]] = {}

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, tool_cls: Type[BaseTool]) -> None:
        """Register a tool."""
        if not hasattr(tool_cls, "name") or not tool_cls.name:
            raise ValueError(f"Tool {tool_cls.__name__} must have a name")

        self._tools[tool_cls.name] = tool_cls
        logging.info("Registered tool: %s", tool_cls.name)

    def get_tool(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def get_all_tools(self) -> Dict[str, Type[BaseTool]]:
        """Get all registered tools."""
        return self._tools

    def format_tool_descriptions(self) -> str:
        """Format tool descriptions for the LLM."""
        descriptions = []
        # Sort tools to ensure web_search comes before page_visit
        sorted_tools = sorted(
            self._tools.items(),
            key=lambda x: (x[0] != "web_search", x[0] != "page_visit", x[0]),
        )
        for _, tool_cls in sorted_tools:
            tool_instance = tool_cls()
            descriptions.append(tool_instance.format_description())

        return "*You only have access to these tools:\n" + "\n".join(descriptions)


# Create a singleton instance
registry = ToolRegistry()


def register_tool(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
    """Decorator to register a tool."""
    registry.register(tool_cls)
    return tool_cls


def get_tool(name: str) -> Optional[Type[BaseTool]]:
    """Get a tool by name."""
    return registry.get_tool(name)


def list_tools() -> List[str]:
    """List all registered tools."""
    return registry.list_tools()


def get_all_tools() -> Dict[str, Type[BaseTool]]:
    """Get all registered tools."""
    return registry.get_all_tools()


def format_tool_descriptions() -> str:
    """Format tool descriptions for the LLM."""
    return registry.format_tool_descriptions()
