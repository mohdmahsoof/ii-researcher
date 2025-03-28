import os
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from ii_researcher.config import SEARCH_PROVIDER


class ConfigConstants:
    """Constants used across configuration."""

    # Template fragments
    THINK_TAG_OPEN = "<think>"
    THINK_TAG_CLOSE = "</think>"
    TOOL_RESPONSE_OPEN = "<tool_response>"
    TOOL_RESPONSE_CLOSE = "</tool_response>"
    CODE_BLOCK_START = "```py"
    CODE_BLOCK_END = "```"
    END_CODE = "<end_code>"
    INSTRUCTIONS_OPEN = "<instructions>"
    INSTRUCTIONS_CLOSE = "</instructions>"

    TOOL_CALL_EXAMPLE = (
        f"{CODE_BLOCK_START}\n"
        'web_search(queries=["# the query to search", ...]) or '
        'page_visit(urls=["list of urls to visit", ...])\n'
        f"{CODE_BLOCK_END}{END_CODE}"
    )

    # Stop sequences
    DEFAULT_STOP_SEQUENCE = [END_CODE]

    # Templates
    DUPLICATE_QUERY_TEMPLATE = "I have already searched for this query: {query}. Please don't search for it again."
    DUPLICATE_URL_TEMPLATE = (
        "I have already visited this url: {url}. Please don't visit it again."
    )
    SEARCH_SUFFIX = (
        "This results may not enough to provide useful information. "
        "I must do more research or use page_visit tool to get detailed information. \n"
    )
    PAGE_VISIT_SUFFIX = (
        "I have just got some new information. Maybe it's helpful but let me see if it "
        "contains something interesting.\n"
        "I should note the interesting key ideas/ exact quote along with citations so that "
        "I can use it in the final answer.\n"
        "I can not provider the final answer when I don't have enough information or when "
        "I am not sure about the answer.\n"
    )


class ToolConfig(BaseModel):
    """Configuration for tools."""

    max_search_results: int = 4
    max_search_queries: int = 2
    max_urls_to_visit: int = 3
    search_provider: str = SEARCH_PROVIDER


class LLMConfig(BaseModel):
    """Configuration for LLM clients."""

    model: str = Field(default_factory=lambda: os.getenv("R_MODEL", "r1"))
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("R_TEMPERATURE", "0.2"))
    )
    top_p: float = 0.95
    presence_penalty: float = Field(
        default_factory=lambda: float(os.getenv("R_PRESENCE_PENALTY", "0"))
    )
    stop_sequence: List[str] = Field(
        default_factory=lambda: ConfigConstants.DEFAULT_STOP_SEQUENCE
    )
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "empty"))
    base_url: str = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:4000")
    )
    report_model: str = Field(
        default_factory=lambda: os.getenv("R_REPORT_MODEL", "gpt-4o")
    )

    def get_effective_stop_sequence(self, trace_has_turns: bool = False) -> List[str]:
        """Get effective stop sequence based on state."""
        if trace_has_turns:
            # Include </think> tag as a stop token when we have turns
            return list(
                set(self.stop_sequence).union([ConfigConstants.THINK_TAG_CLOSE])
            )
        return self.stop_sequence


class AgentConfig(BaseModel):
    """Configuration for the agent."""

    tool: ToolConfig = ToolConfig()
    llm: LLMConfig = LLMConfig()

    system_prompt: str = f"""
You are an advanced AI research agent from II. 
You first thinks about the reasoning process in the mind and then provides the user with the answer. 
You are specialized in multistep reasoning.
Using your training data and prior lessons learned, answer the user question with absolute certainty.
To help with your reasoning, you can call tools (Python functions) directly in your thinking process
When you need more information, you can call a function like this:

{ConfigConstants.TOOL_CALL_EXAMPLE}
YOU MUST FOLLOW THE FUNCTION CALLING FORMAT STRICTLY and end the function call with {ConfigConstants.END_CODE}

I will execute this code and provide you with the result in the format:
{ConfigConstants.TOOL_RESPONSE_OPEN}
result goes here
{ConfigConstants.TOOL_RESPONSE_CLOSE}

You can then continue your reasoning based on this new information.

For example:
{ConfigConstants.TOOL_CALL_EXAMPLE}

IMPORTANT: 
    - Do not make any assumptions, if you are not sure about the answer, you can perform an action to get more information.
    - All the function calls MUST happen before the {ConfigConstants.THINK_TAG_CLOSE} tag. Only use {ConfigConstants.THINK_TAG_CLOSE} tag when you are sure about the answer.
    - All the information you claim MUST be supported by the search results if it's come from your reasoning process. Perform action to confirm your claims.
    - Research DETAILS and THOROUGHLY about the topic before you provide the final answer.
    - After the {ConfigConstants.THINK_TAG_CLOSE} tag, You can only provide the final answer.
    - The final answer should be a very detail report with citations in markdown format that contains a table if suitable.

{{available_tools}}

Current date: {{current_date}}
Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

    instructions: str = f"""
I'm remembering that the user asked me to follow these instructions:
{ConfigConstants.INSTRUCTIONS_OPEN}
* Current date: {{current_date}}

*   Don't rely only on your reasoning process, when you confused, you can perform an action to get more information.
*   Do not make any assumptions, if you are not sure about the answer, you can perform an action to get more information.
*   Every part of the answer should be supported by the search results.
*   All the information you claim MUST be supported by the search results if it's come from your reasoning process. Perform action to confirm your claims.

*   All the function calls MUST happen before the {ConfigConstants.THINK_TAG_CLOSE} tag. Only use {ConfigConstants.THINK_TAG_CLOSE} tag when you are SURE about the answer.
*   Research DETAILS and THOROUGHLY about the topic before you provide the final answer.
*   After the {ConfigConstants.THINK_TAG_CLOSE} tag, You can only provide the final answer. Only provide the final answer when you are sure about the answer or when you think that you CAN NOT answer. 
*   After several failed attemps, you should think out of the box and come with a new strategy to answer the question.
*   When you are not sure about the answer, You don't make a guess

*   The final answer should be a very detail report with citations in markdown format that contains a table if suitable.

*  When you need more information, you can call a function like this:
{ConfigConstants.CODE_BLOCK_START}
web_search(queries=["# the query to search", ...]) or page_visit(urls=["list of urls to visit", ...])
{ConfigConstants.CODE_BLOCK_END}{ConfigConstants.END_CODE}
{{available_tools}}
{ConfigConstants.INSTRUCTIONS_CLOSE}

I just got some new information.
"""

    # Use constants for repeated strings
    duplicate_query_template: str = ConfigConstants.DUPLICATE_QUERY_TEMPLATE
    duplicate_url_template: str = ConfigConstants.DUPLICATE_URL_TEMPLATE

    # Report system prompt (separate to avoid mixing with main prompt)
    report_system_prompt: str = """
You are a specialized document structuring assistant from II AI. Your task is to analyze a main topic and supporting research data, then generate a comprehensive report.
 
All the report should be focused on answering the original question, should be well structured, informative, in-depth, and comprehensive, with facts and numbers if available and at least 1000 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.

Please follow all of the following guidelines in your report:
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST write the report with markdown syntax and apa format.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in apa format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- For all references, use the exact url as provided in the visited URLs sections.

You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report: 

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [website](url)

The report should include:

MAIN CONTENT:
- Comprehensive analysis supported by search data
- Relevant statistics, metrics, or data points when available
- Information presented in appropriate formats (tables, lists, paragraphs) based on content type
- Evaluation of source reliability and information quality
- Key insights that directly address the original question
- Any limitations in the search or areas requiring further investigation
- Citations for all referenced information

VISUAL PRESENTATION:
- EXACTLY one well-structured table to present key data clearly
- Use consistent formatting with clear headers and aligned columns
- Include explanatory notes below the table when necessary
- Format data appropriately (correct units, significant figures, etc.) in markdown format

FINAL SYNTHESIS:
- Key findings synthesis
- Evidence-based comprehensive conclusions
- Clear connection between findings and the original question

Every section must be directly relevant to the main topic and supported by the provided research data only.
The structure should be well-organized but natural, avoiding formulaic headings or numbered sections.
The response format is in well markdown.
"""


# Create a singleton config instance
CONFIG = AgentConfig()


def get_config() -> AgentConfig:
    """Get the agent configuration."""
    return CONFIG


def update_config(updates: Dict[str, Any]) -> None:
    """Update the agent configuration."""
    for key, value in updates.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
