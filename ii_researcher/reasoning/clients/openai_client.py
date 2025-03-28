import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from ii_researcher.reasoning.config import get_config
from ii_researcher.reasoning.models.trace import Trace
from ii_researcher.reasoning.tools.registry import format_tool_descriptions


class OpenAIClient:
    """OpenAI API client."""

    def __init__(
        self, stream_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """Initialize the OpenAI client."""
        self.config = get_config()
        self.stream_event = stream_event

        # Create synchronous client
        self.client = OpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
        )

        # Create async client
        self.async_client = AsyncOpenAI(
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
        )

    def _get_messages(
        self, trace: Trace, instructions: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get the messages for the OpenAI API."""

        available_tools = format_tool_descriptions()
        system_prompt = self.config.system_prompt.format(
            available_tools=available_tools,
            current_date=datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": trace.query},
            {
                "role": "assistant",
                "content": trace.to_string(instructions),
                "prefix": True,
            },
        ]
        return messages

    def generate_completion(
        self, trace: Trace, instructions: Optional[str] = None
    ) -> Any:
        """Generate a completion using the OpenAI API."""
        messages = self._get_messages(trace, instructions)

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm.model,
                messages=messages,
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                presence_penalty=self.config.llm.presence_penalty,
                stop=self.config.llm.stop_sequence,
            )
            return response
        except Exception as e:
            logging.error("Error generating completion: %s", str(e))
            raise

    async def generate_completion_stream(
        self, trace: Trace, instructions: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion using the OpenAI API."""
        messages = self._get_messages(trace, instructions)

        try:
            stream = await self.async_client.chat.completions.create(
                model=self.config.llm.model,
                messages=messages,
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                presence_penalty=self.config.llm.presence_penalty,
                stop=self.config.llm.get_effective_stop_sequence(len(trace.turns) > 0),
                stream=True,
            )

            # Process the stream
            collected_content = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    collected_content += content
                    yield content

        except Exception as e:
            logging.error("Error generating streaming completion: %s", str(e))
            raise

    def generate_report(self, trace: Trace) -> str:
        """Generate a report using the OpenAI API."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.config.report_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"""Here is the research process and findings:

                ### Research Process
                {trace.to_string()}

                ### Original Question
                {trace.query}

                Based on the research above, please provide a clear and comprehensive report.
                """,
                },
            ]

            response = self.client.chat.completions.create(
                model=self.config.llm.report_model,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error("Error generating report: %s", str(e))
            raise

    async def generate_report_stream(
        self, trace: Trace, callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """Generate a streaming report using the OpenAI API."""

        try:
            messages = [
                {
                    "role": "system",
                    "content": self.config.report_system_prompt,
                },
                {
                    "role": "user",
                    "content": f"""This is the diary of the research process: 
                ### Diary
                {trace.to_string()}
                ### Question
                {trace.query}
                Please do your best.
                """,
                },
            ]

            stream = await self.async_client.chat.completions.create(
                model=self.config.llm.report_model,
                messages=messages,
                temperature=0.6,
                top_p=0.95,
                stream=True,
            )

            # Process the stream
            full_content = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content

                    if callback:
                        callback(content)

                    if self.stream_event:
                        await self.stream_event(
                            "writing_report", {"final_report": full_content}
                        )
                        await asyncio.sleep(0)

            return full_content

        except Exception as e:
            logging.error("Error generating streaming report: %s", str(e))
            raise
