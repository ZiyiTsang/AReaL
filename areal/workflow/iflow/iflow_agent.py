import os

from math_verify import parse, verify

from areal.api import AsyncRewardWrapper
from areal.utils import logging

logger = logging.getLogger("IFlowAgent")

try:
    from iflow_sdk import (
        IFlowClient,
        IFlowOptions,
        PlanMessage,
        TaskFinishMessage,
    )
    from iflow_sdk.models import AssistantMessage, ToolCallMessage

    IFLOW_SDK_AVAILABLE = True
except ImportError:
    IFLOW_SDK_AVAILABLE = False


def math_reward_fn(completions: str, answer: str) -> float:
    ans = parse(completions)
    gold = parse(answer)
    return float(verify(ans, gold))


class IFlowAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.reward_fn = kwargs.get("reward_fn", math_reward_fn)
        self.timeout = kwargs.get("timeout", 600.0)
        if not IFLOW_SDK_AVAILABLE:
            raise ImportError(
                "iflow_sdk is not installed. Please install it with: pip install iflow-sdk"
            )

    async def run(self, data: dict, **extra_kwargs) -> float:
        os.environ["IFLOW_baseUrl"] = extra_kwargs.get("base_url", None)
        options = IFlowOptions(
            url="ws://localhost:8090/acp",
            timeout=self.timeout,
        )
        content = data["messages"][-1]["content"]
        final_output = ""

        async with IFlowClient(options) as client:
            await client.send_message(content)
            async for message in client.receive_messages():
                if isinstance(message, AssistantMessage):
                    if message.chunk and message.chunk.text:
                        final_output += message.chunk.text
                        logger.debug(f"Assistant message: {message.chunk.text[:50]}...")
                elif isinstance(message, ToolCallMessage):
                    tool_info = f"\n[ToolCall] {message.tool_name}: {message.status}\n"
                    final_output += tool_info
                elif isinstance(message, PlanMessage):
                    plan_info = "\n[Plan]\n"
                    if hasattr(message, "entries") and message.entries:
                        for entry in message.entries:
                            plan_info += f"- [{entry.priority}] {entry.content}\n"
                    final_output += plan_info
                    await client.send_message("Confirm")
                elif isinstance(message, TaskFinishMessage):
                    break
                else:
                    raise ValueError(f"Unexpected message type: {type(message)}")

        reward_fn = AsyncRewardWrapper(self.reward_fn)
        reward = await reward_fn(
            completions=final_output, answer=data.get("answer", "")
        )
        return reward
