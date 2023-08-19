# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from typing import Any, Dict, List, Optional

from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential

from camel.agents import ChatAgent, ChatAgentResponse
from camel.agents.chat_agent import ChatRecord, FunctionCallingRecord
from camel.functions import OpenAIFunction
from camel.messages import BaseMessage
from camel.typing import ModelType
from camel.utils import num_tokens_from_messages, openai_api_key_required


class PlayerAgent(ChatAgent):
    r"""A class for the critic agent that assists in selecting an option.

    Args:
        system_message (BaseMessage): The system message for the critic
            agent.
        model (ModelType, optional): The LLM model to use for generating
            responses. (default :obj:`ModelType.GPT_3_5_TURBO`)
        model_config (Any, optional): Configuration options for the LLM model.
            (default: :obj:`None`)
        message_window_size (int, optional): The maximum number of previous
            messages to include in the context window. If `None`, no windowing
            is performed. (default: :obj:`6`)
        output_language (str, optional): The language to use for the output
    """

    def __init__(
        self,
        system_message: BaseMessage,
        model: ModelType = ModelType.GPT_3_5_TURBO,
        model_config: Optional[Any] = None,
        message_window_size: int = 6,
        output_language: Optional[str] = None,
        function_list: Optional[List[OpenAIFunction]] = None,
    ) -> None:
        super().__init__(
            system_message,
            model,
            model_config,
            message_window_size,
            output_language,
            function_list,
        )
        self.options_dict: Dict[str, str] = {}

    def update_messages(
        self, role: str, message: BaseMessage
    ) -> List[ChatRecord]:
        if role not in {"system", "user", "assistant", "function"}:
            raise ValueError(f"Unsupported role {role}")
        self.stored_messages.append(ChatRecord(role, message))
        return self.stored_messages

    def submit_message(self, message: BaseMessage) -> None:
        self.stored_messages.append(ChatRecord("user", message))

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    @openai_api_key_required
    def step(
        self,
        input_message: BaseMessage,
    ) -> ChatAgentResponse:
        r"""Performs a single step in the chat session by generating a response
        to the input message.

        Args:
            input_message (BaseMessage): The input message to the agent.
            Its `role` field that specifies the role at backen may be either
            `user` or `assistant` but it will be set to `user` anyway since
            for the self agent any incoming message is external.

        Returns:
            ChatAgentResponse: A struct containing the output messages,
                a boolean indicating whether the chat session has terminated,
                and information about the chat session.
        """
        output_messages: Optional[List[BaseMessage]]
        info: Dict[str, Any]
        called_funcs: List[FunctionCallingRecord] = []
        messages = self.update_messages("user", input_message)
        if (
            self.message_window_size is not None
            and len(messages) > self.message_window_size
        ):
            messages = [ChatRecord("system", self.system_message)] + messages[
                -self.message_window_size :
            ]
        openai_messages = [record.to_openai_message() for record in messages]
        num_tokens = num_tokens_from_messages(openai_messages, self.model)
        if num_tokens >= self.model_token_limit:
            return self.step_token_exceed(num_tokens, called_funcs)
        else:
            response = self.model_backend.run(openai_messages)
            self.validate_model_response(response)
            if not self.model_backend.stream:
                (
                    output_messages,
                    finish_reasons,
                    usage_dict,
                    response_id,
                ) = self.handle_batch_response(response)
            else:
                (
                    output_messages,
                    finish_reasons,
                    usage_dict,
                    response_id,
                ) = self.handle_stream_response(response, num_tokens)

            if (
                self.is_function_calling_enabled()
                and finish_reasons[0] == "function_call"
            ):
                # Do function calling
                (
                    func_assistant_msg,
                    func_result_msg,
                    func_record,
                ) = self.step_function_call(response)

                # Update the messages
                messages = self.update_messages(
                    "assistant", func_assistant_msg
                )
                messages = self.update_messages("function", func_result_msg)
                called_funcs.append(func_record)
            else:
                # Function calling disabled or chat stopped
                info = self.get_info(
                    response_id,
                    usage_dict,
                    finish_reasons,
                    num_tokens,
                    called_funcs,
                )

        return ChatAgentResponse(output_messages, self.terminated, info)

    def __repr__(self) -> str:
        r"""Returns a string representation of the :obj:`ChatAgent`.

        Returns:
            str: The string representation of the :obj:`ChatAgent`.
        """
        return f"PlayerAgent({self.role_name}, {self.role_type}, {self.model})"
