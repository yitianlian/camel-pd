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
"""
Gradio-based web app Agents that uses OpenAI API to generate
a chat between collaborative agents.
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import openai
import openai.error
import tenacity

from apps.agents.text_utils import split_markdown_code
from camel.agents import TaskSpecifyAgent
from camel.messages import BaseMessage
from camel.societies import RolePlaying
from camel.typing import TaskType

REPO_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
)

ChatBotHistory = List[Tuple[Optional[str], Optional[str]]]


@dataclass
class State:
    session: Optional[RolePlaying]
    max_messages: int
    chat: ChatBotHistory
    saved_assistant_msg: Optional[BaseMessage]

    @classmethod
    def empty(cls) -> "State":
        return cls(None, 0, [], None)

    @staticmethod
    def construct_inplace(
        state: "State",
        session: Optional[RolePlaying],
        max_messages: int,
        chat: ChatBotHistory,
        saved_assistant_msg: Optional[BaseMessage],
    ) -> None:
        state.session = session
        state.max_messages = max_messages
        state.chat = chat
        state.saved_assistant_msg = saved_assistant_msg


def parse_arguments():
    """ Get command line arguments. """

    parser = argparse.ArgumentParser("Camel data explorer")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--share", type=bool, default=False, help="Expose the web UI to Gradio"
    )
    parser.add_argument(
        "--server-port", type=int, default=8080, help="Port ot run the web page on"
    )
    parser.add_argument(
        "--inbrowser",
        type=bool,
        default=False,
        help="Open the web UI in the default browser on lunch",
    )
    parser.add_argument(
        "--concurrency-count",
        type=int,
        default=1,
        help="Number if concurrent threads at Gradio websocket queue. "
        + "Increase to serve more requests but keep an eye on RAM usage.",
    )
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print("Unknown args: ", unknown)
    return args


def load_roles(path: str) -> List[str]:
    """ Load roles from list files.

    Args:
        path (str): Path to the TXT file.

    Returns:
        List[str]: List of roles.
    """

    assert os.path.exists(path)
    roles = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r"^\d+\.\s*(.+)\n*$", line)
            if match:
                role = match.group(1)
                roles.append(role)
            else:
                print("Warning: no match")
    return roles


def cleanup_on_launch(state) -> Tuple[State, ChatBotHistory, Dict]:
    """ Prepare the UI for a new session.

    Args:
        state (State): Role playing state.

    Returns:
        Tuple[State, ChatBotHistory, Dict]:
            - Updated state.
            - Chatbot window contents.
            - Start button state (disabled).
    """
    # The line below breaks the every=N runner
    # `state = State.empty()`

    State.construct_inplace(state, None, 0, [], None)

    return state, [], gr.update(interactive=False)


def role_playing_start(
    state,
    society_name: str,
    assistant: str,
    user: str,
    original_task: str,
    max_messages: float,
    with_task_specifier: bool,
    word_limit: int,
) -> Union[Dict, Tuple[State, str, Union[str, Dict], ChatBotHistory, Dict]]:
    """ Creates a role playing session.

    Args:
        state (State): Role playing state.
        society_name:
        assistant (str): Contents of the Assistant field.
        user (str): Contents of the User field.
        original_task (str): Original task field.
        with_task_specifier (bool): Enable/Disable task specifier.
        word_limit (int): Limit of words for task specifier.

    Returns:
        Union[Dict, Tuple[State, str, Union[str, Dict], ChatBotHistory, Dict]]:
            - Updated state.
            - Generated specified task.
            - Planned task (if any).
            - Chatbot window contents.
            - Progress bar contents.
    """

    if state.session is not None:
        print("Double click")
        return {}  # may fail

    if society_name not in {"AI Society", "Code"}:
        print(f"Error: unrecognezed society {society_name}")
        return {}

    meta_dict: Optional[Dict[str, str]]
    extend_sys_msg_meta_dicts: Optional[List[Dict]]
    task_type: TaskType
    if society_name == "AI Society":
        meta_dict = None
        extend_sys_msg_meta_dicts = None
        # Keep user and assistant intact
        task_type = TaskType.AI_SOCIETY
    else:  # "Code"
        meta_dict = {"language": assistant, "domain": user}
        extend_sys_msg_meta_dicts = [meta_dict, meta_dict]
        assistant = f"{assistant} Programmer"
        user = f"Person working in {user}"
        task_type = TaskType.CODE

    try:
        task_specify_kwargs = (
            dict(word_limit=word_limit) if with_task_specifier else None
        )

        session = RolePlaying(
            assistant,
            user,
            original_task,
            with_task_specify=with_task_specifier,
            task_specify_agent_kwargs=task_specify_kwargs,
            with_task_planner=False,
            task_type=task_type,
            extend_sys_msg_meta_dicts=extend_sys_msg_meta_dicts,
            extend_task_specify_meta_dict=meta_dict,
        )
    except (openai.error.RateLimitError, tenacity.RetryError, RuntimeError) as ex:
        print("OpenAI API exception 0 " + str(ex))
        return (state, str(ex), "", [], gr.update())

    # Can't re-create a state like below since it
    # breaks 'role_playing_chat_cont' runner with every=N.
    # `state = State(session=session, max_messages=int(max_messages), chat=[],`
    # `             saved_assistant_msg=None)`

    State.construct_inplace(state, session, int(max_messages), [], None)

    specified_task_prompt = (
        session.specified_task_prompt
        if session.specified_task_prompt is not None
        else ""
    )
    planned_task_prompt = (
        session.planned_task_prompt if session.planned_task_prompt is not None else ""
    )

    planned_task_upd = gr.update(
        value=planned_task_prompt, visible=session.planned_task_prompt is not None
    )

    progress_update = gr.update(maximum=state.max_messages, value=1, visible=True)

    return (state, specified_task_prompt, planned_task_upd, state.chat, progress_update)


def role_playing_chat_init(state) -> Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
    """ Initialize role playing.

    Args:
        state (State): Role playing state.

    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Chatbot window contents.
            - Progress bar contents.
    """

    if state.session is None:
        print("Error: session is none on role_playing_chat_init call")
        return state, state.chat, gr.update()

    session: RolePlaying = state.session

    try:
        init_assistant_msg: BaseMessage
        init_assistant_msg, _ = session.init_chat()
    except (openai.error.RateLimitError, tenacity.RetryError, RuntimeError) as ex:
        print("OpenAI API exception 1 " + str(ex))
        state.session = None
        return state, state.chat, gr.update()

    state.saved_assistant_msg = init_assistant_msg

    progress_update = gr.update(maximum=state.max_messages, value=1, visible=True)

    return state, state.chat, progress_update


# WORKAROUND: do not add type hints for session and chatbot_history
def role_playing_chat_cont(state) -> Tuple[State, ChatBotHistory, Dict, Dict]:
    """ Produce a pair of messages by an assistant and a user.
        To be run multiple times.

    Args:
        state (State): Role playing state.

    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Chatbot window contents.
            - Progress bar contents.
            - Start button state (to be eventually enabled).
    """

    if state.session is None:
        return state, state.chat, gr.update(visible=False), gr.update()

    session: RolePlaying = state.session

    if state.saved_assistant_msg is None:
        return state, state.chat, gr.update(), gr.update()

    try:
        assistant_response, user_response = session.step(state.saved_assistant_msg)
    except (openai.error.RateLimitError, tenacity.RetryError, RuntimeError) as ex:
        print("OpenAI API exception 2 " + str(ex))
        state.session = None
        return state, state.chat, gr.update(), gr.update()

    if len(user_response.msgs) != 1 or len(assistant_response.msgs) != 1:
        return state, state.chat, gr.update(), gr.update()

    u_msg = user_response.msg
    a_msg = assistant_response.msg

    state.saved_assistant_msg = a_msg

    state.chat.append((None, split_markdown_code(u_msg.content)))
    state.chat.append((split_markdown_code(a_msg.content), None))

    if len(state.chat) >= state.max_messages:
        state.session = None

    if "CAMEL_TASK_DONE" in a_msg.content or "CAMEL_TASK_DONE" in u_msg.content:
        state.session = None

    progress_update = gr.update(
        maximum=state.max_messages,
        value=len(state.chat),
        visible=state.session is not None,
    )

    start_bn_update = gr.update(interactive=state.session is None)

    return state, state.chat, progress_update, start_bn_update


def stop_session(state) -> Tuple[State, Dict, Dict]:
    """ Finish the session and leave chat contents as an artefact.

    Args:
        state (State): Role playing state.

    Returns:
        Union[Dict, Tuple[State, ChatBotHistory, Dict]]:
            - Updated state.
            - Progress bar contents.
            - Start button state (to be eventually enabled).
    """

    state.session = None
    return state, gr.update(visible=False), gr.update(interactive=True)


def construct_ui(blocks, api_key: Optional[str] = None) -> None:
    """ Build Gradio UI and populate with topics.

    Args:
        api_key (str): OpenAI API key.

    Returns:
        None
    """

    if api_key is not None:
        openai.api_key = api_key

    society_dict: Dict[str, Dict[str, Any]] = {}
    for society_name in ("AI Society", "Code"):
        if society_name == "AI Society":
            assistant_role_subpath = "ai_society/assistant_roles.txt"
            user_role_subpath = "ai_society/user_roles.txt"
            assistant_role = "Python Programmer"
            user_role = "Stock Trader"
            default_task = "Develop a trading bot for the stock market"
        else:
            assistant_role_subpath = "code/languages.txt"
            user_role_subpath = "code/domains.txt"
            assistant_role = "JavaScript"
            user_role = "Sociology"
            default_task = "Develop a poll app"

        assistant_role_path = os.path.join(REPO_ROOT, f"data/{assistant_role_subpath}")
        user_role_path = os.path.join(REPO_ROOT, f"data/{user_role_subpath}")

        society_info = dict(
            assistant_roles=load_roles(assistant_role_path),
            user_roles=load_roles(user_role_path),
            assistant_role=assistant_role,
            user_role=user_role,
            default_task=default_task,
        )
        society_dict[society_name] = society_info

    default_society = society_dict["AI Society"]

    def change_society(society_name: str) -> Tuple[Dict, Dict, str]:
        society = society_dict[society_name]
        assistant_dd_update = gr.update(
            choices=society["assistant_roles"], value=society["assistant_role"]
        )
        user_dd_update = gr.update(
            choices=society["user_roles"], value=society["user_role"]
        )
        return assistant_dd_update, user_dd_update, society["default_task"]

    with gr.Row():
        with gr.Column(scale=1):
            society_dd = gr.Dropdown(
                ["AI Society", "Code"],
                label="Choose the society",
                value="AI Society",
                interactive=True,
            )
        with gr.Column(scale=2):
            assistant_dd = gr.Dropdown(
                default_society["assistant_roles"],
                label="Example assistant roles",
                value=default_society["assistant_role"],
                interactive=True,
            )
            assistant_ta = gr.TextArea(
                label="Assistant role (EDIT ME)", lines=1, interactive=True
            )
        with gr.Column(scale=2):
            user_dd = gr.Dropdown(
                default_society["user_roles"],
                label="Example user roles",
                value=default_society["user_role"],
                interactive=True,
            )
            user_ta = gr.TextArea(
                label="User role (EDIT ME)", lines=1, interactive=True
            )
        with gr.Column(scale=2):
            gr.Markdown(
                '## CAMEL: Communicative Agents for "Mind" Exploration'
                " of Large Scale Language Model Society\n"
                "Github repo: [https://github.com/lightaime/camel]"
                "(https://github.com/lightaime/camel)"
                '<div style="display:flex; justify-content:center;">'
                '<img src="https://raw.githubusercontent.com/lightaime/camel/'
                'master/misc/logo.png" alt="Logo" style="max-width:50%;">'
                "</div>"
            )
    with gr.Row():
        with gr.Column(scale=9):
            original_task_ta = gr.TextArea(
                label="Give me a preliminary idea (EDIT ME)",
                value=default_society["default_task"],
                lines=1,
                interactive=True,
            )
        with gr.Column(scale=1):
            universal_task_bn = gr.Button("Insert universal task")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                task_specifier_cb = gr.Checkbox(value=True, label="With task specifier")
            with gr.Row():
                ts_word_limit_nb = gr.Number(
                    value=TaskSpecifyAgent.DEFAULT_WORD_LIMIT,
                    label="Word limit for task specifier",
                    visible=task_specifier_cb.value,
                )
        with gr.Column():
            num_messages_sl = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=10,
                interactive=True,
                label="Messages to generate",
            )

        with gr.Column(scale=2):
            with gr.Row():
                start_bn = gr.Button(
                    "Make agents chat [takes time]", elem_id="start_button"
                )
            with gr.Row():
                clear_bn = gr.Button("Interrupt the current query")
    progress_sl = gr.Slider(
        minimum=0,
        maximum=100,
        value=0,
        step=1,
        label="Progress",
        interactive=False,
        visible=False,
    )
    specified_task_ta = gr.TextArea(
        label="Specified task prompt given to the role-playing session"
        " based on the original (simplistic) idea",
        lines=1,
        interactive=False,
    )
    task_prompt_ta = gr.TextArea(
        label="Planned task prompt", lines=1, interactive=False, visible=False
    )
    chatbot = gr.Chatbot(label="Chat between autonomous agents")
    session_state = gr.State(State.empty())

    universal_task_bn.click(lambda: "Help me to do my job", None, original_task_ta)

    task_specifier_cb.change(
        lambda v: gr.update(visible=v), task_specifier_cb, ts_word_limit_nb
    )

    start_bn.click(
        cleanup_on_launch,
        session_state,
        [session_state, chatbot, start_bn],
        queue=False,
    ).then(
        role_playing_start,
        [
            session_state,
            society_dd,
            assistant_ta,
            user_ta,
            original_task_ta,
            num_messages_sl,
            task_specifier_cb,
            ts_word_limit_nb,
        ],
        [session_state, specified_task_ta, task_prompt_ta, chatbot, progress_sl],
        queue=False,
    ).then(
        role_playing_chat_init,
        session_state,
        [session_state, chatbot, progress_sl],
        queue=False,
    )

    blocks.load(
        role_playing_chat_cont,
        session_state,
        [session_state, chatbot, progress_sl, start_bn],
        every=0.5,
    )

    clear_bn.click(stop_session, session_state, [session_state, progress_sl, start_bn])

    society_dd.change(
        change_society, society_dd, [assistant_dd, user_dd, original_task_ta]
    )
    assistant_dd.change(lambda dd: dd, assistant_dd, assistant_ta)
    user_dd.change(lambda dd: dd, user_dd, user_ta)

    blocks.load(change_society, society_dd, [assistant_dd, user_dd, original_task_ta])
    blocks.load(lambda dd: dd, assistant_dd, assistant_ta)
    blocks.load(lambda dd: dd, user_dd, user_ta)


def construct_blocks(api_key: Optional[str]):
    """ Construct Agents app but do not launch it.

    Args:
        api_key (Optional[str]): OpenAI API key.

    Returns:
        gr.Blocks: Blocks instance.
    """

    css_str = "#start_button {border: 3px solid #4CAF50; font-size: 20px;}"

    with gr.Blocks(css=css_str) as blocks:
        construct_ui(blocks, api_key)

    return blocks


def main():
    """ Entry point. """

    args = parse_arguments()

    print("Getting Agents web server online...")

    blocks = construct_blocks(args.api_key)

    blocks.queue(args.concurrency_count).launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_name="0.0.0.0",
        server_port=args.server_port,
        debug=True,
    )

    print("Exiting.")


if __name__ == "__main__":
    main()
