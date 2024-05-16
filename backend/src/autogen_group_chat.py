"""Autogen Group Chat Demo"""

import os
import asyncio
from typing import List, Dict
import autogen
from autogen.agentchat.contrib.capabilities.transform_messages import MessageTransform

# from autogen.agentchat.contrib.capabilities.transforms import MessageHistoryLimiter, MessageTokenLimiter
from autogen.coding import LocalCommandLineCodeExecutor
from user_proxy_webagent import UserProxyWebAgent
from groupchatweb import GroupChatManagerWeb

config_list = [
    {
        "model": "gpt-3.5-turbo",
    }
]

config_list_mistral = [
    {"model": "ollama/mistral", "base_url": "http://localhost:4000", "api_key": "NULL"}
]

config_list_codellama = [
    {
        "model": "ollama/codellama",
        "base_url": "http://localhost:4000",
        "api_key": "NULL",
    }
]

llm_config_mistral = {
    "model": "ollama/mistral",
    "temperature": 0,
    "config_list": config_list_mistral,
}

llm_config_codellama = {
    "model": "ollama/codellama",
    "temperature": 0,
    "config_list": config_list_codellama,
}


def litellm_consistent_output(x) -> str:
    """https://litellm.vercel.app/#more-details"""
    return x.get("choices", "")[0].get("message", "")


class TransformLiteLLM(MessageTransform):
    """overwrite apply_transform"""

    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        """get litellm consistent outputs from messages"""
        return [litellm_consistent_output(x) for x in messages]


#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat:
    """Autogen Group Chat Class: this is where you put your Autogen logic and agents with function call"""
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        self.creator = autogen.AssistantAgent(
            name="creator",
            llm_config=llm_config_mistral,
            max_consecutive_auto_reply=5,
            system_message="""You are a helpful assistant,
            you have creative ideas how to solve tasks from user
            thinking the ideas step by step in the order of execution,
            but do not generate any code, just forward each step to the critic
            with the word CONTINUE at the end.""",
        )

        self.critic = autogen.AssistantAgent(
            name="critic",
            llm_config=llm_config_mistral,
            max_consecutive_auto_reply=5,
            system_message="""You are a helpful assistant,
            you should validade the ideas from the creator and
            forward the breakdown the ideas in form of tasks to the coder
            with the word CONTINUE at the end.""",
        )

        self.coder = autogen.AssistantAgent(
            name="coder",
            llm_config=llm_config_codellama,
            code_execution_config={
                # the executor to run the generated code
                "executor": LocalCommandLineCodeExecutor(
                    work_dir=os.environ.get("EXECUTION_FOLDER")
                ),
            },
            system_message="""You are a helpful coder,
            you should validade the tasks from the critic and once done,
            forward to QA the tasks with the addition of the generated code and execution output
            with the word TERMINATE at the end.
            Otherwise, reply to the critic with the reason why each task is not solved yet
            with word CONTINUE at the end.""",
        )

        self.qa = autogen.AssistantAgent(
            name="QA",
            llm_config=llm_config_codellama,
            max_consecutive_auto_reply=5,
            system_message="""You are a helpful quality analist,
            you should validade all the the code and execution output from the coder and once done,
            reply the result to the user only if the execution is doing all the tasks
            with the word TERMINATE at the end.
            Otherwise, reply to the coder with the reason why each task is not solved yet
            with word CONTINUE at the end.""",
        )

        self.user_proxy = UserProxyWebAgent(
            name="user_proxy",
            human_input_mode="ALWAYS",
            system_message="""You ask for ideas for a specific topic""",
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: x.get("content", "")
            and x.get("content", "").rstrip().endswith("TERMINATE"),
        )

        # add the queues to communicate
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)

        self.groupchat = autogen.GroupChat(
            agents=[self.user_proxy, self.creator, self.critic, self.coder, self.qa],
            messages=[],
            max_round=20,
            speaker_selection_method="round_robin",
        )
        self.manager = GroupChatManagerWeb(
            groupchat=self.groupchat,
            llm_config=llm_config_mistral,
            human_input_mode="ALWAYS",
        )

    async def start(self, message):
        await self.user_proxy.a_initiate_chat(
            self.manager, clear_history=True, message=message
        )
