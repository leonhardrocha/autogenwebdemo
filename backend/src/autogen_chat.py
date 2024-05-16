import autogen
from autogen.agentchat.contrib.capabilities.transform_messages import TransformMessages, MessageTransform
from autogen.agentchat.contrib.capabilities.transforms import (
    MessageHistoryLimiter,
    MessageTokenLimiter,
)
from user_proxy_webagent import UserProxyWebAgent
import asyncio
from typing import List, Dict

# config_list = [
#     {
#         "model": "gpt-3.5-turbo",
#     }
# ]

config_list_mistral = [
    {"model": "ollama/mistral", "base_url": "http://localhost:4000", "api_key": "NULL"}
]

llm_config_assistant = {
    "model": "mistral",
    "temperature": 0,
    "config_list": config_list_mistral,
    # "functions": [
    # {
    #     "name": "search_db",
    #     "description": "Search database for order status",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "order_number": {
    #                 "type": "integer",
    #                 "description": "Order number",
    #             },
    #             "customer_number": {
    #                 "type": "string",
    #                 "description": "Customer number",
    #             }
    #         },
    #         "required": ["order_number","customer_number"],
    #     },
    # },
    # ],
}
# llm_config_proxy = {
#     "model": "gpt-3.5-turbo-0613",
#     "temperature": 0,
#     "config_list": config_list,
# }


def litellm_consistent_output(x) -> str:
    """https://litellm.vercel.app/#more-details"""
    return x.get("choices", "")[0].get("message", "")


class TransformLiteLLM(MessageTransform):
    '''overwrite apply_transform'''
    def apply_transform(self, messages: List[Dict]) -> List[Dict]:
        '''get litellm consistent outputs from messages'''
        return [litellm_consistent_output(x) for x in messages]


#############################################################################################
# this is where you put your Autogen logic, here I have a simple 2 agents with a function call
class AutogenChat:    
    def __init__(self, chat_id=None, websocket=None):
        self.websocket = websocket
        self.chat_id = chat_id
        self.client_sent_queue = asyncio.Queue()
        self.client_receive_queue = asyncio.Queue()

        max_messages = MessageHistoryLimiter(max_messages=10)
        truncate_messages = MessageTokenLimiter(
            max_tokens_per_message=4000, model=llm_config_assistant["model"]
        )

        litellm_transform = TransformLiteLLM()

        transform_messages = TransformMessages(
            # transforms=[max_messages, litellm_transform, truncate_messages]
            transforms=[]
        )

        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config_assistant,
            system_message="""You are a helpful assistant, help the user find the status of his order. 
            Only use the tools provided to do the search. Only execute the search after you have all the information needed.
            When you ask a question, always add the word "BRKT"" at the end.
            When you responde with the status add the word TERMINATE""",
        )

        transform_messages.add_to_agent(self.assistant)

        self.user_proxy = UserProxyWebAgent(
            name="user_proxy",
            human_input_mode="ALWAYS",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: litellm_consistent_output(x)
            and litellm_consistent_output(x).rstrip().endswith("TERMINATE"),
            code_execution_config=False
        )

        transform_messages.add_to_agent(self.user_proxy)

        # add the queues to communicate
        self.user_proxy.set_queues(self.client_sent_queue, self.client_receive_queue)

    async def start(self, message):
        await self.user_proxy.a_initiate_chat(
            self.assistant, clear_history=True, message=message
        )

    # MOCH Function call
    def search_db(self, order_number=None, customer_number=None):
        return "Order status: delivered TERMINATE"
