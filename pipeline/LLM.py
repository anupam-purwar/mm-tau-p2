from openai import OpenAI
from pipeline.utils import LLMResponseWrapper
from config import *

class LLM:

    def __init__(
        self,
        model = AGENT_MODEL,
        temp = 0.0,
    ):
        
        self.model = model
        self.temp = temp
        self.llm = OpenAI()

    def reason(self, messages, stream = False, **kwargs):
        
        messages = ([{"role": "system", "content": "You are a helpful assistant"}]
                    + messages)
        completion = self.llm.chat.completions.create(model=self.model, messages=messages, temperature=self.temp, stream = stream, **kwargs)
        if stream:
            return LLMResponseWrapper(completion)
        return completion.choices[0].message.content

def call_llm(messages, model=AGENT_MODEL, temp=0.0, stream=False, **kwargs):

    llm = OpenAI()
    completion = llm.chat.completions.create(model=model, messages=messages, temperature=temp, stream = stream, **kwargs)
    if stream:
        return LLMResponseWrapper(completion)
    return completion.choices[0].message.content