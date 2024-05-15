import json
import os
import openai
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory

class AgentCollection:
    def __init__(self, vector_store, prompt_path):
        self.vector_store = vector_store
        self.memory = ConversationBufferMemory()
        self.agent = self._initialize_agent()
        self.prompt_template = self._load_prompt_template(prompt_path)

    def _initialize_agent(self):
        try:
            llm = OpenAI(temperature=0.7)
            agent = initialize_agent(
                tools=[],
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True
            )
            return agent
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return None

    def _load_prompt_template(self, prompt_path):
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                prompt_template = json.load(file)
            return prompt_template
        except Exception as e:
            print(f"Error loading prompt template: {e}")
            return None

    def generate_novel_segment(self, query):
        prompt = self._create_prompt(query)
        try:
            response = self.agent.run({"input": prompt})
            return response
        except Exception as e:
            print(f"An error occurred during novel segment generation: {e}")
            return None

    def _create_prompt(self, query):
        # 将 query 插入到 prompt_template 中的适当位置
        prompt_template = self.prompt_template.copy()
        if "query_output_goes_here" in prompt_template:
            prompt_template["query_output_goes_here"] = query
        return json.dumps(prompt_template, ensure_ascii=False)

