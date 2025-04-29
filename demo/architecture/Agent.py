import pandasai as pai
import pathlib
import os # Import os module to handle path separators consistently
import re
from pandasai.config import Config
from pandasai.config import ConfigManager
from pandasai.llm.deepseek_local_llm import DeepSeekLocalLLM
from pandasai.llm.base import LLM
from pandasai.dataframe.base import DataFrame as PaiDataFrame
from Readfile import read_file


class Agent():
    def __init__(self):
        # Set the API key
        pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")
        # Initialize the LLM
        self.llm = pai
        self.df_list = None
        self.request = None

    def readFile(self, file_path, file_name):
        self.df_list = read_file(pai_=self.llm, file_path=file_path, file_name=file_name)
        return self.df_list

    def QueryPlaningAgent(self):
        pass

    

# Example usage
if __name__ == "__main__":
    agent = Agent()
    file_path = "D:\zWork\dream2rich\ollama_agent\EDI-datamapping-and-processing\data\TBC Lille.xlsx"
    file_name = "tbc"
    
    # Read the file
    agent.readFile(file_path, file_name)
    
    # Make a request
    request = '''show me all the columns in the dataset'''
    result = agent.llm.chat(request,*agent.df_list)
    
    print(result)
