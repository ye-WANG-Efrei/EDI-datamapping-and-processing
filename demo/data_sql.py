from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import time
import pandas as pd
import json
import duckdb
import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import re   
import pandas as pd
import json
from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

MODEL = "deepseek-r1:14b"
DATA_FILE_PATH_GLOBAL = "data/employee_data.parquet"

class DataSQL:
    def __init__(self, model_name='deepseek-r1:14b', data_file_path=DATA_FILE_PATH_GLOBAL, base_url="http://localhost:11434"):
        # 初始化 Ollama 模型
        self.llm = Ollama(
            model=MODEL, 
            base_url=base_url,
            
        )
        
        # 数据表名
        self.table_name = "temp_data"
        self.data_file_path = os.path.abspath(data_file_path)
        
        # 检查文件是否存在
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file_path}")
            
        # 初始化数据
        self._initialize_data()
    
    def _initialize_data(self):
        """初始化数据表"""
        try:
            print(f"正在读取数据文件: {self.data_file_path}")
            # 读取 parquet 文件
            self.df = pd.read_parquet(self.data_file_path)
            
            # 删除已存在的表（如果存在）
            duckdb.sql(f"DROP TABLE IF EXISTS {self.table_name}")
            
            # 创建新表 - 使用正确的语法
            duckdb.sql(f"CREATE TABLE {self.table_name} AS SELECT * FROM read_parquet('{self.data_file_path}')")
            
            print("数据表初始化成功！")
            print(f"表名: {self.table_name}")
            print(f"列名: {self.df.columns.tolist()}")
        except Exception as e:
            print(f"初始化数据表时发生错误: {e}")
            self.df = None
    
    def generate_sql_query(self, prompt: str) -> str:
        """
        生成 SQL 查询
        
        :param prompt: 用户提示
        :return: SQL 查询语句
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # SQL 生成提示模板
                sql_generation_prompt = """
                你是一个SQL专家，请根据以下提示生成一个SQL查询。
                IMPORTANT: 只返回SQL查询，不要返回任何其他内容。不要解释，不要markdown，不要思考过程。

                Prompt: {prompt}

                Available columns: {columns}
                Table name: {table_name}

                样例格式
                ```sql
                SELECT column1, column2 FROM table_name WHERE condition;
                ```

                记住，只返回SQL查询，不要返回任何其他内容。不要解释，不要markdown，不要思考过程。
                """
               
                if self.df is None:
                    print("错误: 数据未初始化")
                    return None
                    
                formatted_prompt = sql_generation_prompt.format(
                    prompt=prompt, 
                    columns=self.df.columns.tolist(), 
                    table_name=self.table_name
                )
                
                response = self.llm.invoke(formatted_prompt)
                #print("response: ", response)   
                
                # 清理响应中的任何非SQL内容
                code_blocks = re.findall(r'```sql\n(.*?)\n```', response, re.DOTALL)
                print(f"找到 {len(code_blocks)} 个代码块")
                        
                if code_blocks:
                    print("\n使用最后一个代码块...")
                    # 使用最后一个代码块
                    sql_query = code_blocks[-1].strip()
                    print("提取的代码:", sql_query)
                
                # 确保返回的是有效的 SQL 查询
                if not sql_query or not sql_query.lower().startswith("select"):
                    print("Warning: Generated response is not a valid SQL query")
                    print(f"Raw response: {response}")
                    return None
                
                print(f"Generated SQL: {sql_query}")  # 调试输出
                return sql_query
                
            except Exception as e:
                retry_count += 1
                print(f"尝试 {retry_count}/{max_retries} 失败: {str(e)}")
                if retry_count < max_retries:
                    print("等待 2 秒后重试...")
                    time.sleep(2)
                else:
                    print("达到最大重试次数，操作失败")
                    return None
    # code for tool 2
    def analyze_sales_data(self, prompt: str, data: str) -> str:
        """
        Implementation of AI-powered sales data analysis
        :param prompt: 用户提示
        :param data: 数据
        :return: 分析结果   
        """
        # Construct prompt based on analysis type and data subset
        DATA_ANALYSIS_PROMPT = """
        你是一个SQL专家，请根据以下提示生成一个SQL查询。
        IMPORTANT: 只返回SQL查询，不要返回任何其他内容。不要解释，不要markdown，不要思考过程。

        Prompt: {prompt}

        分析以下数据: {data}
        你的任务是回答以下问题: {prompt}

        样例格式
         ```sql
        SELECT column1, column2 FROM table_name WHERE condition;
         ```

        记住，只返回SQL查询，不要返回任何其他内容。不要解释，不要markdown，不要思考过程。
        """
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)
        #print("formatted_prompt: ", formatted_prompt) 
        response = self.llm.invoke(formatted_prompt)
        print("response: ", response)   
        
        # 清理响应中的任何非SQL内容
        code_blocks = re.findall(r'```\nsql\n(.*?)\n```', response, re.DOTALL)
        print(f"找到 {len(code_blocks)} 个代码块")
                
        if code_blocks:
            print("\n使用最后一个代码块...")
            # 使用最后一个代码块
            sql_query = code_blocks[-1].strip()
            print("提取的代码:", sql_query)
        
        # 确保返回的是有效的 SQL 查询
        if not sql_query or not sql_query.lower().startswith("select"):
            print("Warning: Generated response is not a valid SQL query")
            print(f"Raw response: {response}")
           
        
        print(f"Generated SQL: {sql_query}")  # 调试输出
        return sql_query
    
    def execute_query(self, prompt: str, data='null') -> str:
        """
        执行查询并返回结果
        
        :param prompt: 用户提示
        :return: 查询结果
        """
        try:
            # 生成 SQL 查询
            # 根据chat()的data参数，如果data被传参数了则为analysis query，否则为general query
            if data == 'null':
                sql_query = self.generate_sql_query(prompt)
            else:
                print("执行analysis query")
                sql_query = self.analyze_sales_data(prompt, data)
            
            if not sql_query:
                return "Error: Failed to generate valid SQL query"
            
            # 验证表是否存在
            table_exists = duckdb.sql(f"SELECT 1 FROM {self.table_name} LIMIT 1").fetchone() is not None
            if not table_exists:
                return f"Error: Table {self.table_name} does not exist or is empty"
            
            # 执行查询
            result = duckdb.sql(sql_query).df()
            
            if result.empty:
                return "No results found"
                
            return result.to_string()
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}\nSQL Query: {sql_query if 'sql_query' in locals() else 'Not generated'}"
            print(error_msg)  # 添加错误日志
            return error_msg
    
    def chat(self, user_input: str, data='null') -> str:
        """
        与 AI 进行对话
        
        :param user_input: 用户输入
        :return: AI 响应
        """
        try:
            # 记录开始时间
            start_time = time.time()
            result = self.execute_query(user_input,data)
            
            
            # 记录结束时间  
            end_time = time.time()
            # 打印响应时间  
            print(f"响应时间：{end_time - start_time:.2f}秒")
            return result
            
        except Exception as e:
            return f"Error in chat: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        :return: 对话历史列表
        """
        return self.conversation.memory.chat_memory.messages

def main():
    # 使用示例
    data_sql = DataSQL()
    
    # 测试查询
    user_input = "查询所有前10条item"
    start_time = time.time()
    
    try:
        response = data_sql.chat(user_input)
        print(f"\nAI：{response}")
        end_time = time.time()
        print(f"响应时间：{end_time - start_time:.2f}秒")
    except Exception as e:
        print(f"\n发生错误：{str(e)}")
        print("请检查模型是否正常运行，或尝试重新启动程序")

if __name__ == "__main__":
    main() 