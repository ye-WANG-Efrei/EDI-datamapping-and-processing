from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from typing import Dict
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import time
import pandas as pd
import json
import duckdb
import os
from pydantic import BaseModel, Field
from IPython.display import Markdown
from typing import Optional, List, Dict, Any
import re   

MODEL = "deepseek-r1:14b"
DATA_FILE_PATH_GLOBAL = "data/employee_data.parquet"

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.






class DataSQL:
    
    def __init__(self, model_name='deepseek-r1:14b', data_file_path=DATA_FILE_PATH_GLOBAL, base_url="http://localhost:11434"):
        # 初始化 Ollama 模型
        self.llm = ChatOllama(model=model_name, 
                              base_url=base_url
                              )

        # 用于存储对话历史的字典    
        self.session_store: Dict[str, ChatMessageHistory] = {}

        # 创建带历史记录的运行器
        self.conversation = RunnableWithMessageHistory(
            self.llm,
            self.get_by_session_id,
            input_messages_key="input",
            history_messages_key="history"
        )
  
        
        # SQL 生成提示模板
        self.sql_generation_prompt = """
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
        self.DATA_ANALYSIS_PROMPT = """
        分析以下数据: {data}
        你的任务是回答以下问题: {prompt}
        
        记住，只返回SQL查询，不要返回任何其他内容。不要解释，不要markdown，不要思考过程。
        """
        
        # 数据表名
        self.table_name = "temp_data"
        self.data_file_path = os.path.abspath(data_file_path)
        
        # 检查文件是否存在
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file_path}")
            
        # 初始化数据
        self._initialize_data()


    def get_by_session_id(self,session_id: str) -> ChatMessageHistory:
            """
            根据会话 ID 获取或创建对应的对话历史记录。

            :param session_id: 会话的唯一标识符
            :return: 与该会话关联的 ChatMessageHistory 实例
            """
            if session_id not in self.session_store:
                self.session_store[session_id] = ChatMessageHistory()
            return self.session_store[session_id]
    

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
        
        if self.df is None:
            print("错误: 数据未初始化")
            return None
            
        formatted_prompt = self.sql_generation_prompt.format(
            prompt=prompt, 
            columns=self.df.columns.tolist(), 
            table_name=self.table_name
        )
        
        try:
            print("发送消息到模型...")
            #            # 使用字典格式，以便与 RunnableWithMessageHistory 兼容
            # response = self.conversation.invoke(
            #     {"input": formatted_prompt},
            #     config={"configurable": {"session_id": "default"}}
            # )
            # 直接使用字符串而不是字典
            response = self.llm.invoke(formatted_prompt)
            print("收到模型响应")
            # print("response类型:", type(response))
            # print("response: ", response)   
            
            # 提取 SQL 查詢            # 清理响应中的任何非SQL内容
            # sql_query = response.content.strip()
            # sql_query = sql_query.replace("```sql", "").replace("```", "")
            # sql_query = sql_query.replace("<think>", "").replace("</think>", "")
            # if "\n" in sql_query:
            #     sql_query = sql_query.split("\n")[0]  # 只取第一行
            content = response.content
            # 移除所有標記和思考過程
            content = content.replace("<think>", "").replace("</think>", "")
            # 提取 SQL 代碼塊
            sql_blocks = re.findall(r'```sql(.*?)```', content, re.DOTALL)
            
            if sql_blocks:
                sql_query = sql_blocks[-1].strip()  # 使用最後一個 SQL 代碼塊
            else:
                # 如果沒有找到代碼塊，嘗試直接使用內容
                sql_query = content.strip()
            
            # 確保是有效的 SQL 查詢
            if not sql_query or not sql_query.lower().startswith("select"):
                print("Warning: Generated response is not a valid SQL query")
                print(f"Raw response: {response}")
                return None
            
            print(f"Generated SQL: {sql_query}")
            return sql_query
        except Exception as e:
            print(f"生成SQL查询时出错: {str(e)}")
            print(f"错误类型: {type(e)}")
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
        
        formatted_prompt = self.DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)

        try:
            print("发送分析请求到模型...")
            # 使用字典格式，以便与 RunnableWithMessageHistory 兼容
            response = self.conversation.invoke(
                {"input": formatted_prompt},
                config={"configurable": {"session_id": "default"}}
            )
            print("收到模型响应")
            print("response: ", response)
            
            # 清理响应中的任何非SQL内容
            if hasattr(response, 'content'):
                content = response.content
            else:
                # 处理不同类型的响应
                print("响应没有content属性，尝试其他方式获取内容")
                if isinstance(response, str):
                    content = response
                elif isinstance(response, dict) and 'output' in response:
                    content = response['output']
                else:
                    print(f"无法从响应中提取内容，响应类型: {type(response)}")
                    return None
            
            code_blocks = re.findall(r'```sql(.*?)```', content, re.DOTALL)
            print(f"找到 {len(code_blocks)} 个代码块")
                    
            if code_blocks:
                print("\n使用最后一个代码块...")
                # 使用最后一个代码块
                sql_query = code_blocks[-1].strip()
                print("提取的代码:", sql_query)
            else:
                # 如果没有找到SQL代码块，尝试直接使用内容
                sql_query = content.strip()
                sql_query = sql_query.replace("```sql", "").replace("```", "")
                sql_query = sql_query.replace("<think>", "").replace("</think>", "")
            
            # 确保返回的是有效的 SQL 查询
            if not sql_query or not sql_query.lower().startswith("select"):
                print("Warning: Generated response is not a valid SQL query")
                print(f"Raw response: {response}")
                return None
               
            print(f"Generated SQL: {sql_query}")  # 调试输出
            return sql_query
        except Exception as e:
            print(f"分析数据时出错: {str(e)}")
            print(f"错误类型: {type(e)}")
            return None
            
    def execute_query(self, prompt: str, data) -> str:
        """
        执行查询并返回结果
        
        :param prompt: 用户提示
        :return: 查询结果
        """
        
        print("执行查询")
        #根据chat()的data参数，如果data被传参数了则为analysis query，否则为general query    
        if data == 'null':
            print("执行general query")
            sql_query = self.generate_sql_query(prompt)
        else:
            print("执行analysis query")
            sql_query = self.analyze_sales_data(prompt, data)
        
        # 执行查询
        result = duckdb.sql(sql_query).df()
        
        return result.to_string()
        # except Exception as e:
        #     return f"Error executing query: {str(e)}\nSQL Query: {sql_query if 'sql_query' in locals() else 'Not generated'}"
    
    def chat(self, user_input: str, data='null') -> str:
        """
        与 AI 进行对话
        
        :param user_input: 用户输入
        :return: AI 响应
        """
        
            # 记录开始时间
        start_time = time.time()
        # 根据chat()的data参数，如果data被传参数了则为analysis query，否则为general query
        
        #执行query
        result = self.execute_query(user_input, data)
        
        # 记录结束时间  
        end_time = time.time()
        # 打印响应时间  
        print(f"响应时间：{end_time - start_time:.2f}秒")
        
        return result  # 返回查询结果
            
        # except Exception as e:
        #     return f"Error in chat: {str(e)}"
    
def main():
    # 使用示例
    data_sql = DataSQL()
    
    # 测试查询
    user_input = "查询所有前10条item"
    start_time = time.time()
    
    response = data_sql.chat(user_input)
    print(f"\nAI：{response}")
    end_time = time.time()
    print(f"总响应时间：{end_time - start_time:.2f}秒")

if __name__ == "__main__":
    main() 