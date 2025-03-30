import pandas as pd
import json
import requests
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
import os
import time

MODEL = "deepseek-r1:14b"

class DataMappingAgent:
    def __init__(self, model_name=MODEL):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        # 初始化 ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            base_url=self.base_url,
            temperature=0.1,
            num_thread=8,
            timeout=30,
            stop=["</s>", "user:", "assistant:", "<think>", "</think>"]
        )
        
        self.data = None
        self.mapping_rules = None
        self.conversation = []
    
    def load_file(self, file_path: str):
        """
        加载 CSV 或 Excel 文件
        
        :param file_path: 文件路径
        """
        parquet_file_path = file_path.split(".")[0]+".parquet"
        try:
            # 如果 parquet 文件存在，则直接加载
            if os.path.exists(parquet_file_path):
                self.data = pd.read_parquet(parquet_file_path)
            elif file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
                # 将数据保存为 parquet 文件
                self.data.to_parquet(parquet_file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
                # 将数据保存为 parquet 文件
                self.data.to_parquet(parquet_file_path)
            else:
                raise ValueError("不支持的文件格式。请使用 CSV 或 Excel 文件。")
            
            print("文件加载成功！")
            return self.data
        except Exception as e:
            print(f"加载文件时发生错误: {e}")
            return None


    def _call_ollama(self, prompt: str, if_send_require = False) -> str:
        """
        直接调用 Ollama API
        
        :param prompt: 提示文本
        :return: AI 响应
        """
        try:
            print("\n正在调用 Ollama API...")
            print(f"API URL: {self.base_url}/api/generate")
            print(f"模型: {self.model_name}")
            
            # 检查 Ollama 服务是否运行
            try:
                response = requests.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    print("错误: Ollama 服务未运行或无法访问")
                    return None
                print("Ollama 服务正常运行")
            except requests.exceptions.ConnectionError:
                print("错误: 无法连接到 Ollama 服务，请确保服务已启动")
                return None
            
            # 发送生成请求
            if not if_send_require:
                print("\n发送生成请求...")
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 2048,
                            "num_thread": 8,
                            "num_predict": 1024,
                            "stop": ["</s>", "user:", "assistant:", "<think>", "</think>"]
                        }
                    }
                )
                
                print(f"响应状态码: {response.status_code}")
                if response.status_code != 200:
                    print(f"错误响应: {response.text}")
                    return None
                    
                response_data = response.json()
                print("成功获取响应")
                return response_data.get("response")
            
            
        except Exception as e:
            print(f"\n调用 Ollama API 时发生错误: {str(e)}")
            print("错误类型:", type(e).__name__)
            import traceback
            print("错误堆栈:", traceback.format_exc())
            return None

    def generate_mapping(self, user_request: str):
        """
        根据用户需求生成数据映射
        
        :param user_request: 用户的映射需求描述
        :return: 映射建议
        """
        if self.data is None:
            print("请先加载数据文件！")
            return None
        
        # 获取数据预览（前5行）
        data_preview = self.data.head().to_string()
        
        # 构建对话历史
        conversation_history = ""
        if self.conversation:
            print("\n包含之前的对话历史:",self.conversation)
            for msg in self.conversation:
                role = "用户" if isinstance(msg, HumanMessage) else "AI"
                conversation_history += f"{role}: {msg.content}\n"
                print(f"{role}: {msg.content}")
        
        # 构建消息
        messages = [
            SystemMessage(content="""你是一个专业的数据映射专家。请根据用户的需求和数据预览生成详细的数据映射和转换策略，包括：
            1. 具体的字段映射关系
            2. 需要执行的转换逻辑
            3. 特殊处理建议
            
            请提供可执行的 Python 转换代码建议。"""),
            HumanMessage(content=f"""之前的对话历史:
            {conversation_history}

            当前数据预览:
            {data_preview}

            当前用户需求:
            {user_request}""")
        ]
        
        print("\n发送请求到 AI...")
        print("请求内容:", messages[-1].content)
        
        try:
            # 使用 llm.invoke 生成响应
            response = self.llm.invoke(messages)
            result = response.content
            
            print("\nAI 返回结果:")
            print("结果类型:", type(result))
            print("结果内容:", result)
            print("结果长度:", len(result) if result else 0)
            
            if not result:
                print("警告: AI 返回了空结果")
                return None
            
            # 添加对话到历史记录
            self.conversation.extend(messages)
            self.conversation.append(AIMessage(content=result))
                
            self.mapping_rules = result
            return result
            
        except Exception as e:
            print(f"\n调用 AI 时发生错误: {str(e)}")
            print("错误类型:", type(e).__name__)
            import traceback
            print("错误堆栈:", traceback.format_exc())
            return None
    
    def parse_and_apply_mapping(self, mapping_rules):
        """
        解析 AI 生成的映射规则并直接应用
        
        :param mapping_rules: AI 生成的映射建议文本
        :return: 转换后的 DataFrame
        """
        print("\n开始解析映射规则...")
        print("收到的映射规则:", mapping_rules)
        
        if not self.data.empty:
            try:
                print("\n尝试提取代码块...")
                # 使用正则表达式提取 Python 代码块
                code_blocks = re.findall(r'```python(.*?)```', mapping_rules, re.DOTALL)
                print(f"找到 {len(code_blocks)} 个代码块")
                
                if code_blocks:
                    print("\n使用最后一个代码块...")
                    # 使用最后一个代码块
                    mapping_code = code_blocks[-1].strip()
                    print("提取的代码:", mapping_code)
                    
                    # 创建一个本地命名空间来执行代码
                    local_namespace = {'df': self.data.copy()}
                    
                    print("\n执行映射代码...")
                    # 执行映射代码
                    exec(mapping_code, globals(), local_namespace)
                    
                    # 更新 self.data
                    self.data = local_namespace['df']
                    
                    print("成功应用映射规则！")
                    return self.data
                else:
                    print("\n没有找到代码块，尝试手动映射...")
                    # 如果没有代码块，尝试直接解析文本中的转换逻辑
                    self._manual_mapping_from_text(mapping_rules)
            
            except Exception as e:
                print(f"\n应用映射规则时发生错误: {str(e)}")
                print("错误类型:", type(e).__name__)
                print("尝试手动映射...")
                # 如果自动解析失败，尝试手动映射
                self._manual_mapping_from_text(mapping_rules)
        else:
            print("\n数据为空，无法应用映射规则")
        
        return self.data

    def _manual_mapping_from_text(self, mapping_rules):
        """
        从文本中手动提取映射逻辑
        
        :param mapping_rules: AI 生成的映射建议文本
        """
        print("开始手动映射")
        # 拆分姓名
        if 'full_name' in self.data.columns and 'first_name' not in self.data.columns:
            self.data[['first_name', 'last_name']] = self.data['full_name'].str.split(n=1, expand=True)
        
        # 标准化电子邮件
        if 'email' in self.data.columns:
            self.data['standardized_email'] = self.data['email'].str.lower()
        
        # 年龄分组
        if 'age' in self.data.columns:
            bins = [0, 26, 50, float('inf')]
            labels = ['青年', '中年', '老年']
            self.data['age_group'] = pd.cut(self.data['age'], bins=bins, labels=labels)
        
        print("已应用基本映射逻辑")

    def save_mapped_data(self, output_path: str):
        """
        保存映射后的数据
        
        :param output_path: 输出文件路径
        """
        if self.data is None:
            print("无可保存的数据！")
            return
        
        try:
            if output_path.endswith('.csv'):
                self.data.to_csv(output_path, index=False)
            elif output_path.endswith(('.xls', '.xlsx')):
                self.data.to_excel(output_path, index=False)
            else:
                raise ValueError("不支持的文件格式")
            
            print(f"数据已成功保存至 {output_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {e}")

def main():
    # 使用示例
    agent = DataMappingAgent()
    
    # 设置文件路径和用户需求
    input_file = r'data\employee_data.xlsx'
    output_file = r'data\employee_data_mapped.xlsx'
    user_request = """
    1. 将 full_name 拆分为 first_name 和 last_name
    2. 标准化电子邮件为小写
    3. 根据年龄分组，创建 age_group 字段
    """
    
    # 运行完整流程
    success = agent.run(input_file, output_file, user_request)
    
    if success:
        print("数据处理成功完成！")
    else:
        print("数据处理过程中出现错误，请检查日志。")

if __name__ == "__main__":
    main() 