import pandas as pd
import json
from langchain_ollama import OllamaLLM
from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import ast
import pandasai as pai
import pathlib
import os # Import os module to handle path separators consistently
import re
from pandasai.config import Config
from pandasai.config import ConfigManager
from pandasai.llm.deepseek_local_llm import DeepSeekLocalLLM
from pandasai.llm.base import LLM
from pandasai.dataframe.base import DataFrame as PaiDataFrame

class DataMappingAgent:
    def __init__(self, model_name='deepseek-r1:14b'):
        # 初始化 Ollama 模型
        self.llm = OllamaLLM(model=model_name)
        
        # 定義映射提示模板
        self.mapping_prompt = PromptTemplate(
            input_variables=['data_preview', 'user_request'],
            template="""
            妳是一個專業的數據映射專家。請根據以下信息：
            數據預覽: {data_preview}
            用戶需求: {user_request}
            
            生成詳細的數據映射和轉換策略，包括：  
            1. 具體的欄位映射關係
            2. 需要執行的轉換邏輯
            3. 特殊處理建議
            
            請提供可執行的 Python 轉換代碼建議。
            """
        )
        
        # 創建 LLM 鏈
        self.mapping_chain = LLMChain(
            llm=self.llm, 
            prompt=self.mapping_prompt
        )
        
        self.data = None
        self.mapping_rules = None
    
    def load_file(self, file_path: str):
        """
        載入 CSV 或 Excel 檔案
        
        :param file_path: 檔案路徑
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的檔案格式。請使用 CSV 或 Excel 檔案。")
            
            print("檔案載入成功！")
            return self.data
        except Exception as e:
            print(f"載入檔案時發生錯誤: {e}")
            return None
    
    def generate_mapping(self, user_request: str):
        """
        根據用戶需求生成數據映射
        
        :param user_request: 用戶的映射需求描述
        :return: 映射建議
        """
        if self.data is None:
            print("請先載入數據檔案！")
            return None
        
        # 取得數據預覽（前5行）
        data_preview = self.data.head().to_string()
        
        # 使用 AI 生成映射建議
        result = self.mapping_chain.run(
            data_preview=data_preview,
            user_request=user_request
        )
        
        self.mapping_rules = result
        return result
    
    def parse_and_apply_mapping(self, mapping_rules):
        """
        解析 AI 生成的映射規則並直接應用
        
        :param mapping_rules: AI 生成的映射建議文本
        :return: 轉換後的 DataFrame
        """
        if self.data is None:
            print("錯誤: 數據未載入，無法應用映射規則")
            return None
            
        if self.data.empty:
            print("錯誤: 數據為空，無法應用映射規則")
            return self.data
            
        try:
            # 使用正則表達式提取 Python 代碼區塊
            code_blocks = re.findall(r'```python(.*?)```', mapping_rules, re.DOTALL)
            
            if code_blocks:
                # 使用最後一個代碼區塊
                mapping_code = code_blocks[-1].strip()
                
                # 創建一個本地命名空間來執行代碼
                local_namespace = {'df': self.data.copy()}
                
                # 執行映射代碼
                exec(mapping_code, globals(), local_namespace)
                
                # 更新 self.data
                self.data = local_namespace['df']
                
                print("成功應用映射規則！")
                return self.data
            else:
                # 如果沒有代碼區塊，嘗試直接解析文本中的轉換邏輯
                self._manual_mapping_from_text(mapping_rules)
        
        except Exception as e:
            print(f"應用映射規則時發生錯誤: {e}")
            # 如果自動解析失敗，嘗試手動映射
            self._manual_mapping_from_text(mapping_rules)
        
        return self.data

    def _manual_mapping_from_text(self, mapping_rules):
        """
        從文本中手動提取映射邏輯
        
        :param mapping_rules: AI 生成的映射建議文本
        """
        # 拆分姓名
        if 'full_name' in self.data.columns and 'first_name' not in self.data.columns:
            self.data[['first_name', 'last_name']] = self.data['full_name'].str.split(n=1, expand=True)
        
        # 標準化電子郵件
        if 'email' in self.data.columns:
            self.data['standardized_email'] = self.data['email'].str.lower()
        
        # 年齡分組
        if 'age' in self.data.columns:
            bins = [0, 26, 50, float('inf')]
            labels = ['青年', '中年', '老年']
            self.data['age_group'] = pd.cut(self.data['age'], bins=bins, labels=labels)
        
        print("已應用基本映射邏輯")

    def save_mapped_data(self, output_path: str):
        """
        儲存映射後的數據
        
        :param output_path: 輸出檔案路徑
        """
        if self.data is None:
            print("無可儲存的數據！")
            return
        
        try:
            if output_path.endswith('.csv'):
                self.data.to_csv(output_path, index=False)
            elif output_path.endswith(('.xls', '.xlsx')):
                self.data.to_excel(output_path, index=False)
            else:
                raise ValueError("不支持的輸出檔案格式")
            
            print(f"數據已成功儲存至 {output_path}")
        except Exception as e:
            print(f"儲存檔案時發生錯誤: {e}")

def main():
    # 使用範例
    agent = DataMappingAgent()
    
    # 載入檔案
    file_path = 'D:\zWork\dream2rich\ollama_agent\EDI-datamapping-and-processing\data\employee_data_mapped.xlsx'
    data = agent.load_file(file_path)
    
    # 展示原始數據
    print("原始數據:")
    print(data)
    print("\n")
    
    # 生成映射
    user_request = """
    1. 將 full_name 拆分為 first_name 和 last_name
    2. 標準化電子郵件為小寫
    3. 根據年齡分組，創建 age_group 欄位
    """
    mapping_result = agent.generate_mapping(user_request)
    print("AI 生成的映射建議:")
    print(mapping_result)
    print("\n")
    
    # 直接應用 AI 生成的映射規則
    
    agent.parse_and_apply_mapping(mapping_result)
    # 展示轉換後的數據
    print("轉換後的數據:")
    print(agent.data)
    print("\n")
    
    # 儲存結果
    agent.save_mapped_data('mapped_employee_data.xlsx')

if __name__ == "__main__":
    main() 