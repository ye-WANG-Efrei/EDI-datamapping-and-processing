import time
from data_mapping_agent import DataMappingAgent
from data_mapping import DataMapping
from data_sql_RunnableWithMessageHistory import DataSQL

def main():
    # 使用示例
    
    # 测试数据库
    data_sql = DataSQL("deepseek-r1:14b", r'D:\zWork\dream2rich\ollama_agent\EDI-datamapping-and-processing\data\employee_data.parquet', "http://localhost:11434")
    res = data_sql.chat("查询所有数据")
    print("res: ", res)
    res = data_sql.chat("把刚才的数据通过年龄分成两组，一组是20岁以下的，一组是20岁以上的", res)
    print("res: ", res)


    # # 测试数据映射
    # agent = DataMappingAgent()
    
    # # 测试ollama
    # test = agent._call_ollama("你好")
    # print("agent_test:", test)

    # # 加载文件
    # file_path = r'data\employee_data.xlsx'
    # data = agent.load_file(file_path)
    
    # # 展示原始数据
    # print("原始数据:")
    # print(data)
    # print("\n")
    
    # # 生成映射
    # user_request = """
    # 1. 将 full_name 拆分为 first_name 和 last_name
    # 2. 标准化电子邮件为小写
    # 3. 根据年龄分组，创建 age_group 字段
    # """
    # start_time = time.time()
   
    # mapping_result = agent.generate_mapping(user_request)
    # print("AI 生成的映射建议:")
    # print(mapping_result)
    # end_time = time.time()
    # print(f"生成映射建议时间: {end_time - start_time:.2f} 秒")
    # print("\n")
    
    # # 直接应用 AI 生成的映射规则
    # end_time = time.time()
    # print(f"响应时间：{end_time - start_time:.2f}秒")
    # agent.parse_and_apply_mapping(mapping_result)
    
    # # 展示转换后的数据
    # print("转换后的数据:")
    # print(agent.data)
    # print("\n")
    
    # # 保存结果
    # agent.save_mapped_data(r'data\employee_data_mapped.xlsx')

if __name__ == "__main__":
    main() 