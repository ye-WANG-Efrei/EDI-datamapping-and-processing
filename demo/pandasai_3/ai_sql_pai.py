import pandasai as pai
import pathlib
import os # Import os module to handle path separators consistently
import re
from pandasai.config import Config
from pandasai.config import ConfigManager
from pandasai.llm.deepseek_local_llm import DeepSeekLocalLLM
from pandasai.llm.base import LLM
from pandasai.dataframe.base import DataFrame as PaiDataFrame


def sanitize_column_name(column_name):
    """Sanitize column names to avoid SQL keywords and special characters."""
    # Convert to lowercase
    sanitized = column_name.lower()
    
    # Replace spaces and special characters with underscores
    sanitized = re.sub(r'[^a-z0-9_]', '_', sanitized)
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    
    # Replace common SQL keywords with safer alternatives
    replacements = {
        'desc': 'description',
        'control': 'ctrl',
        'code': 'id',
        'original': 'source',
        'status': 'state',
        'count': 'quantity',
        'number': 'num'
    }
    
    for keyword, replacement in replacements.items():
        sanitized = re.sub(r'\b' + keyword + r'\b', replacement, sanitized)
    
    return sanitized

# Get the directory where the script is located
script_dir = pathlib.Path(__file__).parent.resolve()
# Construct the path to the data file, going up two levels to the project root, then into 'data'
# Using os.path.join ensures correct path separators on Windows
data_file_path_genateq = os.path.join(script_dir.parent.parent, "data", "HUAWEI - Export inventaire GenateQ(1).xlsx")
data_file_path_tbc = os.path.join(script_dir.parent.parent, "data", "TBC Lille.xlsx")


# # 创建 DeepSeekLocalLLM 实例
# deepseek_llm = DeepSeekLocalLLM(model_name="deepseek-r1:14b", host="http://localhost:11434")
# # 创建配置对象
# config = Config(llm=deepseek_llm)
# # 设置全局配置
# ConfigManager.set(config.model_dump())

# 使用 API 密钥
pai.api_key.set("PAI-a4927d55-bcb4-4e97-8cef-4250564d1f69")

# 读取数据
df_genateq = pai.read_excel(data_file_path_tbc)
df_tbc = pai.read_excel(data_file_path_tbc)

df_genateq_list = []
# Create the data layer
for df in df_genateq:
    # 规范化列名
    df.columns = [sanitize_column_name(col) for col in df.columns]
    print("df._table_name: ", df._table_name)
    df_name = df._table_name.replace(" ", "_")
    # Replace underscores with hyphens for the dataset name
    dataset_name = df_name.replace("_", "-")
    df_iteration = "genateq_{}".format(dataset_name)
    
    #if the dataset already exists, skip the creation, load the dataset
    try:
        # 创建 PandasAI DataFrame
        df_copy = PaiDataFrame(df)
        df_copy.columns = [sanitize_column_name(col) for col in df_copy.columns]
        
        df_iteration = pai.create(
            path="myorg/{}".format(dataset_name),
            df=df_copy,
            description="{},columns are in french, there are {} columns".format(df_name, len(df.columns))
        )
        df_genateq_list.append(df_iteration)
    except Exception as e:
        print("error: ", e)
        # 加载数据集时也确保列名被正确应用
        df_loaded = pai.load("myorg/{}".format(dataset_name))
        df_loaded.columns = [sanitize_column_name(col) for col in df_loaded.columns]
        df_genateq_list.append(df_loaded)



print(df_genateq_list[0])
request = '''show me all the columns in the dataset'''
res = pai.chat(request, *df_genateq_list)
# res= pai.chat("what is the average salary of the employees?",companies)
print(res)