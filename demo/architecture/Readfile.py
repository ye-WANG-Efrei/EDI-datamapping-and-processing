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
def read_file(pai_, file_path, file_name="dataset_name"):
  
    # 读取数据
    df_gen = pai_.read_excel(file_path)
  
    df_gen_list = []
    # Create the data layer
    for df in df_gen:
        # 规范化列名
        df.columns = [sanitize_column_name(col) for col in df.columns]
        print("df._table_name: ", df._table_name)
        df_name = df._table_name.replace(" ", "_")
        # Replace underscores with hyphens for the dataset name
        dataset_name = df_name.replace("_", "-")
        df_iteration = "{}_{}".format(file_name,dataset_name)
        
        #if the dataset already exists, skip the creation, load the dataset
        try:
            # 创建 PandasAI DataFrame
            df_copy = PaiDataFrame(df)
            df_copy.columns = [sanitize_column_name(col) for col in df_copy.columns]
            
            df_iteration = pai_.create(
                path="myorg/{}".format(dataset_name),
                df=df_copy,
                description="{},columns are in french, there are {} columns".format(df_name, len(df.columns))
            )
            df_gen_list.append(df_iteration)
        except Exception as e:
            print("error: ", e)
            # 加载数据集时也确保列名被正确应用
            df_loaded = pai_.load("myorg/{}".format(dataset_name))
            df_loaded.columns = [sanitize_column_name(col) for col in df_loaded.columns]
            df_gen_list.append(df_loaded)

    return df_gen_list


