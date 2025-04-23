import pandasai as pai
import pathlib
import os # Import os module to handle path separators consistently
from pandasai.config import Config
from pandasai.config import ConfigManager
from pandasai.llm.deepseek_local_llm import DeepSeekLocalLLM
from pandasai.llm.base import LLM

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
df_genateq = pai.read_excel(data_file_path_genateq)
df_tbc = pai.read_excel(data_file_path_tbc)

# Create the data layer
for df in df_genateq:
    df_name = df._table_name.replace(" ", "_")
    # Replace underscores with hyphens for the dataset name
    dataset_name = df_name.replace("_", "-")
    print("df_name: ", df_name) 
    print("path: ", "myorg/{}".format(dataset_name))
    pai.create(
        path="myorg/{}".format(dataset_name),
        df=df,
        description="{},columns are in french, there are {} columns,{} respectively".format(df_name,len(df.columns),df.columns)
    )


#load the tbc dataset
tbc_dataset_name = "tbc"  # Already in correct format
tbc = pai.create(
    path="myorg/{}".format(tbc_dataset_name),
    df=df_tbc,
    description="tbc dataset, there is only 1 sheet in the excel file, the dataframe name is the sheet name"
)   


# response = companies.chat("top 5 employees by salary")
#print(response)
request = '''Given two datasets, filter both to include only records with 'location' set to 'Lille'. Then, determine how many items are identical between the two datasets and how many are different.'''
res= pai.chat(request,genateq,tbc)
# res= pai.chat("what is the average salary of the employees?",companies)
print(res)