import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('./data/系统 massy evx 20250325 和SN.xlsx')
print("读取成功")
# 保存为 parquet 文件
df.to_parquet('data/Store_Sales_Price_Elasticity_Promotions_Data.parquet')
print("保存成功")