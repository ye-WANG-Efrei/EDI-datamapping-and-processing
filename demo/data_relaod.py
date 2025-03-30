import pandas as pd
import os
from typing import Optional, Union
from pathlib import Path

class DataReload:
    def __init__(self):
        """
        初始化 DataReload 类
        """
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.parquet_path: Optional[str] = None
        self.file_type: Optional[str] = None
        self.columns: Optional[list] = None
        self.shape: Optional[tuple] = None
        
    def load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        加载 CSV 或 Excel 文件
        
        :param file_path: 文件路径
        :return: 加载的数据框，如果失败则返回 None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"错误: 文件 {file_path} 不存在")
                return None
                
            # 设置文件路径和类型
            self.file_path = file_path
            self.file_type = Path(file_path).suffix.lower()
            
            # 设置 parquet 文件路径
            self.parquet_path = str(Path(file_path).with_suffix('.parquet'))
            
            # 如果 parquet 文件存在，则直接加载
            if os.path.exists(self.parquet_path):
                print(f"从 Parquet 文件加载: {self.parquet_path}")
                self.data = pd.read_parquet(self.parquet_path)
            elif self.file_type == '.csv':
                print(f"从 CSV 文件加载: {file_path}")
                self.data = pd.read_csv(file_path)
                # 将数据保存为 parquet 文件
                self.data.to_parquet(self.parquet_path)
            elif self.file_type in ['.xls', '.xlsx']:
                print(f"从 Excel 文件加载: {file_path}")
                self.data = pd.read_excel(file_path)
                # 将数据保存为 parquet 文件
                self.data.to_parquet(self.parquet_path)
            else:
                raise ValueError(f"不支持的文件格式: {self.file_type}。请使用 CSV 或 Excel 文件。")
            
            # 更新数据属性
            self.columns = self.data.columns.tolist()
            self.shape = self.data.shape
            
            print(f"文件加载成功！")
            print(f"数据形状: {self.shape}")
            print(f"列名: {self.columns}")
            
            return self.data
            
        except Exception as e:
            print(f"加载文件时发生错误: {str(e)}")
            import traceback
            print("错误堆栈:", traceback.format_exc())
            return None
            
    def get_data_info(self) -> dict:
        """
        获取数据基本信息
        
        :return: 包含数据信息的字典
        """
        if self.data is None:
            return {"error": "数据未加载"}
            
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "parquet_path": self.parquet_path,
            "shape": self.shape,
            "columns": self.columns,
            "dtypes": self.data.dtypes.to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
            "null_counts": self.data.isnull().sum().to_dict()
        }
        
    def save_data(self, output_path: str) -> bool:
        """
        保存数据到文件
        
        :param output_path: 输出文件路径
        :return: 是否保存成功
        """
        if self.data is None:
            print("无可保存的数据！")
            return False
            
        try:
            output_type = Path(output_path).suffix.lower()
            
            if output_type == '.csv':
                self.data.to_csv(output_path, index=False)
            elif output_type in ['.xls', '.xlsx']:
                self.data.to_excel(output_path, index=False)
            elif output_type == '.parquet':
                self.data.to_parquet(output_path)
            else:
                raise ValueError(f"不支持的文件格式: {output_type}")
                
            print(f"数据已成功保存至 {output_path}")
            return True
            
        except Exception as e:
            print(f"保存文件时发生错误: {str(e)}")
            import traceback
            print("错误堆栈:", traceback.format_exc())
            return False
            
    def clear_data(self):
        """
        清除已加载的数据
        """
        self.data = None
        self.file_path = None
        self.parquet_path = None
        self.file_type = None
        self.columns = None
        self.shape = None
        print("数据已清除")

def main():
    # 使用示例
    data_loader = DataReload()
    
    # 加载文件
    file_path = r'data\employee_data.xlsx'
    data = data_loader.load_file(file_path)
    
    if data is not None:
        # 显示数据信息
        info = data_loader.get_data_info()
        print("\n数据信息:")
        for key, value in info.items():
            print(f"{key}: {value}")
            
        # 保存数据
        output_path = r'data\employee_data_processed.xlsx'
        data_loader.save_data(output_path)
        
        # 清除数据
        data_loader.clear_data()

if __name__ == "__main__":
    main()