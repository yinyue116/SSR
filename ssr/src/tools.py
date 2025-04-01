import os
import re
from datetime import datetime

def walk_through_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

#获取当前人可读的时间字符串,精确到秒 (例如：2023-10-05 14:30:45）
def get_now_time():
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return datetime_string

#按照单个汉字、英文单词、连续数字和其他符号的形式进行切分
def split_text(text):
   pattern = r'[\u4e00-\u9fa5]|[a-zA-Z]+|\d+|[^a-zA-Z0-9\u4e00-\u9fa5]+'
   result = re.findall(pattern, text)
   return result

#写入excel
def write_excel(columns, out_fn):
    df = pd.DataFrame(columns)
    df.to_excel(out_fn, index=False)
