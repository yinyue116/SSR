import os
import sys
import codecs
import json
import random

current_directory = os.path.abspath(os.getcwd())

proj_root_path = os.path.join(current_directory, "../../../")
src_root_path = os.path.join(proj_root_path, "src/")
sys.path.append(src_root_path)

import tools
import pandas as pd
 
def read_parquet(file_path):
    # 读取Parquet文件
    df = pd.read_parquet(file_path)
    return df   

def get_text_object(row):
    json_obj = {"text":row["passage"]}
    return json_obj

def create_choice_item(text, is_right):
    return {"text":text, "is_right":is_right}

def create_choices(row):
    merge_choice_list = [
        "A. True",
        "B. False"
    ]
    choices_str = "\n".join(merge_choice_list)
    bool_choice = row["answer"]
    if bool_choice:
        right_choice = "A"
    else:
        right_choice = "B"
    return (choices_str, right_choice)

def get_q_object(row):
    (choices_str, right_choice) = create_choices(row)

    json_obj = {
        "question":row["question"],
        "options":choices_str,
        "answer":right_choice
    }
    return json_obj

def get_row_object(row):
    text_object = get_text_object(row)
    q_object = get_q_object(row)
    rlt = {"t":text_object, "q":q_object}
    return rlt

def parse_cases(input_fpaths, index_root_path):
    tmp_clean_data_fn = os.path.join(index_root_path, "tmp_clean_data.json")
    tmp_f = codecs.open(tmp_clean_data_fn, "w", "utf-8")

    clean_data_path = os.path.join(index_root_path, "test_clean_data/")
    os.makedirs(clean_data_path, exist_ok=True)
    text_fn = os.path.join(clean_data_path, "texts.json")
    text_f = codecs.open(text_fn, "w", "utf-8")

    usecase_data_path = os.path.join(index_root_path, "usecase/")
    os.makedirs(usecase_data_path, exist_ok=True)
    q_fn = os.path.join(usecase_data_path, "questions.json")
    q_f = codecs.open(q_fn, "w", "utf-8")
   
    #为了清除掉只有存入文件才出现的json阶段，先存文件再读取
    for input_fpath in input_fpaths:
        input_df = read_parquet(input_fpath)
        for index, row in input_df.iterrows():
            row_object = get_row_object(row)
            row_str = json.dumps(row_object, ensure_ascii=False)
            tmp_f.write(row_str+"\n")
    tmp_f.close()

    tmp_f = codecs.open(tmp_clean_data_fn, "r", "utf-8")
    for line in tmp_f:
        try:
            line = line.strip()
            json_obj = json.loads(line)
        except:
            print("illegal json format :[%s]" % (line))
            continue
        text_str = json.dumps(json_obj["t"], ensure_ascii=False)
        text_f.write(text_str+"\n")
        q_str = json.dumps(json_obj["q"], ensure_ascii=False)
        q_f.write(q_str+"\n")

    text_f.close()
    q_f.close()

def main():
    #parse data for merge all
    #all_file_paths = tools.walk_through_files("./data/")
    all_file_paths = ["./raw_data/data/validation-00000-of-00001.parquet"]
    merge_all_path = "./merge_all_data/"
    os.makedirs(merge_all_path, exist_ok=True)
    parse_cases(all_file_paths, merge_all_path)

if __name__ == "__main__":
    #暂时不重新生成数据了，因为生成数据会改变选项顺序
    main()
