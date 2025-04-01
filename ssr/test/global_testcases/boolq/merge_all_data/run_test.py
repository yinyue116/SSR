import os
import sys
import codecs
import json
import time
current_directory = os.path.abspath(os.getcwd())

proj_root_path = os.path.join(current_directory, "../../../../")
src_root_path = os.path.join(proj_root_path, "src/")
sys.path.append(src_root_path)

from indexer import QuickIndex
from rag import GoodRag 

corpus_data_path = os.path.join(current_directory, "test_clean_data/")
index_path = os.path.join(current_directory, "index_data/")
model_path = "/home/yinyue/software/colbert_model/jina-colbert-v2/" 
usecase_fn = os.path.join(current_directory, "usecase/questions.json")
test_rlt_path = os.path.join(current_directory, "test_rlt/")
os.makedirs(test_rlt_path, exist_ok=True)
test_rlt_fn = os.path.join(test_rlt_path, "questions_rlt.json")

def prepare_index():
    index = QuickIndex(index_path, model_path)
    index.build_index(corpus_data_path)

def create_choice_q(q_json):
    q_tmp = """
Answer the following single choice question. Your response should be of the following format: '$LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDE).

Question:\n
%s

Options:\n
%s
    """
    q = q_tmp % (q_json["question"], q_json["options"])
    return q

def simple_search_index():
    q = "什么是集成电路设计"
    index = QuickIndex(index_path, model_path)
    index.search_init()
    rlt = index.search(q)
    for item in rlt:
        print(item)

def search_index():
    index = QuickIndex(index_path, model_path)
    index.search_init()
    #db_type = "milvus"
    #db_type = "es"
    db_type = "local"
    good_rag = GoodRag(index, db_type) 

    input_f = codecs.open(usecase_fn, "r", "utf-8")
    rlt_f = codecs.open(test_rlt_fn, "w", "utf-8")
    line_count = -1
    for line in input_f:
        line_count += 1
        if line_count > 100:
            break
        print('Now Test line count %s' % (line_count))
        line = line.strip()
        if len(line) <= 0:
            continue
        q_json = json.loads(line)
        choice_q = create_choice_q(q_json)
        #rag_rlt = good_rag.create_answer_2_stage(choice_q)
        rag_rlt = good_rag.create_answer_2_stage(choice_q, db_type, "boolq")
        q_json["predict"] = rag_rlt 
        out_line = json.dumps(q_json, ensure_ascii=False)
        rlt_f.write(out_line+"\n")
    rlt_f.close()
        
if __name__ == "__main__":
    start_time = time.time()
    print('start time %s' % (start_time))
    #prepare_index()
    end_time = time.time()
    print('end time %s' % (end_time)) 
    index_time_str = f"Time taken to prepare index: {end_time - start_time} seconds"
    print(index_time_str)

    #simple_search_index()
    search_index()
