#run knowledge limit extract
import os
import codecs
import json
import re
import traceback
import requests
import pandas as pd
from datetime import datetime
from openai import OpenAI
import tools
from pymilvus import MilvusClient, DataType, model, Collection

class GoodRag:
    def __init__(self, indexer, db_type="local"):
        if "local" == db_type:
            self._indexer = indexer
        elif "milvus" == db_type:
            self._client = MilvusClient(
                uri="http://localhost:19530",
                token="root:Milvus"
            )
            self._embedding_factory = model.hybrid.BGEM3EmbeddingFunction(
                model_name='BAAI/bge-m3', # Specify t`he model name
                device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                use_fp16=False # Whether to use fp16. `False` for `device='cpu'`.
            )

    def extract_openai_rlt(self, rlt):
        choices = rlt.choices
        if len(choices)<=0:
            return None
        choice = choices[0]
        content = choice.message.content
        return content

    def call_llm(self, prompt):
        # Set OpenAI's API key and API base to use vLLM's API server.
        openai_api_key = "API_KEY"
        openai_api_base = "IP:PORT"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        chat_response = client.chat.completions.create(
            model="Qwen2.5-72B-Instruct-AWQ",
            messages=[
                {"role": "system", "content": "你是一个问答助手,请根据用户的问题详细回答"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.01
        )
        rlt = self.extract_openai_rlt(chat_response)
        return rlt

    def create_inner_knowledge_rlt(self, q):
        prompt_tmp = """
任务：请根据用户的问题和你已知的知识给出相应的答案。
要求：
1、生成答案的语种要和用户问题的语种保持一致
2、生成答案包括空格、汉字、单词、特殊字符、连续数字等，长度限制在1000字内
用户问题：
%s
当前时间日期：
%s
你所处的国家：
中国
生成格式：
<thoughts>
填充你的思考过程
</thoughts>
<answer>
填充你需要告诉用户的实际答案内容
</answer>
"""
        now_time_str = tools.get_now_time()
        prompt = prompt_tmp % (q, now_time_str)
        llm_rlt = self.call_llm(prompt)
        return llm_rlt 

    def query_parser(self, old_q):
        prompt_tmp = """
你是一个问句解析模块，负责将输入的用户问题的文本转换为便于检索的若干关键词序列，该关键词序列检索获取的结果会被用来作为回答用户问题的参考材料
用户的问题：
%s
生成要求：
1、只生成关键词，不要生成额外的思考信息
2、输出的关键词既要考虑召回的准确性，也要考虑召回的多样性
3、对于可能存在一些其他说法的关键词，需要扩展1到2个同义词输出
4、生成结果最多15个关键词
生成格式：
每个关键词用,分割
"""
        prompt = prompt_tmp % (old_q)
        llm_rlt = self.call_llm(prompt)
        return llm_rlt

    def local_search(self, raw_q, q):
        search_rlt = self._indexer.search(q)
        rerank_search_rlt = search_rlt
        context_list = []
        now_doc_id = -1
        total_word_length = 0
        for now_doc in rerank_search_rlt:
            now_doc_id += 1
            new_item = {"doc_id":now_doc_id, "doc_content":now_doc[2]}
            new_doc = json.dumps(new_item, ensure_ascii=False)
            words = tools.split_text(new_doc)
            if total_word_length + len(words) > 6000:
                break
            context_list.append(new_doc)
            total_word_length += len(words)
        merged_context = "\n".join(context_list)
        if len(merged_context) <= 0:
            merged_context = "空"
        return merged_context 

    def es_search(self, raw_q, q, db_name):
        print('search q:%s' % (q))
        ES_HOST = "http://localhost:9200"
        INDEX_NAME = '%s_rag_index' % (db_name)
        FIELD_NAME = "text"

        headers = {"Content-Type": "application/json"}
        query_body = {
            "query": {
                "match": {  
                    FIELD_NAME: {
                       "query": q 
                    }
                }
           },
           "size": 2,  
           "sort": [ 
                {"_score": {"order": "desc"}}
           ]
        }
        
        response = requests.post(
            url=f"{ES_HOST}/{INDEX_NAME}/_search",
            headers=headers,
            json=query_body
        )
          
        results = response.json()

        top_docs = results["hits"]["hits"]  # 提取前20个文档
        context_list = []
        now_doc_id = -1
        total_word_length = 0
        print('len(top_docs)')
        print(len(top_docs))
        for idx, doc in enumerate(top_docs):
            now_doc_id += 1
            new_item = {"doc_id":now_doc_id, "doc_content":doc["_source"]["text"]}
            new_doc = json.dumps(new_item, ensure_ascii=False)
            words = tools.split_text(new_doc)
            if total_word_length + len(words) > 6000:
                break
            context_list.append(new_doc)
            total_word_length += len(words)
        merged_context = "\n".join(context_list)
        if len(merged_context) <= 0:
            merged_context = "空" 
        return merged_context 

    def milvus_search(self, raw_q, q, db_name):
        collection_name = "%s_rag_collection" % (db_name)
        queries = [q]
        query_vectors = self._embedding_factory.encode_queries(queries)
        dense = query_vectors["dense"]
        data = []
        for i in range(0, len(dense)):
            data.append(dense[i].tolist())

        search_params = {
            "metric_type": "L2",
            "params": {}
        }

        res = self._client.search(
            collection_name=collection_name,
            anns_field = "text_vector",
            data=data,
            limit=2,
            output_fields=["text"],
            search_params=search_params
        )

        context_list = []
        now_doc_id = -1
        total_word_length = 0

        print('len(res[0])')
        print(len(res[0]))

        for item in res[0]:
            now_doc_id += 1
            entity = item['entity']
            text = entity['text']
            new_item = {"doc_id":now_doc_id, "doc_content":text}
            new_doc = json.dumps(new_item, ensure_ascii=False)
            words = tools.split_text(new_doc)
            if total_word_length + len(words) > 6000:
                break
            context_list.append(new_doc)
            total_word_length += len(words)
        merged_context = "\n".join(context_list)
        if len(merged_context) <= 0:
            merged_context = "空"
        return merged_context        

    def to_merge_knowledge_rlt(self, q, inner_knowledge_rlt, search_rlt):
        prompt_tmp = """
外部检索知识：
%s
用户问题：
%s
当前时间日期：
%s
你所处的国家：
中国
任务：请根据用户的问题，以及通过检索获取到的外部知识，优化之前根据内部知识给出的答案。
要求：
1、生成答案的语种要和用户问题的语种保持一致
2、生成答案包括空格、汉字、单词、特殊字符、连续数字等，长度限制在1000字内
3、回复只关注用户问题，不要透露我们内部思考过程
4、外部检索知识不一定为真，按照你的判断有针对性的参考
生成格式：
<inner_thoughts>
填充不参考外部检索知识，仅用模型内部知识对用户问题的思考过程
</inner_thoughts>
<inner_answer>
填充不参考外部检索知识，仅用模型内部知识对用户问题的实际答案内容
</inner_answer>
<summery>
如果有外部检索知识，填充有助于回答问题的外部检索知识的摘要，不超过100个汉字或单词;如果没有有帮助的信息，则不填写
</summery>
<thoughts>
填充你的思考过程
</thoughts>
<answer>
填充你需要告诉用户的实际答案内容
</answer>
"""
        now_time_str = tools.get_now_time()
        prompt = prompt_tmp % (search_rlt, q, now_time_str)
        llm_rlt = self.call_llm(prompt)
        return llm_rlt
 
    def create_answer_2_stage(self, q, db_type="local", db_name=None):
        now_item = {"q":"空", "search_q":"空", "inner_answer":"空", "search_rlt":"空", "answer":"空"}
        try:
            q = q.strip()
            now_item["q"] = q
            inner_knowledge_rlt = self.create_inner_knowledge_rlt(q)
            now_item["inner_answer"] = inner_knowledge_rlt
            #add query parser
            search_q = self.query_parser(q)
            #search_q = q 
            now_item["search_q"] = search_q
            search_rlt = None
            if "local" == db_type:
                search_rlt = self.local_search(q, search_q)
            elif "es" == db_type:
                search_rlt = self.es_search(q, search_q, db_name)
            elif "milvus" ==db_type:
                search_rlt = self.milvus_search(q, search_q, db_name)
            now_item["search_rlt"] = search_rlt
            merge_knowledge_rlt = self.to_merge_knowledge_rlt(q, inner_knowledge_rlt, search_rlt)
            now_item["answer"] = merge_knowledge_rlt
            json_str = json.dumps(now_item, ensure_ascii=False)
        except Exception as e:
            traceback.print_exc()
        return now_item
