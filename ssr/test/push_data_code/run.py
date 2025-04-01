from pymilvus import MilvusClient, DataType, model, Collection
import codecs
import json
import secrets
import string
import json

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

def create_schema(collection_name):
    client.drop_collection(collection_name)
    # 1. Create schema
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    # 2. Add fields to schema
    schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=32, is_primary=True)
    schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=10240)
    schema.add_field(field_name="text_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    # 3. Prepare index parameters
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="text_vector", 
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )

    #create collection
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

def create_random_id(): 
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(32))

def create_doc_vectors(in_fn, out_fn):
    bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3', # Specify t`he model name
        device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False # Whether to use fp16. `False` for `device='cpu'`.
    )

    in_f = codecs.open(in_fn, "r", "utf-8")
    docs = []
    for line in in_f:
        line = line.strip()
        json_obj = json.loads(line)
        docs.append(json_obj["text"])
    in_f.close()

    docs_embeddings = bge_m3_ef.encode_documents(docs) 
    dense = docs_embeddings["dense"]
    out_f = codecs.open(out_fn, "w", "utf-8")
    for i in range(0, len(dense)):
        id_str = create_random_id()
        text = docs[i]
        text_vector = dense[i].tolist()
        out_obj = {"id":id_str, "text": text, "text_vector":text_vector}
        out_str = json.dumps(out_obj, ensure_ascii=False)
        out_f.write(out_str+"\n")
    out_f.close()

def push_data(collection_name, in_fn):
    res = client.list_collections()
    print(res)

    data = []
    in_f = codecs.open(in_fn, "r", "utf-8")
    line_count = 0
    for line in in_f:
        line_count += 1
        line = line.strip()
        json_obj = json.loads(line)
        data.append(json_obj)
    client.insert(collection_name=collection_name, data=data)

def search_milvus(collection_name):
    bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
        model_name='BAAI/bge-m3', # Specify t`he model name
        device='cuda:0', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=False # Whether to use fp16. `False` for `device='cpu'`.
    )

    queries = ["I really very happy"]
    query_vectors = bge_m3_ef.encode_queries(queries)
    dense = query_vectors["dense"]
    data = []
    for i in range(0, len(dense)):
        data.append(dense[i].tolist())

    search_params = {
        "metric_type": "L2",
        "params": {}
    }

    res = client.search(
        collection_name=collection_name,
        anns_field = "text_vector",
        data=data,
        limit=10,
        output_fields=["text"],
        search_params=search_params
    )
    
    for item in res[0]:
        print(item)

 
def run():
    #corpus_name = "openbookqa"
    #corpus_name = "sciq"
    corpus_name = "boolq"
    collection_name = "%s_rag_collection" % (corpus_name)

    create_schema(collection_name)
    print('finish create schema')
    in_fn = "../global_testcases/%s/merge_all_data/test_clean_data/texts.json" % (corpus_name) 
    out_fn = "../global_testcases/%s/merge_all_data/push_data/milvus_data.json" % (corpus_name)
    create_doc_vectors(in_fn, out_fn)
    print('finish create doc vectors')
    push_data(collection_name, out_fn)
    print('finish push data')
    search_milvus(collection_name)


if __name__ == "__main__":
    run()
