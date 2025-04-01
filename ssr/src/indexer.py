import faiss
import traceback
import codecs
import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import h5py
import cupy as cp
import re
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import torch
import torch.nn.functional as F
import concurrent.futures

import tools

class QuickIndex:
    def __init__(self, root_path, model_path, dim=128):
        #token对应的vector维度
        self._dim = dim
        #各类正排、倒排文件所存放的根目录
        self._root_path = root_path
        os.makedirs(self._root_path, exist_ok=True)
        
        #分batch存放token的vector正排信息
        self._batch_tensor_root_path = os.path.join(self._root_path, "batch_tensor/")
        os.makedirs(self._batch_tensor_root_path, exist_ok=True)

        #分batch并且分segment存放token的vector正排信息
        self._split_batch_tensor_root_path = os.path.join(self._root_path, "split_batch_tensor/")
        os.makedirs(self._split_batch_tensor_root_path, exist_ok=True)
        
        #存放一些全局信息
        self._global_data_path = os.path.join(self._root_path, "global_data/")
        os.makedirs(self._global_data_path, exist_ok=True)
        #存放segment分段各自对应的哪个batch，以h5列存的方式存储
        self._segment2batch_fn = os.path.join(self._global_data_path, "segment2batch.h5")

        #存放方便快速获取的列存正排信息，以h5列存的方式存储
        self._simple_corpus_root_path = os.path.join(self._root_path, "simple_corpus/")
        os.makedirs(self._simple_corpus_root_path, exist_ok=True)

        #存放faiss的向量索引文件
        self._faiss_index_path = os.path.join(self._root_path, "faiss_index/")
        os.makedirs(self._faiss_index_path, exist_ok=True)
        self._faiss_index_fn = os.path.join(self._faiss_index_path, "token_index.faiss")

        #colbert模型路径
        self._model_path = model_path
        #加载colbert模型
        self._ckpt = Checkpoint(self._model_path, colbert_config=ColBERTConfig()) 

    #start build index functions
    def parse_one_batch_from_file(self, input_file_name, segment_batch_output_root_path, split_segment_batch_output_root_path,  simple_corpus_root_path, batch_id, last_segment_id, last_token_id, dim):
        segment_batch_output_real_fn = "segment_batch_%s.h5" % ( batch_id )
        segment_batch_output_file_name = os.path.join(segment_batch_output_root_path, segment_batch_output_real_fn)
        print('segment_batch_output_file_name')
        print(segment_batch_output_file_name)

        split_segment_batch_output_real_fn = "split_segment_batch_%s.h5" % (batch_id)
        split_segment_batch_output_file_name = os.path.join(split_segment_batch_output_root_path, split_segment_batch_output_real_fn)
        print('split_segment_batch_output_file_name')
        print(split_segment_batch_output_file_name)

        output_simple_corpus_real_fn = "simple_corpus_batch_%s.h5" % (batch_id)
        output_simple_corpus_fn = os.path.join(simple_corpus_root_path, output_simple_corpus_real_fn)

        simple_raw_texts = []
        simple_segment_ids = []
        #doc_matrix = np.empty((0, dim), dtype=np.float32)
        segmentid2tokeninfos = [] 
        segment_ids = np.empty((0, 1), dtype=np.int32)
        token_ids = np.empty((0, 1), dtype=np.int32)
        in_f = codecs.open(input_file_name, "r", "utf-8")

        for line in in_f:
            last_segment_id += 1
            line = line.strip()
            #print(line)
            json_obj = json.loads(line)
            

            #收集简单正文信息
            text = json_obj["text"]
            simple_raw_texts.append(text)
            simple_segment_ids.append(last_segment_id)

            #收集tensor索引
            doc_vector = self._ckpt.docFromText([text])[0].tolist()
            tmp_doc_matrix = np.array(doc_vector, dtype=np.float32)
            #doc_matrix = np.vstack((doc_matrix, tmp_doc_matrix))

            token_length = len(doc_vector)
            segment_id_array = np.full(token_length, last_segment_id)
            trans_segment_id_array = segment_id_array.reshape(segment_id_array.shape[0],1)
            segment_ids = np.vstack((segment_ids, trans_segment_id_array))

            token_id_array = np.arange(last_token_id+1, last_token_id+1+token_length)
            last_token_id = token_id_array[-1]
            trans_token_id_array = token_id_array.reshape(token_id_array.shape[0],1)
            token_ids = np.vstack((token_ids, trans_token_id_array))

            segmentid2tokeninfos.append((last_segment_id, tmp_doc_matrix, trans_token_id_array))
        in_f.close()

        np_simple_raw_texts = np.array(simple_raw_texts, dtype=str)
        np_simple_raw_texts_bytes = np.array([s.encode('utf-8') for s in np_simple_raw_texts])
        np_simple_segment_ids = np.array(simple_segment_ids)
        vlen_dtype = h5py.special_dtype(vlen=str)
        with h5py.File(output_simple_corpus_fn, 'w') as simple_f:
            simple_f.create_dataset('chapter_infos', data=np_simple_raw_texts_bytes)
            simple_f.create_dataset('segment_ids', data=np_simple_segment_ids)

        with h5py.File(segment_batch_output_file_name, 'w') as f:
            #f.create_dataset('token_doc_vectors', data=doc_matrix)
            f.create_dataset('segment_ids', data=segment_ids)
            f.create_dataset('token_ids', data=token_ids)

        with h5py.File(split_segment_batch_output_file_name, 'w') as f:
            for (segment_id, sub_matrix, sub_token_id_array) in segmentid2tokeninfos:
                f.create_dataset(f'segment2vector_{segment_id}', data=sub_matrix)
                f.create_dataset(f'segment2tokenids_{segment_id}', data=sub_token_id_array)
        return (last_segment_id, last_token_id)

    def build_forward_index(self, raw_data_path):
        print('start build forward index')
        global_batch_id = -1
        global_segment_id = -1
        global_token_id = -1
        #存储segment对应的batch的id
        global_segment2batch_ids = np.empty((0, 1), dtype=np.int32)
        #存储segment对应的batch内部的位置
        global_segment2batch_poses = np.empty((0, 1), dtype=np.int32)

        #从文件夹遍历所有的文件
        file_paths = tools.walk_through_files(raw_data_path)
        for file_path in file_paths:
            print("raw_file_path:%s" % (file_path))
            global_batch_id += 1
            #TODO
            """
            if global_batch_id > 2:
                break
            """
            if global_batch_id % 10 == 0:
                print(global_batch_id)
            try:
                (last_segment_id, last_token_id) = self.parse_one_batch_from_file(file_path, self._batch_tensor_root_path, self._split_batch_tensor_root_path, self._simple_corpus_root_path, global_batch_id, global_segment_id, global_token_id, self._dim)
            except Exception as e:
                traceback.print_exc()
                global_batch_id -= 1
                continue
            tmp_segment_count = last_segment_id-global_segment_id
            segment2batch_ids = np.full(tmp_segment_count, global_batch_id)
            trans_segment2batch_ids = segment2batch_ids.reshape(segment2batch_ids.shape[0],1)
            global_segment2batch_ids = np.vstack((global_segment2batch_ids, trans_segment2batch_ids))

            segment2batch_poses = np.arange(tmp_segment_count)
            trans_segment2batch_poses = segment2batch_poses.reshape(segment2batch_poses.shape[0],1)
            global_segment2batch_poses = np.vstack((global_segment2batch_poses, trans_segment2batch_poses))

            global_segment_id = last_segment_id
            global_token_id = last_token_id

            with h5py.File(self._segment2batch_fn, 'w') as f:
                f.create_dataset('batch_ids', data=global_segment2batch_ids)
                f.create_dataset('batch_poses', data=global_segment2batch_poses)
        print('end build forward index')

    def extract_batch_id(self, file_name):
        # 正则表达式匹配 "segment_batch_" 后面的数字
        match = re.search(r'.*segment_batch_(\d+)\.h5', file_name)
        if match:
            # 提取匹配到的数字部分
            batch_id = int(match.group(1))
            return batch_id
        else:
            raise ValueError("Filename format is incorrect")

    def load_file_tensor_dataset(self, file_name):
        batch_id = self.extract_batch_id(file_name)
        with h5py.File(file_name, 'r') as f:
            #token_doc_matrix_loaded = f['token_doc_vectors'][:]
            segment_ids_loaded = f['segment_ids'][:]
            token_ids_loaded = f['token_ids'][:]
        return (batch_id, segment_ids_loaded, token_ids_loaded)

    def load_segid2token_infos(self, file_name):
        batch_id = self.extract_batch_id(file_name)
        segid2token_infos = {}
        prefixs = ["segment2vector_", "segment2tokenids_"]
        with h5py.File(file_name, 'r') as f:
            keys = f.keys()
            for key in keys:
                now_value = f[key][:]
                for prefix in prefixs:
                    if prefix not in key:
                        continue
                    segment_id = key.replace(prefix, "")
                    if segment_id not in segid2token_infos:
                        segid2token_infos[segment_id] = {}
                    segid2token_infos[segment_id][prefix] = now_value
                    break
        return segid2token_infos 
                
    def build_inverted_index(self):
        print('start build inverted index')
        #file_paths = tools.walk_through_files(self._batch_tensor_root_path)
        file_paths = tools.walk_through_files(self._split_batch_tensor_root_path)
        dim = self._dim
        m = 8
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, 1024, m, 8)
        index = faiss.IndexIDMap(index)
        all_data = []
        all_ids = []
        file_count = 0
        for file_path in file_paths:
            file_count += 1
            if file_count % 10 ==0:
                print(file_count)
            #(batch_id, segment_ids, token_ids) = self.load_file_tensor_dataset(file_path)
            segid2token_infos = self.load_segid2token_infos(file_path)
            for segid in segid2token_infos:
                tokeninfos = segid2token_infos[segid]
                token_doc_matrix = tokeninfos["segment2vector_"]
                token_ids = tokeninfos["segment2tokenids_"]
                reshape_token_ids = token_ids.reshape((token_ids.shape[0],))
                all_data.extend(token_doc_matrix)
                all_ids.extend(reshape_token_ids)

        combined_data = np.vstack(all_data)
        combined_ids = np.array(all_ids)
        print('start train')
        index.train(combined_data)
        print('end train')
        print('start add data to index')
        index.add_with_ids(combined_data, combined_ids)
        print('end add data to index')
        print('start write index')
        faiss.write_index(index, self._faiss_index_fn)  
        print('end write index')
        print('end build inverted index')

    #main build index function
    def build_index(self, raw_data_path):
        self.build_forward_index(raw_data_path)
        self.build_inverted_index()
    #end build index functions

    #start search index functions
    def unique_positive_integers(self, matrix):
        # 筛选出矩阵中的所有正整数
        positive_integers = matrix[matrix >= 0]
        # 使用集合去重
        unique_integers = set(positive_integers)
        # 转换为列表并排序
        sorted_unique_integers = sorted(list(unique_integers))
        #print("duplicate count :%s" % count_duplicates(sorted_unique_integers))
        return sorted_unique_integers

    def create_need_batch_ids2seg_ids(self, need_segment_ids, seg2batch_id_maps):
        group_dict = {}
        for element_id, group_ids in zip(need_segment_ids, seg2batch_id_maps):
            for group_id in group_ids:
                # 如果分组id不在字典中，初始化一个空列表
                if group_id not in group_dict:
                    group_dict[group_id] = []
                # 将元素id添加到对应分组id的列表中
                group_dict[group_id].append(element_id)
        return group_dict

    #根据匹配到的segment和batch的ids，加载可以进行明细打分的tensor信息
    def load_matched_tensors(self, batch_ids2seg_ids, batch_tensor_root_path):
        seg2matrix = {}
        for (batch_id, need_segment_ids) in batch_ids2seg_ids.items():
            load_batch_matched_start_time = time.time()
            print("load matched tensor batch_id:%s need_segment_ids:%s" % (batch_id, len(need_segment_ids)))
            pure_input_fn = "segment_batch_%s.h5" % ( batch_id )
            input_fn = os.path.join(batch_tensor_root_path, pure_input_fn)
            (batch_id, token_doc_matrix, segment_ids, token_ids) = self.load_file_tensor_dataset(input_fn)
            load_batch_matched_from_file_time = time.time()
            need_segid_cycle_count = 0
            dup_count = 0
            for now_segment_id in need_segment_ids:
                need_segid_cycle_count += 1
                mask = segment_ids[:,0] == now_segment_id
                if now_segment_id in seg2matrix:
                    dup_count += 1
                seg2matrix[now_segment_id] = token_doc_matrix[mask]
            print('need_segid_cycle_count:%s, dup_count:%s' % (need_segid_cycle_count, dup_count))
            load_batch_matched_end_file_time = time.time()
            load_batch_matched_time_str = f"Time taken to load batch matched: {load_batch_matched_from_file_time - load_batch_matched_start_time} seconds"
            print(load_batch_matched_time_str)
            load_batch_extract_time_str = f"Time taken to load batch extract: {load_batch_matched_end_file_time - load_batch_matched_from_file_time} seconds"
            print(load_batch_extract_time_str)
        print("len(seg2matrix):%s" % (len(seg2matrix)))
        return seg2matrix

    def load_split_seg2matrix(self, need_segment_ids, input_fn):
        seg2matrix = {}
        with h5py.File(input_fn, 'r') as f:
            for need_segment_id in need_segment_ids:
                now_key = f'segment2vector_{need_segment_id}'
                now_matrix = f[now_key][:]
                seg2matrix[need_segment_id] = now_matrix
        return seg2matrix

    """
    def load_split_seg2matrix_2(self, need_segment_ids, input_fn):
        seg2matrix = {}
        with h5py.File(input_fn, 'r') as f:
            dataset = f['segments']
            for data in dataset:
                segment_id = data['segment_id']
                flattened_matrix = data['token_matrix']
                matrix_shape = data['matrix_shape']
                token_matrix = flattened_matrix.reshape(matrix_shape)
                if segment_id in need_segment_ids:
                    seg2matrix[segment_id] = token_matrix
        return seg2matrix
    """

    def load_matched_tensors_by_splitfile(self, batch_ids2seg_ids, split_batch_tensor_root_path):
        seg2matrix = {}
        for (batch_id, need_segment_ids) in batch_ids2seg_ids.items():
            pure_input_fn = "split_segment_batch_%s.h5" % ( batch_id )
            input_fn = os.path.join(split_batch_tensor_root_path, pure_input_fn)
            sub_seg2matrix = self.load_split_seg2matrix(need_segment_ids, input_fn)
            #sub_seg2matrix = self.load_split_seg2matrix_2(need_segment_ids, input_fn)
            seg2matrix.update(sub_seg2matrix)
        return seg2matrix

    def compute_segment_score(self, seg_id, q_tensor, tensor_matrix):
        #使用gpu计算
        simi_matrix = cp.dot(cp.array(q_tensor), cp.array(tensor_matrix).T).get()
        score = np.sum(np.max(simi_matrix, axis=1))
        return score

    def search_q_faiss(self, q_info, index, token2seg_id_maps, seg2batch_id_maps, batch_tensor_root_path, split_batch_tensor_root_path, simple_corpus_root_path):
        q_search_start_time = time.time()
        q_tensor = np.array(q_info[1], dtype=np.float32)
        D, I = index.search(q_tensor, 10000)
        q_search_faiss_time = time.time()
        de_token_ids = self.unique_positive_integers(I)
        q_get_de_token_ids_time = time.time()

        need_segment_ids = np.array(token2seg_id_maps[de_token_ids])
        de_segment_ids = self.unique_positive_integers(need_segment_ids)

        q_get_de_segment_ids_time = time.time()
        need_batch_ids = np.array(seg2batch_id_maps[de_segment_ids])
        batch_ids2seg_ids = self.create_need_batch_ids2seg_ids(de_segment_ids, need_batch_ids)
        q_get_batch_ids2seg_ids_time = time.time()
        #老的加载过程
        #matched_tensors = self.load_matched_tensors(batch_ids2seg_ids, batch_tensor_root_path)
        #新的加载过程
        matched_tensors = self.load_matched_tensors_by_splitfile(batch_ids2seg_ids, split_batch_tensor_root_path)
        q_load_matched_tensors_time = time.time()
        raw_rlt = []
        for (seg_id, tensor_matrix) in matched_tensors.items():
            now_score = self.compute_segment_score(seg_id, q_tensor, tensor_matrix)
            raw_rlt.append((seg_id, now_score))
        q_compute_all_scores_time = time.time()
        rlt = sorted(raw_rlt, key = lambda x : -x[1])
        q_sorted_rlt_time = time.time()

        search_faiss_time_str = f"Time taken to search faiss: {q_search_faiss_time - q_search_start_time} seconds"
        print(search_faiss_time_str)
        get_de_token_ids_time_str = f"Time taken to get de token_ids time str: {q_get_de_token_ids_time - q_search_faiss_time} seconds"
        print(get_de_token_ids_time_str)
        get_de_segment_ids_time_str = f"Time taken to get de segment_ids time str: {q_get_de_segment_ids_time - q_get_de_token_ids_time} seconds"
        print(get_de_segment_ids_time_str)
        get_batch_ids2seg_ids_time_str = f"Time taken to get batch_ids2seg_ids time str: {q_get_batch_ids2seg_ids_time - q_get_de_segment_ids_time} seconds"
        print(get_batch_ids2seg_ids_time_str)
        load_matched_time_str = f"Time taken to load matched time str: {q_load_matched_tensors_time-q_get_batch_ids2seg_ids_time} seconds"
        print(load_matched_time_str)
        compute_all_score_time_str = f"Time taken to compute all scores time str: {q_compute_all_scores_time - q_load_matched_tensors_time} seconds"
        print(compute_all_score_time_str)
        return rlt
        

    def add_text_infos(self, search_rlt, seg2batch_id_maps, simple_corpus_root_path):
        batch2segs = {}
        for search_item in search_rlt:
            seg_id = search_item[0]
            batch_id = seg2batch_id_maps[seg_id][0]
            if batch_id not in batch2segs:
                batch2segs[batch_id] = []
            batch2segs[batch_id].append(seg_id)
        seg2details = {}
        for (batch_id, segs) in batch2segs.items():
            pure_simple_corpus_fn = "simple_corpus_batch_%s.h5" % (batch_id)
            simple_corpus_fn = os.path.join(simple_corpus_root_path, pure_simple_corpus_fn)
            with h5py.File(simple_corpus_fn, 'r') as f:
                chapter_infos_loaded = f['chapter_infos'][:]
                segment_ids_loaded = f['segment_ids'][:]
            for seg_id in segs:
                seg_pos = self._seg2batch_poses_maps[seg_id]
                seg2details[seg_id] = chapter_infos_loaded[seg_pos][0].decode("utf-8")
        rlt = []
        for (seg_id, now_score) in search_rlt:
            chapter_info = seg2details[seg_id]
            rlt.append((seg_id, now_score, chapter_info))
        return rlt        

    def load_id_maps(self, batch_tensor_root_path, segment2batch_fn):
        token2seg_id_maps = np.empty((0, 1), dtype=np.int32)
        file_paths = tools.walk_through_files(batch_tensor_root_path)
        for batch_id in range(0, len(file_paths)):
            pure_file_name = "segment_batch_%s.h5" % (batch_id)
            file_name = os.path.join(batch_tensor_root_path, pure_file_name)
            with h5py.File(file_name, 'r') as f:
                segment_ids_loaded = f['segment_ids'][:]
                token_ids_loaded = f['token_ids'][:]
            #此处会进行token2seg的拼接
            token2seg_id_maps = np.vstack((token2seg_id_maps, segment_ids_loaded))
        with h5py.File(segment2batch_fn, 'r') as f:
            seg2batch_ids_loaded = f['batch_ids'][:]
            seg2batch_poses_loaded = f['batch_poses'][:]
        return (token2seg_id_maps, seg2batch_ids_loaded, seg2batch_poses_loaded)
 
    def search_init(self):
        self._index = faiss.read_index(self._faiss_index_fn)
        (token2seg_id_maps, seg2batch_id_maps, seg2batch_poses_maps) = self.load_id_maps(self._batch_tensor_root_path, self._segment2batch_fn)
        self._token2seg_id_maps = token2seg_id_maps
        self._seg2batch_id_maps = seg2batch_id_maps
        self._seg2batch_poses_maps = seg2batch_poses_maps

    #main search index function
    def search(self, q):
        q_vector = np.array(self._ckpt.queryFromText([q])[0].tolist())
        q_info = (q, q_vector)
        search_rlt = self.search_q_faiss(q_info, self._index, self._token2seg_id_maps, self._seg2batch_id_maps, self._batch_tensor_root_path, self._split_batch_tensor_root_path, self._simple_corpus_root_path)
        print('len(serch_rlt)')
        print(len(search_rlt))
        texted_search_rlt = self.add_text_infos(search_rlt[:2], self._seg2batch_id_maps, self._simple_corpus_root_path)
        return texted_search_rlt
    #end search index functions 
