import codecs
import json
import os
import re 

#去除非字母的其他字符

def extract_answer(old_str):
    thoughts_pattern = r'<thoughts>(.*?)</thoughts>'
    answer_pattern = r'<answer>(.*?)</answer>'
    thoughts_contents = re.search(thoughts_pattern, old_str, re.DOTALL)
    answer_contents = re.search(answer_pattern, old_str, re.DOTALL)
    answer_str = answer_contents.group(1) if answer_contents else "" 
    return answer_str

def clean_answer(old_str):
    new_str = ""
    #找到第一个字母
    for now_char in old_str:
        if now_char.isalpha():
            new_str = now_char
            break
    return new_str

def static_one_line(line_index, line, error_f):
    line = line.strip()
    json_obj = json.loads(line)
    std_answer = clean_answer(json_obj["answer"])
    predict_info = json_obj["predict"]
    predict_inner_answer = clean_answer(extract_answer(predict_info["inner_answer"]))
    predict_answer = clean_answer(extract_answer(predict_info["answer"]))
    search_rlt = predict_info["search_rlt"]
    predict_inner_ok = 0
    if std_answer == predict_inner_answer:
        predict_inner_ok = 1
    predict_ok = 0
    if std_answer == predict_answer:
        predict_ok = 1
    static_item = {"index":line_index, "std_answer":std_answer, "predict_inner_answer":predict_inner_answer, "predict_answer":predict_answer,
            "predict_inner_ok":predict_inner_ok, "predict_ok":predict_ok, "search_rlt_len":len(search_rlt)}
    if predict_ok == 0:
        out_line = json.dumps(json_obj, indent=2, ensure_ascii=False)
        error_f.write(out_line+"\n")
    return static_item

def compute_one_static_info(static_items):
    static_info = {"total_count": len(static_items), "predict_inner_ok":0, "predict_ok":0, "ok2error":0, "error2ok":0, "both_ok":0, "both_error":0, "empty_search_rlt_count":0}
    for now_item in static_items:
        static_info["predict_inner_ok"] += now_item["predict_inner_ok"]
        static_info["predict_ok"] += now_item["predict_ok"]
        if now_item["search_rlt_len"] <= 0:
            static_info["empty_search_rlt_count"] += 1
        if now_item["predict_inner_ok"] == 1:
            if now_item["predict_ok"] == 1:
                static_info["both_ok"] += 1
            else:
                static_info["ok2error"] += 1
        else:
            if now_item["predict_ok"] == 1:
                static_info["error2ok"] += 1
            else:
                static_info["both_error"] += 1

    static_info["predict_inner_ok_ratio"] = (1.0*static_info["predict_inner_ok"]) / static_info["total_count"]
    static_info["predict_ok_ratio"] = (1.0*static_info["predict_ok"]) / static_info["total_count"]
    return static_info

def compute_global_static_info(static_infos):
    global_static_infos = {"package_count":0, "total_count":0, "predict_inner_ok":0, "predict_inner_ok_sum_ratio":0.0, "predict_ok":0, "predict_ok_sum_ratio":0.0}
    for now_static_info in static_infos:
        global_static_infos["package_count"] += 1
        global_static_infos["total_count"] += now_static_info["total_count"]
        global_static_infos["predict_inner_ok"] += now_static_info["predict_inner_ok"]
        global_static_infos["predict_inner_ok_sum_ratio"] += now_static_info["predict_inner_ok_ratio"]
        global_static_infos["predict_ok"] += now_static_info["predict_ok"]
        global_static_infos["predict_ok_sum_ratio"] += now_static_info["predict_ok_ratio"]
    global_static_infos["pre_predict_inner_ok_ratio"] = global_static_infos["predict_inner_ok_sum_ratio"] / global_static_infos["package_count"]
    global_static_infos["pre_predict_ok_ratio"] = global_static_infos["predict_ok_sum_ratio"] / global_static_infos["package_count"]
    global_static_infos["post_predict_inner_ok_ratio"] = global_static_infos["predict_inner_ok"] / global_static_infos["total_count"]
    global_static_infos["post_predict_ok_ratio"] = global_static_infos["predict_ok"] / global_static_infos["total_count"]
    return global_static_infos

def run():
    out_f = codecs.open("./static_rlt.json", "w", "utf-8")
    static_infos = []
    #for i in range(1,6):
    static_items = []
    now_fn = "./test_rlt/questions_rlt.json"
    now_f = codecs.open(now_fn, "r", "utf-8")
    now_error_fn = "./test_rlt/questions_rlt_error.jsonl"
    now_error_f = codecs.open(now_error_fn, "w", "utf-8") 
    line_index = -1
    for line in now_f:
        line_index += 1
        static_item = static_one_line(line_index, line, now_error_f)
        static_items.append(static_item)
    now_static_info = compute_one_static_info(static_items)
    out_str = json.dumps(now_static_info, ensure_ascii=False)
    out_f.write(out_str + "\n")
    static_infos.append(now_static_info)

    global_static_info = compute_global_static_info(static_infos)
    out_str = json.dumps(global_static_info, ensure_ascii=False)
    out_f.write(out_str+"\n")

if __name__ == "__main__":
    run()
