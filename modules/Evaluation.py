'''
# 評估參考程式: Evaluation script for CMRC 2018
Source code: https://github.com/ymcui/cmrc2018/blob/master/baseline/cmrc2018_evaluate.py

# 大語言模型中常用的評估指標
https://cloud.tencent.com/developer/article/2314960
'''

import re, sys
import nltk

import opencc
converter = opencc.OpenCC('s2tw.json')

'''
Multi-Choices (選擇題) - mc
'''
# 取得選擇題答案
# source: https://github.com/mtkresearch/MR-Models/blob/main/TC-Eval/evaluate.py#L81
def extract_choice(response):
    # 如果生成文字中，有太多符合 (A) - (D) 的格式化文字，則代表回答失敗
    # if len(re.findall(r"\(([A-Ea-e])\)", response)) > 1:
    #     return ''

    # 自訂 pattern
    # 例如: 台灣最長的人工隧道是 [D]雪山隧道。
    # 例如: The correct answer is (A)釋證嚴
    # 例如: D. 淡水河
    # 例如: ( A )
    # 例如: 答案是B。捷安特
    patterns = [
        r"\b([A-Ea-e1-5])\b",
        r"\(\s*([A-Ea-e1-5])\s*\)\s*",
        r"correct answer is\s*[(|【|（|［|\[|<|＜]([A-Ea-e1-5])[)|】|）|］|\]|>|＞)]\s*",
        r"(?<=答案)\s*.+\s*\(([A-Ea-e1-5])\)",
        r"答案是:?\s*\(?([A-Ea-e1-5])\)?\s*",
        r"\(([A-Ea-e1-5])\).+正確",
        r"\[\/?INST\]\s*\(([A-Ea-e1-5])\)",
        r"ASSISTAs?NT:\s*\(([A-Ea-e1-5])\)",
        r"\s?[【|（|［|\[|<|＜]([A-Ea-e1-5])[】|）|］|\]|>|＞)]\s?",
        r"(?<=\w|\s)\(([A-Ea-e1-5])\)(?=\w|\s)",
        r"\(([A-Ea-e1-5])\)\s*[\u4E00-\u9FFF]",
        r"[\u4E00-\u9FFF]\s*\(([A-Ea-e1-5])\)",
        r"(?<=選項)\s*\(([A-Ea-e1-5])\)",
        r"([A-Ea-e1-5])\.\s*[\u4E00-\u9FFF]",
        r"([A-Ea-e1-5])\.\s*\w",
        r"([A-Ea-e1-5])\.?\s*[「|【|『|《|«]",
        r"\(\s*([A-Ea-e1-5])\s*\)",
        r"\(?([A-Ea-e1-5])\)\s*",
    ]

    # 去除不必要的生成內容(例如換行符號)
    # response = re.sub(r"\n", "", response)

    # 尋找符合 pattern 的生成結果 (整個句子只能出現一個 ABCD 字母，超過一個就不算)
    for regex in patterns:
        list_ = re.findall(regex, response)
        if len(list_) > 0:
            return list_[0].upper()
    
    return ''

# 評估
def evaluate_mc(ground_truth_file, list_predicted_results, task_name):
    list_choices = "ABCDE"
    correct = 0
    wrong = 0
    total_count = 0
    accuracy = 0
    list_wrong_data = []
    for index, instance in enumerate(ground_truth_file.items()):
        # 累計計算的問題數
        total_count += 1

        # 取得題目編號與問答的參考資料
        id = instance[0]
        d = instance[1]

        # 取得參考資料
        query_id = id.strip()
        answers 	= d['choices'][d['answer']]
        answer_index = d['answer']
       
        # 假設可替選的正確答案有很多人，則一個一個比對
        if type(answers) != list:
              answers = [answers]
       
        # 取得生成的結果
        prediction = list_predicted_results[index]['generated_text']

        # 如果生成結果符合 ABCD 格式，則進行答案比對
        choice = extract_choice(prediction)
        if  list_choices[answer_index].upper() == choice.upper():
            correct += 1
        else:
            list_wrong_data.append({
                "task_name": task_name,
                "id": query_id,
                "source": d,
                "inferred_results": list_predicted_results[index],
                "generated_text": prediction,
            })
            wrong += 1

    # 計算 accuracy
    accuracy = round((correct / total_count) * 100.0, 2)

    return correct, wrong, total_count, accuracy, list_wrong_data


'''
Exact Matching
'''
# 若是沒有生成選項ABCD，而是選項的文字答案，則使用前綴詞比對 
# Source: https://github.com/mtkresearch/MR-Models/blob/main/TC-Eval/evaluate.py#L16C1-L20C61
def prefix_exact_match(answers, prediction):
    if not prediction: 
        return 0
    
    # 將文字轉換成臺灣用字/語
    prediction = converter.convert(prediction)

    # 取得每一個答案，進行比對
    pem = 0
    for ans in answers:
        # 例如: "一見如故啊".startswith("一見") => 「一見如故啊」是以「一見」開頭
        ans = converter.convert(ans)
        if prediction.strip().startswith(ans.strip()):
            pem = 1
            break
    return pem


'''
Longest Common (Sub) String - lcs
'''
nltk.download('punkt')

# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = in_str.lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out

# 移除標點符號
def remove_punctuation(in_str):
    in_str = in_str.lower().strip()
    sp_char = ['-',':','_','*','^','/','\\','~','`','+','=',
                '，','。','：','？','！','“','”','；','’','《','》','……','·','、',
                '「','」','（','）','－','～','『','』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)

# 找出最長共同子字串
def find_lcs(s1, s2):
    '''
    s1 = "今天天氣真好，適合出門散步和呼吸新鮮空氣"
    s2 = "今天天氣不錯，適合去海邊散步和享受陽光"
    使用 find_lcs 函式找出它們之間的最長共同子字串。
    結果是最長共同子字串為 "今天天氣"，其長度為 4（以中文字計算）。
    這表示 "今天天氣" 是這兩個句子中連續相同的最長部分。
    '''
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j]+1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p-mmax:p], mmax

# 計算 F1 score (多個答案，以最高的比對分數回傳)
def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision 	= 1.0 * lcs_len/len(prediction_segs)
        recall 		= 1.0 * lcs_len/len(ans_segs)
        f1 			= (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)

# 計算 extract match 的分數，ans 與 prediction 完全一樣，就將 em 設定為 1
def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em

# 評估
def evaluate_lcs(ground_truth_file, list_predicted_results, task_name):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    threshold = 0.4
    list_wrong_data = []
    for index, instance in enumerate(ground_truth_file.items()):
        id = instance[0]
        d = instance[1]

        total_count += 1
        query_id    = id.strip()
        query_text  = d['question'].strip()
        answers 	= d['choices'][d['answer']]

        if type(answers) != list:
            answers = [answers]

        flag_id_in = False
        for _ in list_predicted_results:
            if query_id == _['id']:
                flag_id_in = True
                break

        if flag_id_in == False:
            sys.stderr.write('Unanswered question: {}\n'.format(query_id))
            skip_count += 1
            continue

        prediction 	= list_predicted_results[index]['generated_text']
        
        # 計算 f1-score 和 exact matching score
        f1_score = calc_f1_score(answers, prediction)
        em_score = calc_em_score(answers, prediction)
        f1 += f1_score
        em += em_score

        # 如果當前的 average (f1-score + em_score) 沒有大於
        if (f1_score + em_score) * 0.5 < threshold:
            list_wrong_data.append({
                "task_name": task_name,
                "id": query_id,
                "source": d,
                "generated_text": prediction,
            })


    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count, list_wrong_data

