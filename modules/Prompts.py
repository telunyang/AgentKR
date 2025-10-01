import re

# 取得 router 的系統提示詞
def get_router_sys_prompt():
    return '''\
使用者提出的問題，名稱是 user_query。
不要隨意修改 user_query 的內容。
不要直接回答 user_query 的內容，只要進行 function calling (tool) 就可以了。

===================================

以下是區分單跳與多跳問答的基礎：
1. 單跳問答 (Single-hop QA)
    只需要 一個直接的資訊來源 就能得出答案。
    問題和答案之間的關聯 明確且直接，通常可以從單一段落或資料點中取得答案。
    例子：
    問：「太陽系中最大的行星是什麼？」
    答：「木星。」（這只需要查詢行星的基本資料）
    問：「愛因斯坦提出了哪個著名的理論？」
    答：「相對論。」

2. 多跳問答 (Multi-hop QA)
    需要 多個資訊來源或步驟 才能推理出答案。
    問題的答案 不能直接從單一資料點獲取，而是需要綜合多個資訊來推論。
    可能涉及：
        - 資訊合併：例如將兩個或多個句子的資訊組合以得出答案。
        - 中間推理：需要回答一個子問題，再用這個答案來推導最終答案。
    例子：
        問：「發現萬有引力定律的科學家在哪一年出生？」
        這需要兩步推理：
            先找到「萬有引力定律的科學家是牛頓」。
            再查找「牛頓的出生年份是 1643 年」。
            最終答：「1643 年」。
        問：「喬治·華盛頓的母親去世時，美國總統是誰？」
        這需要兩步推理：
            查「喬治·華盛頓的母親何時去世」（1789 年）。
            查「1789 年時的美國總統」（華盛頓本人）。
            最終答：「喬治·華盛頓。」
        問：「劉備的結拜兄弟，參與過哪些戰役？」
        這需要兩步推理：
            查「劉備的結拜兄弟有哪些。」。
            查「結拜兄弟們各自參與過哪些戰役？」（結拜兄弟為關羽、張飛）。
            最終答：「關羽參與汜水關之戰、白馬延津之戰、過五關斬六將、赤壁之戰、襄樊之戰等；張飛參與虎牢關之戰、長坂坡之戰、益州之戰等。」

直接使用 tool，進行 function calling: get_search_results()：
1. 如果是 多跳 (multi-hop) 的問題，請將這個問題分解成多個簡單直白的 sub questions，連同 user_intent、query、hop_type、user_query、num_results、model_name 進行傳送。
2. 如果是 單跳 (single-hop) 的問題，連同 user_intent、query、hop_type、user_query、num_results、model_name 進行傳送。
'''


# 取得摘要的提示詞
def get_summarization_prompt(q, context) -> str:
    # 取得摘要
    user_prompt = f'''\
QUESTION:
{q}

========================================

CONTEXT:
{context}

========================================

RULES:
1. Pick 5 paragraphs ONLY from CONTEXT.
2. Paragraphs MUST be able to answer the QUESTION, or provide answers which can be used to answer the QUESTION.
3. Summarize ONLY the selected paragraphs.

========================================

RESPONSED FORMAT:
Return ONLY a valid JSON array of three strings.  
Example: ["SUMMARY 1", "SUMMARY 2", "SUMMARY 3", "SUMMARY 4", "SUMMARY 5"]

At the end, verify that your output is valid JSON and matches the example format exactly.

========================================

OUTPUT:'''
    return user_prompt


# 取得建立知識三元組的提示詞
def get_triplets_prompt(d_proper_knowledge):
    li_text = []
    for references in d_proper_knowledge['references']:
        li_text.append(references['text'])

    message = "\n".join(li_text)

    context = f'''Triples describe relationships between entities and consist of three elements. In computer science, triples are commonly used to represent data in relational databases. A typical triple contains three components: Subject, Predicate, and Object. They form the basic elements of a Knowledge Graph.

=====================================================================================

Format:
triples = [
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ['Subject', 'Predicate', 'Object'],
    ...
]
...and so on.

Instructions:
1. If multiple Subjects refer to the same entity, unify them into a single Subject to ensure correct relationships and reduce unnecessary duplication.
2. If a single sentence or text paragraph contains a Subject associated with more than one Predicate and Object, list all such relationships.
3. Do not answer any questions or provide explanations — only extract and list the triples.

=====================================================================================

Extract all triples from the following text:
{message}

At the end, verify that your output is valid LIST OF TRIPLES and matches the example format exactly.

Output:'''
    return context


# 取得 LLM 重新排序的提示詞
def get_rerank_results_by_llm(ranks, q) -> str:
    prompt = f'''\
Re-rank the following search results based on their relevance to the user's query.

Query:
{q}

=========================

Search results (list of dictionaries):
{ranks}

=========================

The list is reordered based on relevance to the user's query by yourself, but the originally provided scores are retained.
The list will be used to calculate cumulative gain (CG), discounted cumulative gain (DCG), and normalized discounted cumulative gain (NDCG).
Keep the original relevance scores, URLs, and summaries.
Do not include any explanations, comments, or extra text.
Do not include any characters/strings that would break JSON format.
Only output the final result as a valid JSON array.
You MUST follow the example format below exactly.
Avoid this issue: "json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes".
You MUST ensure the output is valid JSON.

=========================

FORMAT:
[
    {{
        "corpus_id": original_id,
        "score": original_score,
        "text": "original_summary",
        "url": "original_url"
    }},
    ...
]

Ensure the output is valid JSON.

Output: '''
    return prompt




# 將檢索結果整合到 user prompt 當中
def get_rag_prompt(d: dict):
    knowledge = ""
    for d_ in d['proper_knowledge']:
        # 整理 triplets 之間的關係
        str_triplets = ""
        if 'triplets' in d_:
            triplets = d_['triplets']
            for t in triplets:
                # 將每個三元組轉換為字串格式
                if len(t) < 3:
                    continue
                str_triplets += f'`Subject: "{t[0]}", Predicate: "{t[1]}", Object: "{t[2]}"`\n'

        # 整理 reference
        for reference in d_['references']:
            # 整理 knowledge 字串
            knowledge += f'''\
----------------------------------
Sub-question: {d_['q']}
Reference: {reference['text']}
Relevance Score: {reference['score']}
Source: {reference['url']}
'''

        knowledge += f'''\
----------------------------------
Triplets (Knowledge Graph): 
{str_triplets}
'''

    # 整理 user prompt
    context = f'''\
Original Question:
{d['user_query']}

=========================

Please refer to the following references to answer the question:
{knowledge}

=========================

Notes:
1. The answer must begin with the correct option from the question choices.
2. If the question is multi-choice, choose the best answer (A), (B), (C) or (D) based on your understanding of the question.
3. If the reference knowledges are unavailable or unreasonable, you need to answer with your own understanding.
4. Answer the question in English.

=========================

Answer:'''
    return context