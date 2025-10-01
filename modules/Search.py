from googlesearch import search

from playwright.sync_api import sync_playwright
from time import sleep, time
import json, os, re, sys, ast
from typing import Annotated
from pprint import pprint
from urllib.parse import unquote
from bs4 import BeautifulSoup as bs
import requests as req
from modules.Model import generate
from modules.Prompts import get_triplets_prompt, get_rerank_results_by_llm, get_summarization_prompt
from modules.Func import rrf, wm, rrf_wm, ndcg_at_k

# 整理搜尋結果
def get_search_results(
        user_intent: Annotated[str, "使用者意圖"], 
        hop_type: Annotated[str, "FORMAT: single-hop or multi-hop"], 
        user_query: Annotated[str, "使用者提出的原始問題，千萬不要修改 user_query 的內容。"], 
        query: Annotated[str | list[str], "取得 single-hop 的 question，或是 multi-hop 的 sub-questions。"],
        num_results: Annotated[int, "搜尋結果數量"],
        model_name: Annotated[str, "模型名稱"]
    ) -> str:

    # 回傳資料初始化
    json_string = ''

    # 計算程式執行時間
    t1 = time()

    try:
        # 取得使用者的意圖、hop_type、使用者問題、query
        d_info = {
            "user_intent": user_intent,
            "hop_type": hop_type,
            "user_query": user_query,
            "query": query,
            "model_name": model_name,
            "search_results": [],
            "proper_knowledge": [],
            "reranked_summaries_by_reranker": [],
            "reranked_summaries_by_local_llm": [],
            "reranked_summaries_by_gemini_llm": []
        }

        '''
        判斷 query 的型別:
        - 如果是 list，就代表是 multi-hop 的 sub-questions
        - 如果是 str，就代表是 single-hop 的 question
        '''
        if isinstance(query, list): 
            sentence = ""
            for index, q in enumerate(query):
                last_q = q + ' ' 
                if sentence != "":
                    q = f"{q}     {last_q}{sentence}"
                    sentence = ""

                # 搜尋
                li_context = run_web_search(q, num_results, "us", model_name)

                # 整理摘要之後的文章段落，整理在 list 當中，方便 re-rank
                li_sentences = []
                li_urls = []
                for d in li_context:
                    # 取得摘要
                    summaries = d['summaries']
                    for summary in summaries:
                        li_sentences.append(summary.strip())
                        li_urls.append(d['url'])
                
                # Re-ranking，取得與 query 最相關的 document 或 sentence
                # ranks = cross_encoder.rank(q, li_sentences, return_documents=True)
                ranks = req.post('http://127.0.0.1:5004/rerank', json={"q": q, "li_sentences": li_sentences, "li_urls": li_urls}).json()['ranks']

                 # 將 re-ranker 的排序結果，讓 LLM 重新進行排序
                count = 5
                str_general_ranks_by_llm = ""
                str_ideal_ranks_by_llm = ""
                li_general_ranks_by_llm = []
                li_ideal_ranks_by_llm = []
                while count > 0:
                    try:
                        # 整理 re-ranker 的排序結果，讓 LLM 重新進行排序的 prompt
                        str_rerank_by_llm = get_rerank_results_by_llm(ranks, q)

                        # 由 LLM 重新進行排序
                        str_general_ranks_by_llm = generate(str_rerank_by_llm, model_name)
                        str_general_ranks_by_llm = re.sub(r"\n|```json|```", "", str_general_ranks_by_llm)
                        li_general_ranks_by_llm = json.loads(str_general_ranks_by_llm)

                        # 由雲端服務的 LLM 重新進行排序，嘗試列出理想的排名，用來計算 NDCG
                        str_ideal_ranks_by_llm = generate(str_rerank_by_llm, "gemini-2.5-flash-lite")
                        str_ideal_ranks_by_llm = re.sub(r"\n|```json|```", "", str_ideal_ranks_by_llm)
                        li_ideal_ranks_by_llm = json.loads(str_ideal_ranks_by_llm)

                        break
                    except Exception as e:
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        print(f"Error in re-ranking by LLM: {e}")
                        count -= 1
                        sleep(4)


                # 依據先前自訂的 query 結構特性，取得主要的 query 
                q = q.split("     ")[0]

                # 記錄搜尋結果
                d_info['reranked_summaries_by_reranker'].append(ranks)
                d_info['reranked_summaries_by_local_llm'].append(li_general_ranks_by_llm)
                d_info['reranked_summaries_by_gemini_llm'].append(li_ideal_ranks_by_llm)
                d_info["search_results"].append(li_context)
                d_proper_knowledge = {
                    "q": last_q.strip(),
                    "references": ranks[:5],
                }

                # 用 re-ranker 排名第一的句子，作為下一個 sub-question 的背景知識
                sentence = ranks[0]['text'] # 也可以改成 li_general_ranks_by_llm[0]['text']
                
                # 繪製知識圖譜
                kg_prompt = get_triplets_prompt(d_proper_knowledge)

                # 你的三元組資料
                str_triplets = generate(kg_prompt, model_name)
                triplets = re.search(r"\[(.|\s)+\]", str_triplets)[0]
                triplets = eval(triplets)

                # 將回傳資料加入三元組的資訊
                d_proper_knowledge['triplets'] = triplets
                d_info["proper_knowledge"].append(d_proper_knowledge)

        elif isinstance(query, str):
            # 搜尋
            li_context = run_web_search(user_query, num_results, "us", model_name)

            # 整理摘要之後的文章段落，整理在 list 當中，方便 re-rank
            li_sentences = []
            li_urls = []
            for d in li_context:
                # 取得摘要
                summaries = d['summaries']
                for summary in summaries:
                    li_sentences.append(summary.strip())
                    li_urls.append(d['url'])

            # Re-ranking，取得與 query 最相關的 document 或 sentence
            # ranks = cross_encoder.rank(query, li_sentences, return_documents=True)
            ranks = req.post('http://127.0.0.1:5004/rerank', json={"q": user_query, "li_sentences": li_sentences, "li_urls": li_urls}).json()['ranks']

            # 將 re-ranker 的排序結果，讓 LLM 重新進行排序
            count = 5
            str_general_ranks_by_llm = ""
            str_ideal_ranks_by_llm = ""
            li_general_ranks_by_llm = []
            li_ideal_ranks_by_llm = []
            while count > 0:
                try:
                    # 整理 re-ranker 的排序結果，讓 LLM 重新進行排序的 prompt
                    str_rerank_by_llm = get_rerank_results_by_llm(ranks, user_query)

                    # 由 LLM 重新進行排序
                    str_general_ranks_by_llm = generate(str_rerank_by_llm, model_name)
                    str_general_ranks_by_llm = re.sub(r"\n|```json|```", "", str_general_ranks_by_llm)
                    li_general_ranks_by_llm = json.loads(str_general_ranks_by_llm)

                    # 由雲端服務的 LLM 重新進行排序，嘗試列出理想的排名，用來計算 NDCG
                    str_ideal_ranks_by_llm = generate(str_rerank_by_llm, "gemini-2.5-flash-lite")
                    str_ideal_ranks_by_llm = re.sub(r"\n|```json|```", "", str_ideal_ranks_by_llm)
                    li_ideal_ranks_by_llm = json.loads(str_ideal_ranks_by_llm)

                    break
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print(f"Error from the local LLM as another re-ranker: {e}")
                    count -= 1
                    sleep(4)

            # 整理回傳所需要的資料
            d_info['reranked_summaries_by_reranker'].append(ranks)
            d_info['reranked_summaries_by_local_llm'].append(li_general_ranks_by_llm)
            d_info['reranked_summaries_by_gemini_llm'].append(li_ideal_ranks_by_llm)
            d_info["search_results"].append(li_context)
            d_proper_knowledge = {
                "q": user_query,
                "references": ranks,
            }
            
            # 繪製知識圖譜
            kg_prompt = get_triplets_prompt(d_proper_knowledge)

            # 你的三元組資料
            str_triplets = generate(kg_prompt, model_name)
            triplets = re.search(r"\[(.|\s)+\]", str_triplets)[0]
            triplets = eval(triplets)

            # 將回傳資料加入三元組的資訊
            d_proper_knowledge['triplets'] = triplets
            d_info["proper_knowledge"].append(d_proper_knowledge)

        # 將整理結果的資料轉換成 JSON 格式
        json_string = json.dumps(d_info, ensure_ascii=False, indent=None)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

    # 計算程式執行時間
    t2 = time()

    print(f"程式執行時間: {t2 - t1} 秒")

    return json_string



# 搜尋引擎
def run_web_search(q: str, num_results: int, lang: str, model_name: str) -> list:
    # 放置檢索結果
    li_context = []

    # 隨機選擇搜尋引擎 (以免額度被用完)
    urls = search(q, num_results=num_results, sleep_interval=10, unique=True, lang=lang)

    # 使用 playwright 來取得網頁內容
    with sync_playwright() as playwright:
        # 設定瀏覽器
        pw = playwright.chromium # "chromium" or "firefox" or "webkit".
        browser = pw.launch(headless=True, args=["--start-maximized"])
        context = browser.new_context(accept_downloads=False, no_viewport=True)
        page = context.new_page()
    
        # 走訪每個連結
        for url in urls:
            try:
                # 前往頁面
                page.goto(url, timeout=10*1000)
                
                print("=" * 50)
                print("查詢的問題:", q)
                print(f"正在取得 {unquote(url)} 的內容...")

                if url.lower().endswith(".pdf") or 'arxiv' in url.lower():
                    # m = hashlib.md5()
                    # m.update(url.encode('utf-8'))
                    # file_name = m.hexdigest()
                    # file_path = f"./tmp/{file_name}.pdf"
                    # if not os.path.exists(file_path):
                    #     print('正在下載 PDF 檔案...')
                    #     wget.download(url, file_path)
                    #     print('下載完成！')
                    # else:
                    #     print('PDF 檔案已存在，跳過下載。')
                    # try:
                    #     doc = pymupdf.open(file_path)
                    #     context = ''
                    #     for p in doc:
                    #         context += p.get_text()
                    #     print()
                    # except Exception as e:
                    #     raise Exception(f"無法讀取 PDF 檔案 {url} -> {file_path}: {e}")

                    print(f"略過 PDF 檔案: {url}")
                    continue
                else:
                    # 取得 HTML 元素
                    html = page.content()

                    # 使用 BeautifulSoup 解析 HTML
                    soup = bs(html, "lxml")

                    # 取得網頁內文
                    context = soup.get_text(strip=True)
                    # context = html
                    context = re.sub(r"\s+", " ", context)

                # 依據先前自訂的 query 結構特性，取得主要的 query 
                q = q.split("     ")[0]

                # 取得摘要的提示詞
                user_prompt = get_summarization_prompt(q, context)

                # 生成內容
                generated_text = generate(user_prompt, model_name)
                generated_text = eval(re.search(r"\[\s*(?:(?:\"[^\"]+\"\s*,?\s*)+)\]", generated_text.replace("\n", ""))[0])

                # 整理回傳所需要的資料
                d = {
                    "q": q,
                    "url": unquote(url),
                    "summaries": generated_text
                }
                li_context.append(d)
                print("已取得 knowledge:")
                pprint(d)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print('發生錯誤:', exc_type, fname, exc_tb.tb_lineno)
                continue

        # 關閉瀏覽器
        browser.close()

    return li_context


