import os, json, sys
from datetime import datetime
from autogen import ConversableAgent, LLMConfig
from autogen import register_function

from modules.Prompts import get_router_sys_prompt, get_rag_prompt
from modules.Search import get_search_results
from modules.Model import get_llm_config, generate
from modules.Func import termination_msg, get_kg_example_html, rrf, wm, rrf_wm, ndcg_at_k

'''
模型設定
'''
# 模型組態設定
os.environ["OAI_CONFIG_LIST"] = get_llm_config()
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

'''
模型設定
'''
# 模型列表
li_models = [
    'mistral-small3.1:24b',
    'qwen3:32b',
    'llama3.3:70b',
    'llama4:16x17b',
    'gpt-oss:20b',
    'gpt-oss:120b',
]

# 選擇模型
seed = 42
model_name = li_models[4]
llm_config = LLMConfig.from_json(
    path="OAI_CONFIG_LIST",
    seed=seed,
).where(model=model_name)



'''
建立 Agents
'''
# 建立解析使用者需求的 agent
with llm_config:
    # 這個 agent 負責解析使用者的需求，並決定要使用哪一個 agent 來執行
    router_agent = ConversableAgent(
        name="router_agent",
        system_message=get_router_sys_prompt(),
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=termination_msg,
        description="如果 function calling (tool) 有正確回傳，請說出 `DONE!`; 如果沒有完成，請要求 executor_agent 再一次使用 function calling (tool) 來取得更多資訊。",
    )

    # 用來執行 function calling 的 agent
    executor_agent = ConversableAgent(
        name="executor_agent",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=termination_msg,
    )

    

'''
Function Calling 設定
'''
# 取得搜尋結果
register_function(
    get_search_results,
    caller=router_agent,
    executor=executor_agent,
    name="get_search_results",
    description=f'''取得 user_intent、hop_type、user_query、query、model_name、search_results 和 proper_knowledge，
num_results 預設為 15，
model_name 為 `{model_name}`，
如果 `Response from calling tool` 成功回傳，請說出 `DONE!`。''',
)


if __name__ == "__main__":
    q = "臺灣的國寶魚叫做什麼?"
    q_rpl = q.replace("\n", " ")

    while True:
        try:
            # Agents 互動
            chat_result = router_agent.initiate_chat(
                recipient=executor_agent,
                message=q_rpl,
                max_turns=4,
            )

            d = None
            for obj in chat_result.chat_history:
                if obj["role"] == "tool" and obj['name'] == 'executor_agent' and obj["tool_responses"] != None:
                    # 取得 function calling 的結果
                    d = eval(obj["tool_responses"][0]["content"])

                    # 結束條件
                    break

            if d is None:
                print("No function calling result found.")
                continue

            # 取得自訂的 user prompt
            d['user_query'] = q
            
            user_prompt = get_rag_prompt(d)

            # 對 Agentic Advanced RAG，進行生成式的回答
            generated_text = generate(user_prompt)

            # 取得重新排序的結果 (使 re-ranker 與 local LLM)
            ranks_reranker = d['reranked_summaries_by_reranker'][0]
            ranks_local = d['reranked_summaries_by_local_llm'][0]

            # 三種融合
            fused_rrf = rrf(ranks_reranker, ranks_local, k=60, top_k=None)
            fused_wm = wm(ranks_reranker, ranks_local, weight1=0.2, weight2=0.8, top_k=None)
            fused_hybrid = rrf_wm(ranks_reranker, ranks_local, k=60, weight1=0.5, weight2=0.5, alpha=0.5, top_k=None)

            # 取得理想的排名 (雲端服務的 LLM)
            gemini_ranks = d['reranked_summaries_by_gemini_llm'][0]

            # relevance dict: 讓第一名 relevance 最大，最後一名 relevance 最小
            relevant_dict = {r['corpus_id']: len(gemini_ranks) - idx for idx, r in enumerate(gemini_ranks)}

            # 計算 NDCG: 對三種融合結果各自算出 NDCG@5
            ndcg_rrf = ndcg_at_k(fused_rrf, relevant_dict, k=5)
            ndcg_wm = ndcg_at_k(fused_wm, relevant_dict, k=5)
            ndcg_hybrid = ndcg_at_k(fused_hybrid, relevant_dict, k=5)

            # 整理結果
            d['fused_rrf'] = fused_rrf
            d['fused_wm'] = fused_wm
            d['fused_hybrid'] = fused_hybrid
            d['relevance_dict'] = relevant_dict
            d['ndcg_rrf'] = ndcg_rrf
            d['ndcg_wm'] = ndcg_wm
            d['ndcg_hybrid'] = ndcg_hybrid

            # 將資料寫入檔案
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            with open(f"./output/search_result_{ts}.json", "w", encoding="utf8") as f:
                f.write(json.dumps(d, ensure_ascii=False, indent=None))
            with open(f"./output/user_prompt_{ts}.txt", "w", encoding="utf8") as f:
                f.write(user_prompt + generated_text)
            with open(f"./output/kg_example_{ts}.html", "w", encoding="utf8") as f:
                li_triplets = []
                for pk in d['proper_knowledge']:
                    if 'triplets' in pk:
                        li_triplets.extend( pk['triplets'] )
                f.write(get_kg_example_html( str(li_triplets) ))

            break

        except Exception as e:
            print(f"Error: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)