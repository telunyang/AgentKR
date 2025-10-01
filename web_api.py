from time import time
from pprint import pprint
from sentence_transformers import CrossEncoder
from flask import Flask, jsonify, request
from flask_ipfilter import IPFilter, Whitelist


'''
Flask Web API
'''
# 建立 Flask 物件
app = Flask(__name__)

# 設定白名單
ip_filter = IPFilter(app, ruleset=Whitelist())
ip_filter.ruleset.permit("127.0.0.1")

# Re-ranker
cross_encoder = CrossEncoder(
    'BAAI/bge-reranker-v2-m3',
    device='cpu', # 'cpu', # 'cuda:0', 
    trust_remote_code=True
)

# 重新排序文字內容
@app.route("/rerank", methods = ["POST"])
def rerank():
    # 取得請求資料
    q = request.json['q']
    li_sentences = request.json['li_sentences']
    li_urls = request.json['li_urls']

    # 重新排序
    t1 = time()
    print('重新排序中...')
    ranks = cross_encoder.rank(q, li_sentences, return_documents=True)
    for index, obj in enumerate(ranks):
        ranks[index]['score'] = float(obj['score'])
        ranks[index]['url'] = li_urls[obj['corpus_id']]
    pprint(ranks)
    print(f"重新排序花費時間：{time() - t1:.2f}")

    # 回傳結果
    return jsonify({"ranks": ranks})

# 主程式區域
if __name__ == '__main__':
    app.debug = False
    app.json.ensure_ascii = False
    app.run(
        host='127.0.0.1', # 0.0.0.0 
        port=5004,
        threaded=True
    )

