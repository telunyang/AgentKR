from pprint import pprint
import re
import numpy as np
from typing import List, Dict, Tuple

'''
# Reciprocal Rank Fusion (RRF): 
https://blog.csdn.net/UbuntuTouch/article/details/131200354
https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
https://medium.com/@danushidk507/rag-vii-reranking-with-rrf-d8a13dba96de

# DNCG (Discounted Normalized Cumulative Gain):
https://ithelp.ithome.com.tw/articles/10299050
https://www.evidentlyai.com/ranking-metrics/ndcg-metric
'''

'''
Agent termination condition
'''
# 結束條件
def termination_msg(x):
    '''
    {
        'content': '', 
        'tool_responses': [{'tool_call_id': '7475', 'role': 'tool', 'content': ''}], 
        'role': 'tool', 
        'name': 'executor_agent'
    }
    '''
    # print("termination_msg:")
    # pprint(x)
    return 'DONE!' in x.get("content", "").strip()



'''
Make HTML for Knowledge Graph visualization
'''
# 建立 KG 範例的 HTML 字串
def get_kg_example_html(li_triplets) -> str:
    html = '''<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <title>Knowledge Graph 知識圖譜</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }
    svg {
      width: 100vw;
      height: 100vh;
      background: #fff;
    }
    .edge-label {
      font-size: 14px;
      fill: #333;
      pointer-events: none;
    }
    .node text {
      fill: #111;
    }
  </style>
</head>
<body>
<svg></svg>
<script>
// === Triplets ===
const triplets = ''' + str(li_triplets) + ''';

// === Graph Data Conversion ===
const nodesSet = new Set();
const links = triplets.map(([source, label, target]) => {
  nodesSet.add(source);
  nodesSet.add(target);
  return { source, target, label };
});
const nodes = Array.from(nodesSet).map(id => ({ id }));

// Map repeated edges
const linkGroups = {};
links.forEach(link => {
  const key = `${link.source}->${link.target}`;
  if (!linkGroups[key]) linkGroups[key] = [];
  linkGroups[key].push(link);
});
Object.values(linkGroups).forEach(group => {
  group.forEach((link, i) => {
    link.linkIndex = i;
    link.linkTotal = group.length;
  });
});

// === SVG Setup ===
const svg = d3.select("svg");
const width = window.innerWidth;
const height = window.innerHeight;
const zoomGroup = svg.append("g");

svg.call(d3.zoom().scaleExtent([0.2, 5]).on("zoom", e => {
  zoomGroup.attr("transform", e.transform);
}));

svg.append("defs").append("marker")
  .attr("id", "arrow")
  .attr("viewBox", "0 -8 16 16")
  .attr("refX", 34)
  .attr("refY", 0)
  .attr("markerWidth", 12)
  .attr("markerHeight", 12)
  .attr("orient", "auto")
  .append("path")
  .attr("d", "M0,-8L16,0L0,8")
  .attr("fill", "#999");

// === Dynamic Forces ===
const tripletCount = triplets.length;
const linkDistance = 160 + tripletCount * 6;
const chargeStrength = -300 - tripletCount * 4;

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(linkDistance))
  .force("charge", d3.forceManyBody().strength(chargeStrength))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collide", d3.forceCollide().radius(35).strength(1));

// === Draw Curved Edges ===
const link = zoomGroup.append("g")
  .selectAll("path")
  .data(links)
  .enter()
  .append("path")
  .attr("stroke", "#999")
  .attr("stroke-opacity", 0.6)
  .attr("stroke-width", 1.5)
  .attr("fill", "none")
  .attr("marker-end", "url(#arrow)");

// === Edge Labels ===
const edgeLabels = zoomGroup.append("g")
  .selectAll("text")
  .data(links)
  .enter()
  .append("text")
  .attr("class", "edge-label")
  .attr("text-anchor", "middle")
  .text(d => d.label);

// === Draw Nodes ===
const node = zoomGroup.append("g")
  .selectAll("g")
  .data(nodes)
  .enter()
  .append("g")
  .call(d3.drag()
    .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
    .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }));

node.append("circle")
  .attr("r", 24)
  .attr("fill", "#69b3a2")
  .attr("stroke", "#333")
  .attr("stroke-width", 1.5);

node.append("text")
  .attr("x", tripletCount < 10 ? 0 : 28)
  .attr("y", 5)
  .attr("text-anchor", tripletCount < 10 ? "middle" : "start")
  .attr("font-size", tripletCount < 10 ? "18px" : "14px")
  .text(d => d.id);

// === Curve Path Calculation ===
function computeCurvePath(d) {
  const x1 = d.source.x, y1 = d.source.y;
  const x2 = d.target.x, y2 = d.target.y;
  const dx = x2 - x1, dy = y2 - y1;
  const dr = Math.sqrt(dx * dx + dy * dy);
  const curveOffset = 50 * ((d.linkIndex % 2 === 0 ? 1 : -1) * Math.ceil((d.linkIndex + 1) / 2));
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const nx = -(y2 - y1), ny = x2 - x1;
  const norm = Math.sqrt(nx * nx + ny * ny);
  const cx = mx + (curveOffset * nx) / norm;
  const cy = my + (curveOffset * ny) / norm;
  return `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`;
}

// === Tick Update ===
simulation.on("tick", () => {
  link.attr("d", computeCurvePath);
  edgeLabels
    .attr("x", d => {
      const x1 = d.source.x, x2 = d.target.x;
      return (x1 + x2) / 2;
    })
    .attr("y", d => {
      const y1 = d.source.y, y2 = d.target.y;
      const offset = 20 + d.linkIndex * 6;
      return (y1 + y2) / 2 - offset;
    });
  node.attr("transform", d => `translate(${d.x},${d.y})`);
});

// === Auto Zoom Fit ===
simulation.on("end", () => {
  const coords = node.data().map(d => [d.x, d.y]);
  const xExtent = d3.extent(coords, d => d[0]);
  const yExtent = d3.extent(coords, d => d[1]);
  const dx = xExtent[1] - xExtent[0];
  const dy = yExtent[1] - yExtent[0];
  const cx = (xExtent[0] + xExtent[1]) / 2;
  const cy = (yExtent[0] + yExtent[1]) / 2;
  const scale = 0.85 / Math.max(dx / width, dy / height);
  const translate = [width / 2 - scale * cx, height / 2 - scale * cy];
  svg.transition().duration(500)
    .call(d3.zoom().transform, d3.zoomIdentity.translate(...translate).scale(scale));
});
</script>
</body>
</html>
'''
    return html



'''
Metric for ranking quality
'''
# 正規化分數
def normalize_scores(ranks):
    scores = [r['score'] for r in ranks]
    min_s, max_s = min(scores), max(scores)
    for r in ranks:
        r['score'] = (r['score'] - min_s) / (max_s - min_s + 1e-8)
    return ranks

# Reciprocal Rank Fusion (RRF)
def rrf(ranks1, ranks2, k=60, top_k=None):
    rank_pos1 = {r['corpus_id']: idx + 1 for idx, r in enumerate(ranks1)}
    rank_pos2 = {r['corpus_id']: idx + 1 for idx, r in enumerate(ranks2)}
    all_ids = set(rank_pos1.keys()) | set(rank_pos2.keys())
    fused = []
    for cid in all_ids:
        pos1 = rank_pos1.get(cid, float('inf'))
        pos2 = rank_pos2.get(cid, float('inf'))
        score = 0
        if pos1 != float('inf'):
            score += 1.0 / (k + pos1)
        if pos2 != float('inf'):
            score += 1.0 / (k + pos2)
        text = next((r['text'] for r in ranks1 if r['corpus_id'] == cid),
                    next((r['text'] for r in ranks2 if r['corpus_id'] == cid), ""))
        fused.append({'corpus_id': cid, 'score': score, 'text': text})
    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused if not top_k else fused[:top_k]

# Weighted Mean (normalized)
def wm(ranks1, ranks2, weight1=0.5, weight2=0.5, top_k=None):
    ranks1 = normalize_scores(ranks1.copy())
    ranks2 = normalize_scores(ranks2.copy())
    dict1 = {r['corpus_id']: r['score'] for r in ranks1}
    dict2 = {r['corpus_id']: r['score'] for r in ranks2}
    all_ids = set(dict1.keys()) | set(dict2.keys())
    fused = []
    for cid in all_ids:
        score1 = dict1.get(cid, 0.0)
        score2 = dict2.get(cid, 0.0)
        fused_score = (weight1 * score1 + weight2 * score2) / (weight1 + weight2)
        text = next((r['text'] for r in ranks1 if r['corpus_id'] == cid),
                    next((r['text'] for r in ranks2 if r['corpus_id'] == cid), ""))
        fused.append({'corpus_id': cid, 'score': fused_score, 'text': text})
    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused if not top_k else fused[:top_k]

# RRF + Weighted (RRF scores與normalized score相加)
'''
alpha 的意義：
    alpha = 1.0
        → 完全用 RRF，忽略原始分數，排序只依據「位置」。
    alpha = 0.0
        → 完全用加權平均，只靠標準化後的分數。
    alpha = 0.5
        → 各取一半，結合兩種策略（常用預設值）。
'''
def rrf_wm(ranks1, ranks2, k=60, weight1=0.5, weight2=0.5, alpha=0.5, top_k=None):
    # 計算RRF分數
    rrf_fused = rrf(ranks1, ranks2, k=k)
    rrf_scores = {r['corpus_id']: r['score'] for r in rrf_fused}

    # 正規化原始分數並加權
    ranks1 = normalize_scores(ranks1.copy())
    ranks2 = normalize_scores(ranks2.copy())
    dict1 = {r['corpus_id']: r['score'] for r in ranks1}
    dict2 = {r['corpus_id']: r['score'] for r in ranks2}
    all_ids = set(dict1.keys()) | set(dict2.keys()) | set(rrf_scores.keys())
    fused = []
    for cid in all_ids:
        score1 = dict1.get(cid, 0.0)
        score2 = dict2.get(cid, 0.0)
        weighted_score = (weight1 * score1 + weight2 * score2) / (weight1 + weight2)
        final_score = alpha * rrf_scores.get(cid, 0.0) + (1 - alpha) * weighted_score
        text = next((r['text'] for r in ranks1 if r['corpus_id'] == cid),
                    next((r['text'] for r in ranks2 if r['corpus_id'] == cid), ""))
        fused.append({'corpus_id': cid, 'score': final_score, 'text': text})
    fused.sort(key=lambda x: x['score'], reverse=True)
    return fused if not top_k else fused[:top_k]

# NDCG (Normalized Discounted Cumulative Gain)
def ndcg_at_k(result, relevant_dict, k=5, use_exp=True):
    """
    result: 排名結果 [{'corpus_id': int, ...}, ...]
    relevant_dict: {corpus_id: relevance_score}，例如 {9:3, 3:2, 0:1}
    k: 計算前k名
    use_exp: 是否使用 2^rel - 1 來加權
    """
    def gain(rel):
        return (2**rel - 1) if use_exp else rel

    dcg = 0
    for i, item in enumerate(result[:k]):
        rel = relevant_dict.get(item['corpus_id'], 0)
        dcg += gain(rel) / np.log2(i + 2)

    ideal_rels = sorted(relevant_dict.values(), reverse=True)[:k]
    idcg = sum(gain(rel) / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0

# 計算 Top-1 Accuracy
def top1_accuracy(result, ground_truth_ids):
    """
    result: 排名結果 [{'corpus_id': int, ...}, ...]
    ground_truth_ids: set 或 list，代表理想答案的 corpus_id（通常從 gemini_llm 排名第一名來取）
    回傳: Top-1 是否正確 (單查詢為 1 或 0，或多查詢時為比例)
    """
    if not result:
        return 0.0
    top1_id = result[0]['corpus_id']
    return 1.0 if top1_id in ground_truth_ids else 0.0



'''
Citation coverage (CC)
'''
Triplet = Tuple[str, str, str]

# 正規化字串（小寫、去除多餘空白）
def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

# 別名比對（完全相同或子字串）
def alias_match(token: str, aliases: List[str]) -> bool:
    t = norm(token)
    return any(t == norm(a) or t in norm(a) or norm(a) in t for a in aliases)

# 判斷是否支援該題金標 triplet（單一答案）
def supports_gold_triplet_extractive(
    trip: Triplet,
    answer_aliases: List[str],
    allowed_rels: List[str] = None
) -> bool:
    h, r, t = (norm(trip[0]), norm(trip[1]), norm(trip[2]))
    # 只要 head 或 tail 命中答案實體即可（如果題型需要 head/tail 指定，可加嚴）
    entity_hit = alias_match(h, answer_aliases) or alias_match(t, answer_aliases)
    if not entity_hit:
        return False
    if allowed_rels is None or len(allowed_rels) == 0:
        return True  # 實體對齊模式
    return norm(r) in {norm(rr) for rr in allowed_rels}

# 判斷 Top-K 是否支援該題金標 triplet（單一答案）
def supported_by_topk(q: Dict, K: int) -> int:
    """回傳 0/1：Top-K 是否支援該題金標 triplet（單一答案）。"""
    ans_aliases = q["answer_aliases"]
    allowed_rels = q.get("allowed_rels", [])  # 可空
    refs = q["references"][:K]
    for ref in refs:
        # 先用抽取的 triplets 判斷
        for tri in ref.get("triplets", []):
            if supports_gold_triplet_extractive(tri, ans_aliases, allowed_rels):
                return 1
        # （可選）退而求其次：若沒抽到 triplets，直接在原文 text 做實體命中
        txt = norm(ref.get("text", ""))
        if any(norm(a) in txt for a in ans_aliases):
            return 1
    return 0

# 計算 Coverage@K（多題平均）
def coverage_at_k(dataset: List[Dict], K: int) -> float:
    hits = sum(supported_by_topk(q, K) for q in dataset)
    cov = hits / len(dataset) if dataset else 0.0
    return round(cov, 4)  # 四位小數

# 計算 Coverage@1 到 Coverage@15（多題平均）
def coverage_curve_k1_15(dataset: List[Dict]) -> List[Tuple[int, float]]:
    return [(k, coverage_at_k(dataset, k)) for k in range(1, 16)]