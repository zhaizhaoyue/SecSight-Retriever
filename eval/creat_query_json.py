import json

# [TRANSLATED] queries
queries = [
    # [TRANSLATED]、[TRANSLATED]
    "What was Apple’s total revenue in fiscal year 2023?",
    "How much did Tesla spend on research and development in 2022?",
    "What is Microsoft’s net income in Q4 2023?",
    "What were Nvidia’s operating expenses in 2021?",
    "How much cash and cash equivalents did Amazon report in 2023?",

    # [TRANSLATED]、[TRANSLATED]/[TRANSLATED]
    "What are Apple’s main sources of revenue in its 2023 10-K?",
    "How does Tesla describe risks related to supply chain disruptions?",
    "What did Microsoft highlight as key growth drivers in 2023?",
    "How does Google explain its strategy for AI and cloud computing?",
    "What competitive risks did Meta identify in its annual report?",

    # [TRANSLATED]、[TRANSLATED]/[TRANSLATED]
    "Compare Apple’s iPhone revenue in 2023 vs 2022.",
    "What is the year-over-year change in Amazon’s advertising revenue in 2023?",
    "How did foreign exchange rates impact Tesla’s financial results in 2022?",
    "Which segment grew the fastest for Microsoft in 2023?",
    "Did Nvidia’s data center revenue surpass gaming revenue in 2022?",

    # [TRANSLATED]、[TRANSLATED]/[TRANSLATED]
    "Show me Apple’s sales breakdown by product in 2023.",
    "What part of Tesla’s revenue comes from regulatory credits?",
    "How much money did Microsoft make from cloud last year?",
    "Tell me about Google’s expenses related to R&D.",
    "Did Meta mention risks about competition with TikTok?"
]

# [TRANSLATED] JSONL [TRANSLATED]
output_path = "data/test_queries.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps({"query": q}, ensure_ascii=False) + "\n")

output_path
