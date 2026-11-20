# Evaluation summary: groq_hyde_on

| metric | value |
|---|---|
| scoring | LLM judge with reference answers (qwen/qwen3.8-27b); leak check in code |
| judge_vs_strings_agree | 2/2 |
| judge_errors | 0 |
| use_hyde | True |
| use_mmr | False |
| repeats | 1 |
| cases | 2 |
| cases_passed | 2 |
| case_pass_rate | 1.0 |
| run_pass_rate | 1.0 |
| retrieval_hit_at_5 | 1.0 |
| multi_section_coverage_at_5 | None |
| multi_section_full_coverage | None |
| multi_section_run_pass_rate | None |
| status_accuracy | 1.0 |
| citation_accuracy | 1.0 |
| content_accuracy | 1.0 |
| leaks | 0 |
| infra_errors | 0 |
| hyde_fallbacks | 0 |
| inconsistent_cases | 0 |
| latency_ms_p50 | 1246 |
| latency_ms_p95 | 1268 |
| latency_ms_p95_llm_answered_path | 1268 |
| latency_ms_max | 1268 |
| llm_calls_avg | 2.0 |
| llm_calls_max | 2 |
| max_citations | 2 |
| status_counts | {'answered': 2} |
| pass_by_category | {'direct': '1/1', 'injection': '1/1'} |
| failed_cases | [] |
