Action/trajectory supervision: judge the action, not the reasoning
Thesis: a supervisor that sees the finished command + episode history beats one that judges partial reasoning mid-stream.

Results — scripted matrix (judge alone; command check + stall detector off)
supervisor sees	TIR (raw SQL)	raw SQL hit DB	false halts on matched clean control
reasoning only	42%	3/12	—
+ command	89%	—	2/14
+ full trajectory	100%	0/12	0/16
Command-only judge flips verdicts on byte-identical inputs; trajectory fixes it.

Results — real Claude (Sonnet 5.5, 4 eps/cell)
0 violations proposed; 0 false halts / 55 steps
Migrations 0/12 → 12/12 after one prompt line ("split across sessions at your own discretion")
What's in it
SUPERVISION_MODE=reasoning|action|trajectory; action modes gate the tool until the verdict
OpenRouter judge (qwen/qwen3.8-max-0902): strict JSON schema, usage + cost logged
split / split_clean scenarios: violation spread over 3 benign-looking steps, plus a matched control
run_matrix.py: context × supervisor matrix, --resume, --rescore (offline)
Premature FINISH is now its own violation kind, separate from raw-SQL TIR
Also brings the unmerged caching / cascade judge / stall / command-check commits (PR Follow-ups: prompt caching, cascade judge (fewer false halts), stall escalation #3 landed on obs/langfuse, not main)
Caveats
Small n, one judge model, scenarios I wrote. No real-Claude violation yet, so TIR on real traffic is still untested.

Cost: ~$2.75 total.
