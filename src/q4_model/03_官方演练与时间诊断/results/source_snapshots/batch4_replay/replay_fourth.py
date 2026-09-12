import json
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parents[2]
runtime = max((p for p in (root/'tmp/official_archive_audit').glob('run-*')
               if (p/'src/q4_model/analyze_official_results.py').is_file()), key=lambda p:p.stat().st_mtime)
sys.path.insert(0,str(runtime/'src/q4_model'))
import analyze_official_results as audit
report = json.loads((root/'src/q4_model/03_官方演练与时间诊断/results/official_q4_20260912_batch4_analysis.json').read_text(encoding='utf-8'))
case = report['cases'][3]
with zipfile.ZipFile(sys.argv[1]) as z:
    rows=[json.loads(l) for l in z.read('测试结果/q4-practice-04.jsonl').decode('utf-8-sig').splitlines() if l.strip()]
    saved_trace=json.loads(z.read('测试结果/q4-practice-04.jsonl.beliefs.json').decode('utf-8-sig'))
summary=next(r for r in rows if r.get('type')=='session_summary')
calls=[r for r in rows if r.get('path') in ('/measure','/clear')]
actions=[dict(a,response=r['response']) for a,r in zip(case['actions'],calls)]
try:
    result,trace,details=audit.replay_logged_actions(actions)
    audit.assert_tree_close(trace,saved_trace)
    audit.assert_tree_close(json.loads(json.dumps(result)),{k:summary[k] for k in result})
    details.update(trace_and_certificate_match=True, official_simulator_contacted=False)
except AssertionError as exc:
    details=dict(conditional_replay_failed=True,error=str(exc)[:700],official_simulator_contacted=False)
output=root/'src/q4_model/03_官方演练与时间诊断/results/official_q4_20260912_batch4_replay_detail.json'
with output.open('x',encoding='utf-8') as f:json.dump(details,f,ensure_ascii=False,indent=2)
print(json.dumps(details,ensure_ascii=False,indent=2))
