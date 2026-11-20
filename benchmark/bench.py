"""Isolated synthetic benchmark. Run with python -m benchmark.bench --help."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, platform, random, statistics, subprocess, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'eval/benchmark/v1'
RESULTS = ROOT/'eval/benchmark/results'
# Compose 내부 DNS 이름이다. 벤치마크는 호스트에서 돌므로 이 주소로는 모델 서버에 닿지 않는다.
COMPOSE_ONLY_HOSTS = {'local-llm','local-stt','ai-service','gateway'}

def load_environment(profile=None, root=ROOT):
    """셸 환경 변수 > 모델 프로필(deploy/models/<profile>.env) > .env 순으로 우선한다(scripts/compose-model.sh와 같은 순서).

    프로필 파일은 따옴표 없는 공백·JSON 값이 있어 셸 ``source``로 읽으면 깨지므로 python-dotenv로 읽는다.
    """
    from dotenv import load_dotenv
    if profile is not None:
        path=root/'deploy/models'/f'{profile}.env'
        if not profile or Path(profile).name!=profile or not path.is_file(): raise ValueError(f'unknown model profile: {profile}')
        load_dotenv(path,override=False)
    load_dotenv(root/'.env',override=False)

def check_model_endpoint(config):
    from urllib.parse import urlsplit
    host=urlsplit(config.llm_base_url or '').hostname
    if host in COMPOSE_ONLY_HOSTS:
        raise ValueError(f'N_NOTE_LLM_BASE_URL uses the Compose-internal host {host!r}; set it to an address reachable from this machine '
                         '(e.g. N_NOTE_LLM_BASE_URL=http://127.0.0.1:18000/v1 through an SSH tunnel, see benchmark/README.md)')

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def tree_digest(paths):
    h=hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(ROOT)).encode()); h.update(b'\0'); h.update(path.read_bytes()); h.update(b'\0')
    return h.hexdigest()
def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
def write_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
def write_jsonl(path, rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def validate(path):
    manifest=json.loads((path/'manifest.json').read_text())
    for name in ('inputs.jsonl','references.jsonl'):
        if digest(path/name)!=manifest['files'][name]: raise ValueError(f'dataset hash mismatch: {name}')
    inputs, refs=read_jsonl(path/'inputs.jsonl'), read_jsonl(path/'references.jsonl')
    ids=[x['case_id'] for x in inputs]; rids=[x['case_id'] for x in refs]
    if len(ids)!=100 or len(set(ids))!=100 or set(ids)!=set(rids) or len(set(rids))!=100: raise ValueError('expected 100 unique paired cases')
    byid={r['case_id']:r for r in refs}; families={}; texts=set()
    for row in inputs:
        req=row['request']; ref=byid[row['case_id']]
        if set(req)!={'patient_id','encounter_id','schema_id','language_hint'}: raise ValueError('unexpected request fields')
        if req['schema_id'] not in ('nursing-daily-note','sbar-handoff'): raise ValueError('unknown schema')
        form=json.loads((ROOT/'forms'/req['schema_id']/'form.json').read_text())
        fields={f['field_id'] for s in form['sections'] for f in s['fields']}
        if not all(x['field_id'] in fields and x['evidence'] in row['transcript'] for x in ref['facts']): raise ValueError('invalid evidence or field')
        if not set(ref['must_be_empty'])<=fields: raise ValueError('invalid empty field')
        if datetime.fromisoformat(row['reference_time']).tzinfo is None: raise ValueError('naive reference time')
        if row['transcript'] in texts: raise ValueError('duplicate transcript')
        texts.add(row['transcript']); families.setdefault(row['family'],set()).add(row['split'])
    if any(len(v)!=1 for v in families.values()): raise ValueError('scenario family crosses split')
    return manifest, inputs, refs

def percentile(values,p):
    return sorted(values)[math.ceil(p*len(values))-1] if values else None

def stats(records, include_repeats=True):
    groups={}
    for status in ('ok','error'):
        rows=[r for r in records if r['status']==status]
        groups[status]={'count':len(rows)}
        for key in ('total_seconds','llm_seconds'):
            v=[r[key] for r in rows if isinstance(r.get(key),(int,float))]
            groups[status][key]={'n':len(v), **{f'p{p}':percentile(v,p/100) for p in (50,95,99)}}
    result={'requested':len(records),'groups':groups,'errors':sum(r['status']!='ok' for r in records),
            'timeouts':sum(r.get('error_type')=='TimeoutExpired' for r in records), 'percentile_method':'nearest-rank'}
    if include_repeats:
        result['by_repeat']={str(i):stats([r for r in records if r.get('repeat')==i],False) for i in sorted({r['repeat'] for r in records if 'repeat' in r})}
    return result

def invoke(row,timeout):
    # Explicit allowlist: case identifiers, summary, reference answers and labels never reach the worker.
    payload={'request':row['request'],'transcript':row['transcript'],'reference_time':row['reference_time']}
    start=time.perf_counter()
    try:
        proc=subprocess.run([sys.executable,'-m','benchmark.worker'],input=json.dumps(payload,ensure_ascii=False),
                            text=True,capture_output=True,timeout=timeout,cwd=ROOT,env=os.environ.copy())
        lines=proc.stdout.strip().splitlines()
        result=json.loads(lines[-1]) if lines else {'status':'error','error_type':'WorkerNoOutput'}
        if proc.returncode and result.get('status')=='ok': result={'status':'error','error_type':'WorkerExitError'}
    except subprocess.TimeoutExpired: result={'status':'error','error_type':'TimeoutExpired'}
    except (ValueError,OSError): result={'status':'error','error_type':'WorkerProtocolError'}
    result['wall_seconds']=time.perf_counter()-start
    if result['status']!='ok': result['total_seconds']=result['wall_seconds']
    return result

def run(args):
    manifest, inputs, _ = validate(args.dataset)
    from n_note_ai_service_layers import settings
    if settings.get().llm_tool_calling_enabled:
        raise ValueError('set N_NOTE_LLM_TOOL_CALLING=false before running the no-tools benchmark')
    check_model_endpoint(settings.get())
    # The model worker is never given references, labels, case IDs or summaries.
    out=args.out/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    rng=random.SystemRandom(); records=[]
    for repeat in range(1,args.repeat+1):
        batch=inputs.copy(); rng.shuffle(batch)
        for row in batch:
            result=invoke(row,args.timeout)
            records.append({'case_id':row['case_id'],'repeat':repeat,**result})
            print(f"{repeat}/{args.repeat} {len(records)} {result['status']}",file=sys.stderr,flush=True)
    if digest(args.dataset/'inputs.jsonl')!=manifest['files']['inputs.jsonl'] or digest(args.dataset/'references.jsonl')!=manifest['files']['references.jsonl']: raise ValueError('dataset changed during run')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True).stdout.strip())
    hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'forms').glob('*/form.json'))}
    config=settings.get()
    write_json(out/'manifest.json',{'run_id':args.run_id,'created_at':datetime.now(timezone.utc).isoformat(),
               'dataset':manifest,'git_commit':commit,'dirty':dirty,'form_hashes':hashes,'repeat':args.repeat,
               'timeout_seconds':args.timeout,'concurrency':1,'tool_mode':'no-tools-v1',
               'platform':platform.platform(),'machine':platform.machine(),'processor':platform.processor(),'python':platform.python_version(),
               'model_profile':config.model_profile,'llm_base_url':config.llm_base_url,
               'model':config.llm_endpoint.model, 'api':config.llm_api,
               'temperature':config.llm_temperature, 'max_tokens':config.llm_max_tokens,
               'model_max_retries':config.model_max_retries,'llm_timeout_seconds':config.llm_timeout_seconds,
               'llm_extra_body_hash':hashlib.sha256(json.dumps(config.llm_extra_body,sort_keys=True).encode()).hexdigest(),
               'service_code_hash':tree_digest((ROOT/'src/n_note_ai_service_layers').rglob('*.py')),
               'benchmark_code_hash':tree_digest((ROOT/'benchmark').glob('*.py')),
               'prompt_source_hash':digest(ROOT/'src/n_note_ai_service_layers/adapters/openai_draft_generation.py'),
               'reference_review_status':manifest['review_status']})
    write_jsonl(out/'results.jsonl',records); write_json(out/'summary.json',stats(records))
    print(out)
    return 0 if all(r['status']=='ok' for r in records) else 1

def score(args):
    manifest, _, refs=validate(args.dataset); runmanifest=json.loads((args.run/'manifest.json').read_text())
    if manifest!=runmanifest['dataset']: raise ValueError('dataset differs from run')
    references={r['case_id']:r for r in refs}; results=read_jsonl(args.run/'results.jsonl'); scores=[]
    reviews=read_jsonl(args.reviews) if args.reviews else []
    byreview={(x['case_id'],x['repeat']):x for x in reviews}
    if len(byreview)!=len(reviews): raise ValueError('duplicate review')
    expected={(x['case_id'],x['repeat']) for x in results}
    if not set(byreview)<=expected: raise ValueError('review does not match a result')
    for result in results:
        ref=references[result['case_id']]; fields=result.get('fields',{}); text=json.dumps(fields,ensure_ascii=False)
        matches=[]
        review=byreview.get((result['case_id'],result['repeat']))
        if review:
            if not review.get('reviewer') or not isinstance(review.get('unsupported_fact'),bool) or not isinstance(review.get('critical_error'),bool): raise ValueError('review requires reviewer and boolean error decisions')
            if set(review.get('facts',{}))!={f['id'] for f in ref['facts']} or any(v not in ('correct','incorrect','missing','uncertain') for v in review['facts'].values()): raise ValueError('invalid reviewed facts')
        for fact in ref['facts']:
            value=str(fields.get(fact['field_id'],'') or '')
            matches.append({'id':fact['id'],'anchor_present':any(a in value for a in fact['anchors']),'status':review['facts'][fact['id']] if review else 'uncertain', 'critical':fact['critical']})
        privacy=any(x in text for x in ref['sensitive_values']) or bool(__import__('re').search(r'\[\[[A-Z_]+_\d+\]\]',text))
        scores.append({'case_id':result['case_id'],'repeat':result['repeat'],'status':result['status'],
                       'facts':matches,'privacy_leak':privacy,'unexpected_nonempty':[f for f in ref['must_be_empty'] if fields.get(f)],
                       'human_review':'complete' if review and all(f['status']!='uncertain' for f in matches) else 'pending',
                       'reviewer':review.get('reviewer') if review else None,
                       'unsupported_fact':review.get('unsupported_fact') if review else None,
                       'critical_error':review.get('critical_error') if review else None})
    write_jsonl(args.run/'scores.jsonl',scores)
    # Keyword presence is a screening hint, never a semantic correctness judgment.
    pending=sum(x['human_review']=='pending' for x in scores)
    correct=sum(f['status']=='correct' for x in scores for f in x['facts'])
    total=sum(len(x['facts']) for x in scores)
    failures=sum(x['status']!='ok' or x['privacy_leak'] or x['unsupported_fact'] is True or x['critical_error'] is True or any(f['critical'] and f['status']=='incorrect' for f in x['facts']) for x in scores)
    complete=len(scores)==100*runmanifest['repeat'] and len({(x['case_id'],x['repeat']) for x in scores})==len(scores)
    if failures: verdict='FAIL'
    elif pending or not complete or manifest['review_status']!='approved': verdict='INCONCLUSIVE'
    elif correct/total<.95: verdict='FAIL'
    else: verdict='PASS'
    write_json(args.run/'assessment.json',{'verdict':verdict,'scope':'single-run absolute thresholds; comparison thresholds require compare',
                'count':len(scores),'correct_facts':correct,'total_facts':total,'privacy_leaks':sum(x['privacy_leak'] for x in scores),
                'failed_cases':failures,'unreviewed_cases':pending,'dataset_review_status':manifest['review_status']})
    print(f'{verdict}: anchor matches are screening hints, not semantic scores')
    return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[verdict]

def compare(args):
    a=json.loads((args.baseline/'manifest.json').read_text()); b=json.loads((args.candidate/'manifest.json').read_text())
    comparable=['dataset','form_hashes','repeat','timeout_seconds','concurrency','tool_mode','platform','machine','processor','python','model_max_retries','llm_timeout_seconds','benchmark_code_hash']
    diffs=[k for k in comparable if a.get(k)!=b.get(k)]
    if diffs: print('INCONCLUSIVE: incompatible '+', '.join(diffs)); return 2
    ar=read_jsonl(args.baseline/'results.jsonl'); br=read_jsonl(args.candidate/'results.jsonl')
    aa={(x['case_id'],x['repeat']):x for x in ar}; bb={(x['case_id'],x['repeat']):x for x in br}
    if len(aa)!=len(ar) or len(bb)!=len(br) or set(aa)!=set(bb) or len(aa)!=100*a['repeat']:
        print('INCONCLUSIVE: missing, duplicate or mismatched case/repeat results'); return 2
    assessments=[]
    for path in (args.baseline,args.candidate):
        f=path/'assessment.json'
        assessments.append(json.loads(f.read_text()) if f.exists() else None)
    def latency(rows):
        return [percentile([r['total_seconds'] for r in rows if r['repeat']==repeat and r['status']=='ok'],.95) for repeat in range(1,a['repeat']+1)]
    av,bv=latency(ar),latency(br)
    if any(x is None for x in av+bv): ratio=None
    else:
        base=statistics.median(av); ratio=statistics.median(bv)/base if base else None
    if assessments[1] and assessments[1]['verdict']=='FAIL': verdict='FAIL'
    elif any(x is None or x['verdict']!='PASS' for x in assessments) or ratio is None: verdict='INCONCLUSIVE'
    elif ratio>1.2 or assessments[1]['correct_facts']/assessments[1]['total_facts'] < assessments[0]['correct_facts']/assessments[0]['total_facts']-.02: verdict='FAIL'
    else: verdict='PASS'
    result={'verdict':verdict,'reason':'absolute review results and paired aggregate thresholds', 'p95_ratio':ratio,
            'configuration_changes':{k:[a.get(k),b.get(k)] for k in ('git_commit','dirty','model_profile','llm_base_url','model','api','temperature','max_tokens','llm_extra_body_hash','prompt_source_hash','service_code_hash') if a.get(k)!=b.get(k)},
            'baseline':stats(ar),'candidate':stats(br),
            'cases':[{'case_id':key[0],'repeat':key[1],'baseline_status':aa[key]['status'],'candidate_status':bb[key]['status'],
                      'baseline_seconds':aa[key].get('total_seconds'),'candidate_seconds':bb[key].get('total_seconds'),
                      'changed_fields':{f:{'baseline':aa[key].get('fields',{}).get(f),'candidate':bb[key].get('fields',{}).get(f)}
                                        for f in sorted(set(aa[key].get('fields',{}))|set(bb[key].get('fields',{})))
                                        if aa[key].get('fields',{}).get(f)!=bb[key].get('fields',{}).get(f)}}
                     for key in sorted(aa)]}
    write_json(args.output,result); print(f'{verdict}: {args.output}'); return {'PASS':0,'FAIL':1,'INCONCLUSIVE':2}[verdict]

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('validate'); v.add_argument('--dataset',type=Path,default=DATA)
    r=sub.add_parser('run'); r.add_argument('--dataset',type=Path,default=DATA); r.add_argument('--out',type=Path,default=RESULTS)
    r.add_argument('--profile',default=None,help='deploy/models/<profile>.env의 모델 설정을 쓴다(셸 환경 변수가 우선)')
    r.add_argument('--run-id',default=None); r.add_argument('--repeat',type=int,default=3); r.add_argument('--timeout',type=float,default=180)
    s=sub.add_parser('score'); s.add_argument('--dataset',type=Path,default=DATA); s.add_argument('--reviews',type=Path); s.add_argument('run',type=Path)
    c=sub.add_parser('compare'); c.add_argument('--baseline',type=Path,required=True); c.add_argument('--candidate',type=Path,required=True); c.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    try:
        # 워커 프로세스는 이 환경을 그대로 물려받는다.
        load_environment(getattr(args,'profile',None))
        if args.command=='validate':
            m, rows, _=validate(args.dataset); print(json.dumps({'count':len(rows),'categories':dict(collections.Counter(x['category'] for x in rows)), 'review_status':m['review_status']},ensure_ascii=False)); return 0
        if args.command=='run':
            if args.repeat<1 or args.timeout<=0: raise ValueError('repeat and timeout must be positive')
            args.run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
            if not args.run_id or Path(args.run_id).name!=args.run_id or args.run_id in ('.','..'): raise ValueError('invalid run id')
            return run(args)
        return score(args) if args.command=='score' else compare(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f'benchmark error: {exc}',file=sys.stderr); return 2
if __name__=='__main__': sys.exit(main())
