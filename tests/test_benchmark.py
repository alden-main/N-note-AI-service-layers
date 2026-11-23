from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark import bench, worker


class BenchmarkTests(unittest.TestCase):
    def test_workflow_runs_with_fixed_reference_and_without_answers(self):
        from n_note_ai_service_layers.contracts.draft import DraftResult
        class Generator:
            def generate(self, payload, schema):
                self.payload = payload
                return DraftResult(proposed_fields={'assessment':'통증 호소함.', 'plan_and_intervention':'심호흡 안내함.'}, evidence_refs={}, needs_confirmation=False, model_version='test', prompt_version='test')
        generator=Generator()
        _,rows,_=bench.validate(bench.DATA)
        data={k:rows[0][k] for k in ('request','transcript','reference_time')}
        with patch.dict('os.environ',{'N_NOTE_LLM_TOOL_CALLING':'false'}), patch('n_note_ai_service_layers.adapters.model_providers.draft_generator',return_value=generator):
            result=worker.run(data)
        self.assertIn('통증',result['fields']['assessment'])
        self.assertTrue(generator.payload['reference_time'].startswith('2026-11-18T09:00'))
        self.assertNotIn('summary',generator.payload)

    def test_dataset_and_splits(self):
        manifest, rows, refs = bench.validate(bench.DATA)
        self.assertEqual((len(rows), len(refs)), (100,100))
        self.assertEqual(sum(x['request']['schema_id']=='nursing-daily-note' for x in rows),60)
        self.assertEqual(sum(x['split']=='holdout' for x in rows),30)
        self.assertEqual(manifest['review_status'],'unreviewed')

    def test_percentiles_and_failures(self):
        records=[{'status':'ok','total_seconds':float(i),'llm_seconds':float(i)} for i in range(1,101)]
        records.append({'status':'error','error_type':'TimeoutExpired','total_seconds':5})
        s=bench.stats(records)
        self.assertEqual(s['groups']['ok']['total_seconds']['p95'],95)
        self.assertEqual(s['groups']['ok']['total_seconds']['p99'],99)
        self.assertEqual(s['timeouts'],1)
        self.assertIsNone(bench.percentile([], .99))

    def test_worker_allowlist_does_not_receive_references(self):
        _, rows, refs=bench.validate(bench.DATA)
        row=dict(rows[0], expected=refs[0], summary='SECRET')
        class P:
            returncode=0
            stdout='{"status":"ok","fields":{}}\n'
        with patch('benchmark.bench.subprocess.run',return_value=P()) as run:
            self.assertEqual(bench.invoke(row,1)['status'],'ok')
        payload=json.loads(run.call_args.kwargs['input'])
        self.assertEqual(set(payload),{'request','transcript','reference_time'})
        self.assertNotIn('SECRET',json.dumps(payload))

    def test_worker_rejects_extra_data(self):
        import io
        with patch('sys.stdin',io.StringIO('{"request":{},"transcript":"", "reference_time":"x", "answer":"secret"}')), patch('sys.stdout',new_callable=io.StringIO) as out:
            self.assertEqual(worker.main(),1)
            self.assertNotIn('secret',out.getvalue())

    def test_score_never_treats_anchor_as_correct(self):
        m,rows,refs=bench.validate(bench.DATA)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            bench.write_json(p/'manifest.json',{'dataset':m,'repeat':1})
            bench.write_jsonl(p/'results.jsonl',[{'case_id':rows[0]['case_id'],'repeat':1,'status':'ok','fields':{refs[0]['facts'][0]['field_id']:refs[0]['facts'][0]['anchors'][0]}}])
            args=type('Args',(),{'dataset':bench.DATA,'run':p,'reviews':None})()
            self.assertEqual(bench.score(args),2)
            self.assertEqual(bench.read_jsonl(p/'scores.jsonl')[0]['facts'][0]['status'],'uncertain')

    def test_dataset_tamper_detected(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data'; shutil.copytree(bench.DATA,p)
            with (p/'references.jsonl').open('a') as f: f.write(' ') 
            with self.assertRaises(ValueError): bench.validate(p)

    def test_comparison_refuses_incomplete_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=root/'a'; b=root/'b'; a.mkdir(); b.mkdir()
            manifest={'dataset':{},'repeat':1}
            for p in (a,b):
                bench.write_json(p/'manifest.json',manifest)
                bench.write_jsonl(p/'results.jsonl',[{'case_id':'syn-001','repeat':1,'status':'ok','total_seconds':1}])
            args=type('Args',(),{'baseline':a,'candidate':b,'output':root/'comparison.json'})()
            self.assertEqual(bench.compare(args),2)
            self.assertFalse(args.output.exists())

    def test_reported_unsupported_fact_is_a_failure(self):
        m,rows,_=bench.validate(bench.DATA)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            bench.write_json(p/'manifest.json',{'dataset':m,'repeat':1})
            bench.write_jsonl(p/'results.jsonl',[{'case_id':rows[0]['case_id'],'repeat':1,'status':'ok','fields':{}}])
            review=p/'reviews.jsonl'
            bench.write_jsonl(review,[{'case_id':rows[0]['case_id'],'repeat':1,'reviewer':'r','facts':{'primary':'correct'},'unsupported_fact':True,'critical_error':False}])
            args=type('Args',(),{'dataset':bench.DATA,'run':p,'reviews':review})()
            self.assertEqual(bench.score(args),1)

    def test_profile_environment_precedence_and_values(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'deploy/models').mkdir(parents=True)
            (root/'deploy/models/demo.env').write_text(
                'N_NOTE_LLM_MODEL=from-profile\nN_NOTE_LLM_BASE_URL=http://local-llm:8000/v1\n'
                'N_NOTE_LOCAL_LLM_ARGS=--max-model-len 16384 --enable-auto-tool-choice\n'
                'N_NOTE_LLM_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}\n',encoding='utf-8')
            (root/'.env').write_text('N_NOTE_LLM_MODEL=from-dotenv\nN_NOTE_LLM_TEMPERATURE=0.5\n',encoding='utf-8')
            keep={k:v for k,v in os.environ.items() if not k.startswith(('N_NOTE_','OPENAI_','MCP_'))}
            with patch.dict('os.environ',{**keep,'N_NOTE_LLM_BASE_URL':'http://127.0.0.1:18000/v1'},clear=True):
                bench.load_environment('demo',root)
                self.assertEqual(os.environ['N_NOTE_LLM_BASE_URL'],'http://127.0.0.1:18000/v1')  # 셸 우선
                self.assertEqual(os.environ['N_NOTE_LLM_MODEL'],'from-profile')                   # 프로필 > .env
                self.assertEqual(os.environ['N_NOTE_LLM_TEMPERATURE'],'0.5')                      # .env 보충
                self.assertEqual(os.environ['N_NOTE_LOCAL_LLM_ARGS'],'--max-model-len 16384 --enable-auto-tool-choice')
                self.assertEqual(json.loads(os.environ['N_NOTE_LLM_EXTRA_BODY']),{'chat_template_kwargs':{'enable_thinking':False}})
            for bad in ('missing','../demo',''):
                with self.assertRaises(ValueError): bench.load_environment(bad,root)

    def test_repository_profiles_parse(self):
        from dotenv import dotenv_values
        for path in sorted((bench.ROOT/'deploy/models').glob('*.env')):
            values=dotenv_values(path)
            if 'N_NOTE_LLM_EXTRA_BODY' in values: json.loads(values['N_NOTE_LLM_EXTRA_BODY'])
            self.assertTrue(values.get('N_NOTE_MODEL_PROFILE'),path.name)

    def test_compose_internal_model_host_is_rejected(self):
        cfg=lambda url: type('Config',(),{'llm_base_url':url})()
        with self.assertRaises(ValueError): bench.check_model_endpoint(cfg('http://local-llm:8000/v1'))
        bench.check_model_endpoint(cfg('http://127.0.0.1:18000/v1'))
        bench.check_model_endpoint(cfg(None))

if __name__=='__main__': unittest.main()
