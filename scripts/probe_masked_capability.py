"""Two fixed source-only capability questions; output is text, never executed.

The existing tdrun mlx_py runner owns serialization and Metal isolation. Input is
an authored source fixture; no model downloads, tools, training or worker execution.
"""
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
import psutil


def main():
    import sys
    fixture=Path(sys.argv[1]); data=json.loads(fixture.read_text())
    out=Path(os.environ['TDRUN_RUN_DIR'])
    swap0=psutil.swap_memory().used
    def guard():
        available=psutil.virtual_memory().available;growth=psutil.swap_memory().used-swap0
        if available<8*1024**3 or growth>128*1024**2:raise RuntimeError('resource guard; no retry')
        return dict(available_bytes=available,swap_growth_bytes=growth)
    config=dict(source='executed',purpose='source-only capability floor, two independent questions, not reward groups',
        model_revision=data['model'].split('/')[-1],model='Qwen3-4B-Instruct-2507-4bit',
        seed=0,temperature=0,max_tokens=1024,context=8192,thinking=False,
        fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),min_available_gib=8,max_swap_growth_mib=128,
        versions={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','transformers')})
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    results=[]
    try:
        before=guard()
        import mlx.core as mx
        from mlx_lm import load,stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.set_default_device(mx.gpu)
        model,tok=load(data['model'],tokenizer_config={'trust_remote_code':False,'local_files_only':True})
        mx.eval(model.parameters());guard()
        for name,question in data['questions'].items():
            before=guard();mx.random.seed(0)
            prompt=tok.apply_chat_template([{'role':'user','content':question+'\n\n'+data['source']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            assert len(ids)+1024<=8192,'context budget exceeded'
            parts=[];tokens=[];started=time.perf_counter();last=None
            for r in stream_generate(model,tok,ids,max_tokens=1024,sampler=make_sampler(temp=0)):
                parts.append(r.text);tokens.append(r.token);last=r
                if len(tokens)%16==0:guard()
            mx.synchronize()
            results.append(dict(question=name,prompt=prompt,raw=''.join(parts),token_ids=tokens,
                prompt_tokens=len(ids),generation_tokens=last.generation_tokens,finish_reason=last.finish_reason,
                wall_s=time.perf_counter()-started,before=before,after=guard(),peak_mlx_gb=last.peak_memory))
            (out/'responses.json').write_text(json.dumps(results,indent=2)+'\n');mx.clear_cache()
    except Exception as e:
        (out/'interruption.json').write_text(json.dumps(dict(source='executed',error_class=type(e).__name__,error=str(e),completed_responses=len(results)),indent=2)+'\n')
        raise
    print('PASS: two source-only responses saved as text; no generated code executed')

if __name__=='__main__':main()
