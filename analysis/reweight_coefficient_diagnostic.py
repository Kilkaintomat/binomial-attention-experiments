"""Read-only diagnosis of strict CPU audit versus stored inference coefficients."""
from common import *
from transport_reasoning.reachability.config import allocation

allocation()
freeze_protocol()
rows = [read(p)['coefficients'] for p in sorted((OUT/'evaluations').glob('**/*.json'))]
unique = {json.dumps(x,sort_keys=True):x for x in rows}.values()
result = {'evaluations':len(rows),'settings':len(unique),'threads':[]}
for threads in [1,4]:
    torch.set_num_threads(threads)
    cases = []
    for saved in unique:
        expected = weight_row(saved['filter'],saved['r'],saved['rule'])
        diffs = {}
        for key in saved:
            if key in ['filter','rule','r'] or saved[key] is None:
                assert saved[key] == expected[key]
            else:
                diffs[key] = float(np.max(np.abs(np.asarray(saved[key])-np.asarray(expected[key]))))
        cases.append({'filter':saved['filter'],'r':saved['r'],'rule':saved['rule'],
            'exact_dictionary_match':saved==expected,'absolute_differences':diffs,
            'float32_weights_equal':bool(np.array_equal(np.array(saved['weights'],np.float32),np.array(expected['weights'],np.float32)))})
    result['threads'].append({'threads':threads,'cases':cases})
write_json(OUT/'coefficient_audit_diagnostic.json',result)
print(json.dumps(result),flush=True)
