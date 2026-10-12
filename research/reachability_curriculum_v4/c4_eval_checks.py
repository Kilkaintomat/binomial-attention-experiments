"""CPU contracts for the prepared evaluator; no checkpoint or test predictions."""
from c4_common import *
from c4_data import augmented_graph
from c4_evaluate import threshold,calibrated
from transport_reasoning.reachability.config import allocation
from transport_reasoning.reachability.data import stress_graph,metadata,closure


def run():
    allocation();torch.set_num_threads(1);cases=0
    for y,z,expected_t,expected_ba in [
        ([0,0,1,1],[-4.,-3.,-2.,-1.],-2.,1.),
        ([0,1,0,1],[-2.,-2.,-2.,-2.],0.,.5),
        ([0,0,1,1],[-2.,-1.,0.,2.],0.,1.),
        ([0,1],[1.,0.],0.,.5),
    ]:
        t,ba=threshold(y,z);assert t==expected_t and ba==expected_ba
        assert calibrated(y,z,t)['balanced_accuracy']==ba;cases+=1
    graphs=0
    for variant in PROTOCOL['test_plan']['stress_variants']:
        for ell in PROTOCOL['test_plan']['stress_lengths']:
            rng=np.random.default_rng(987000+graphs)
            if variant.startswith('detour'):B,pairs=augmented_graph(256,ell,variant.endswith('cyclic'),rng)
            else:B,pairs,_=stress_graph(256,ell,'cyclic' if variant.endswith('cyclic') else 'dag',rng)
            m=metadata(B,pairs);C=closure(B)
            assert m['distance'].tolist()==[ell,-1] and m['weak_same'].all()
            assert [bool(C[s,t]) for s,t in pairs]==[True,False]
            if variant.endswith('dag'):assert not np.diag(C).any()
            graphs+=1
    # Check path conventions against existing artifacts, without reading test scores.
    controls_count=0
    for value in read(BASE/'final_selection.json')['configs']:
        c=BaseConfig(**value)
        if c.model not in ['F4-one','F4-bin','F4-exp','F4-geo','R32-one']:continue
        for name in ['fixed_test_ID','fixed_test_far_128','fixed_stress']:
            for extension in ['.json','.npz']:assert (control_directory(c)/'evaluation'/(name+extension)).exists()
        controls_count+=1
    assert controls_count==50
    result={'status':'passed','threshold_cases':cases,'stress_generator_cases':graphs,'original_checkpoint_paths':controls_count,
        'test_predictions_performed':False,'new_test_created':False,'source_hash':file_hash(__file__),
        'scheduler_job':os.environ['SLURM_JOB_ID']}
    write_json(ROOT/'checks/evaluation_contracts.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':run()
