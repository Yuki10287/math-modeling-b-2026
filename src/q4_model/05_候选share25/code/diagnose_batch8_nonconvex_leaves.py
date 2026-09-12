"""Read-only H1 diagnostic: keep the frozen exact leaf union, not its convex hull.

No solver or new action is run. Reuse recorded depth-6/255-node proof trees;
recover a discarded tree only when the recorded run excluded leaves without
shrinking their convex hull, with identical budget and statistics required.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
sys.path.insert(0, str(Q4))
import negative_region
import negative_region_proof as verifier
import prune_proof


def distance2_to_triangle(query, triangle):
    """Exact Euclidean distance to one rational closed triangle."""
    q = tuple(F(float(x)) for x in query)
    def sub(a, b):
        return tuple(x-y for x, y in zip(a, b))
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]
    edges = list(zip(triangle, triangle[1:]+triangle[:1]))
    signs = [cross(sub(b, a), sub(q, a)) for a, b in edges]
    if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
        return F(0)
    values = []
    for a, b in edges:
        v, w = sub(b, a), sub(q, a)
        length = dot(v, v)
        t = min(F(1), max(F(0), dot(v, w)/length)) if length else F(0)
        values.append(sum((q[k]-a[k]-t*v[k])**2 for k in (0, 1)))
    return min(values)


def retime(events):
    position, channel, total = [0., 0.], 1, 0.
    for e in events:
        total += math.dist(position, e['position'])/5
        if e['action'] == 'measure':
            total += 5+int(channel != e['channel'])
            channel = e['channel']
        else:
            total += 5 if e['clear_result'] == 'success' else 3
        position = e['position']
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    paths = [Path(__file__), Path(negative_region.__file__),
             Path(verifier.__file__), Path(prune_proof.__file__)]
    hashes = {p.relative_to(Q4.parent.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    rows, cases, members = [], [], {}
    with zipfile.ZipFile(args.zip) as archive:
        for name in sorted(n for n in archive.namelist() if n.endswith('.beliefs.json')):
            trace = json.loads(archive.read(name))
            log_name = name.removesuffix('.beliefs.json')
            records = [json.loads(line) for line in archive.read(log_name).decode('utf-8-sig').splitlines() if line]
            members[name] = hashlib.sha256(archive.read(name)).hexdigest()
            members[log_name] = hashlib.sha256(archive.read(log_name)).hexdigest()
            events = []
            for record in records:
                if record.get('path') not in ('/measure', '/clear'):
                    continue
                request, response = record['request'], record['response']
                assert response['accepted'] is True
                action = record['path'][1:]
                events.append(dict(action=action, channel=request['channel'],
                    position=[request['position']['x'], request['position']['y']],
                    **{action+'_result': response[action+'_result']}))
            old_audit = prune_proof.verify_trace(trace, events)
            removed_events, case_rows = set(), []
            for index, row in enumerate(trace):
                if row['phase'] != 'ordered_optical_prune':
                    continue
                info, certificate = row['info'], row['info']['certificate']
                recovered = False
                if certificate is None and info['statistics'].get('excluded_leaves', 0) > 0:
                    result = negative_region.contract_region(row['original_polygon'], row['positives'],
                                                            row['negatives'], max_depth=6, max_nodes=255)
                    assert result['statistics'] == info['statistics'], 'recovered tree statistics changed'
                    certificate, recovered = result['certificate'], True
                plans = trace[index+1]
                assert plans['phase'] == 'optical_plan' and plans['path'] == row['original_path']
                actual = []
                for a in trace[index+2:]:
                    if a['phase'] != 'actual_action':
                        break
                    assert a['reason'] == 'optical_cover' and a['channel'] == row['channel']
                    actual.append(a)
                    if a['result'] == 'success':
                        break
                assert actual and actual[-1]['result'] == 'success'
                assert [a['position'] for a in actual] == [row['original_path'][i] for i in row['kept_indices'][:len(actual)]]
                extra = set()
                proof_audit = None
                if certificate is not None:
                    assert certificate['max_depth'] == 6 and certificate['max_nodes'] == 255
                    proof_audit = verifier.verify_certificate(certificate, row['original_polygon'], row['positives'],
                        row['negatives'], certificate['output_polygon'], require_reduction=info['applied'])
                    original = verifier.convex_hull(verifier.rational_points(certificate['original_polygon']))
                    roots = [(original[0], original[k], original[k+1]) for k in range(1, len(original)-1)]
                    cells = [verifier.descendants(roots[r['root']], r['branch']) for r in certificate['retained_leaves']]
                    assert cells
                    excluded = {i for i, q in enumerate(row['original_path'])
                                if all(distance2_to_triangle(q, cell) > 400 for cell in cells)}
                    assert set(row['removed_indices']) <= excluded, 'nonconvex set lost an old deletion'
                    extra = excluded-set(row['removed_indices'])
                actual_extra = []
                for old_index, action in zip(row['kept_indices'], actual):
                    if old_index in extra:
                        assert action['result'] == 'no_target_in_range', 'proved deletion included success'
                        assert events[action['event']-1]['clear_result'] == 'no_target_in_range'
                        actual_extra.append(action['event'])
                        removed_events.add(action['event'])
                entry = dict(case=log_name.rsplit('/', 1)[-1], channel=row['channel'],
                    planned_points=len(row['original_path']), actual_clear_attempts=len(actual),
                    old_removed_planned=len(row['removed_indices']),
                    nonconvex_tree_available=certificate is not None, recovered_discarded_tree=recovered,
                    retained_leaves=len(certificate['retained_leaves']) if certificate else None,
                    excluded_leaves=len(certificate['excluded_leaves']) if certificate else None,
                    recorded_reason=info['reason'], proof_audit=proof_audit,
                    extra_removed_planned_indices=sorted(extra),
                    extra_removed_actual_event_numbers=actual_extra)
                rows.append(entry);case_rows.append(entry)
            old_time = retime(events)
            candidate_time = retime([e for i, e in enumerate(events, 1) if i not in removed_events])
            assert candidate_time <= old_time+1e-7
            cases.append(dict(case=log_name.rsplit('/', 1)[-1], old_proof_audit=old_audit,
                plans=len(case_rows), old_reconstructed_s=old_time, union_restricted_reconstruction_s=candidate_time,
                additional_saving_s=old_time-candidate_time, extra_removed_actual=len(removed_events),
                extra_removed_planned=sum(len(r['extra_removed_planned_indices']) for r in case_rows)))
            print(cases[-1]['case'], 'extra_actual=',len(removed_events),'saved_s=',old_time-candidate_time,flush=True)
    assert hashes == {p.relative_to(Q4.parent.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths}
    output = dict(passed=True, archive_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),
        member_sha256=members, source_sha256=hashes, source_stable=True,
        local_only=True, official_contacted=False, new_solver_runs=0,
        mechanism='Exact distance from each original optical disk to the union of frozen retained rational triangles, preserving the original order.',
        fixed_max_depth=6, fixed_max_nodes=255,
        plan_count=len(rows), trees_checked=sum(r['nonconvex_tree_available'] for r in rows),
        discarded_trees_recovered=sum(r['recovered_discarded_tree'] for r in rows),
        extra_removed_actual=sum(c['extra_removed_actual'] for c in cases),
        extra_removed_planned=sum(c['extra_removed_planned'] for c in cases),
        additional_saving_s=sum(c['additional_saving_s'] for c in cases), cases=cases, rows=rows,
        limits=['H1 only; no failed-clear disk refinement, no new radius or depth, no changed order.',
                'Skipped/no-excluded-leaf cases follow the original wrapper scope; they do not introduce new contraction trees.',
                'Restricted same-feedback reconstruction is not a new official trial.'])
    with args.out.open('x',encoding='utf-8') as stream:
        json.dump(output,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:output[k] for k in ('passed','plan_count','trees_checked','discarded_trees_recovered',
                     'extra_removed_actual','extra_removed_planned','additional_saving_s')},ensure_ascii=False))


if __name__ == '__main__':
    main()
