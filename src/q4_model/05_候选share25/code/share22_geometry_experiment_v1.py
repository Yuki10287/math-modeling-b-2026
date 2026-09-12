"""Share25 task policy with only the fixed discovery geometry substituted.

The baseline module and its globals are untouched. A separate function object
reuses its exact code with a private globals dictionary and one replacement.
"""
from types import FunctionType

from station22_geometry_v1 import Station22Cover
import task_sharing_solver as baseline

_namespace = dict(baseline.__dict__)
_namespace['PolarCover'] = Station22Cover
_solve = FunctionType(baseline.solve_multi.__code__, _namespace,
    'solve_geometry22', baseline.solve_multi.__defaults__, baseline.solve_multi.__closure__)
assert _solve.__code__ is baseline.solve_multi.__code__
assert {key for key in baseline.__dict__ if _namespace[key] is not baseline.__dict__[key]} == {'PolarCover'}


def solve_multi(api, variant='share25', trace=None, max_active_measures=6):
    result = _solve(api, variant=variant, trace=trace, max_active_measures=max_active_measures)
    if 'certificate' in result:
        certificate = result['certificate']
        if certificate['basis'] != 'count_upper_bound':
            certificate['basis'] = 'directional_cell_cover'
        certificate.pop('spacing', None)
        certificate.update(Station22Cover().geometry_certificate())
    return result
