"""Local in-situ fast-path check; writes only synthetic example results."""
import argparse
import json
import os
import statistics
import time

from agent_optimize.dispatch import observe_dispatch
from agent_optimize.evaluator import compare_results
from benchmarks.workloads import WORKLOADS, make_events

parser = argparse.ArgumentParser()
parser.add_argument('--case', required=True, choices=WORKLOADS)
parser.add_argument('--events', type=int, default=1000)
options = parser.parse_args()
events = make_events(options.events)
function = WORKLOADS[options.case].function
if os.environ.get('AGENT_OPTIMIZE_SESSION'):
    result = function(events)
    print(json.dumps({'phase': 'recording', 'length': len(result)}))
else:
    stats = {}
    start = time.perf_counter_ns()
    with observe_dispatch(stats):
        result = function(events)
    first_use_ns = time.perf_counter_ns() - start
    expected = function.__wrapped__(events)
    equal, reason = compare_results(expected, result)
    if not stats.get('used') or not equal:
        raise SystemExit(f'generated implementation not used or output mismatch: {stats} {reason}')
    def timed(call):
        start = time.perf_counter_ns()
        call(events)
        return time.perf_counter_ns() - start
    for _ in range(3):
        function(events)
        function.__wrapped__(events)
    samples = {'reference': [], 'fast': []}
    for i in range(21):
        for label in (('reference', 'fast') if i % 2 == 0 else ('fast', 'reference')):
            samples[label].append(timed(function if label == 'fast' else function.__wrapped__))
    med = {label: statistics.median(values) for label, values in samples.items()}
    print(json.dumps({'phase': 'ordinary', 'case': options.case, 'events': options.events,
                      'dispatch': stats, 'exact_match': equal, 'first_use_ms': first_use_ns / 1e6,
                      'reference_ms': med['reference'] / 1e6, 'fast_ms': med['fast'] / 1e6,
                      'speedup': med['reference'] / med['fast'], 'samples': 21}))
