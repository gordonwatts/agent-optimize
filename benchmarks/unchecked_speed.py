"""Exploratory timing of previously checked kernels, without dispatch checks.

This is not an acceptance benchmark or a production dispatch mode. It reads
the local comparison evidence, verifies each selected output once, then times
the reference, an unchecked buffer-adaptation/kernel/output path, and the
kernel alone. No candidates are installed or model calls made.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import awkward as ak

from agent_optimize.candidate import compile_kernel, contract_from_dict
from agent_optimize.evaluator import compare_results
from agent_optimize.serialization import load_value
from benchmarks.workloads import WORKLOADS


def measure(fn, iterations):
    started = time.perf_counter_ns()
    for _ in range(iterations):
        fn()
    return (time.perf_counter_ns() - started) / iterations


def calibrate(fn):
    count = 1
    while count < 4096:
        elapsed = measure(fn, count) * count
        if elapsed >= 10_000_000:
            break
        count *= 2
    return count


def run_comparison(comparison):
    results = []
    for row in comparison['results']:
        if not row['expected_supported_graph']:
            continue
        report = json.loads(Path(row['report_path']).read_text(encoding='utf-8'))
        for function in report['functions']:
            for proposal, evaluation in enumerate(function.get('evaluations', [])):
                if evaluation.get('correctness_status') != 'passed':
                    continue
                workspace = Path(evaluation['workspace'])
                source = (workspace / 'candidate.py').read_text(encoding='utf-8')
                if hashlib.sha256(source.encode()).hexdigest() != evaluation['candidate_sha256']:
                    raise ValueError('candidate changed after evaluation')
                contract = contract_from_dict(evaluation['contract'])
                job = json.loads((workspace / 'reference-job.json').read_text(encoding='utf-8'))
                for case_id in evaluation['recorded_case_ids']:
                    directory = workspace / 'cases' / case_id
                    metadata = json.loads((directory / 'inputs.json').read_text(encoding='utf-8'))
                    arguments = {name: load_value(value, directory) for name, value in metadata['arguments'].items()}
                    reference = WORKLOADS[row['case']].function.__wrapped__
                    positional = job['positional_names']
                    args = tuple(arguments[name] for name in positional)
                    kwargs = {name: value for name, value in arguments.items() if name not in positional}
                    array = arguments[contract.array_argument]
                    scalars = tuple(arguments[name] for name in contract.scalar_names)
                    kernel = compile_kernel(source)

                    # Assumes exactly the previously checked array layout. Unlike
                    # adapt_inputs, this deliberately does not validate it.
                    def inputs():
                        layout = array.layout
                        for field in contract.collection_path:
                            layout = layout.content(field)
                        return layout.offsets.data, tuple(layout.content.content(name).data for name in contract.fields), scalars

                    prepared = inputs()
                    def unchecked():
                        raw = kernel(*inputs())
                        return ak.Array(raw) if contract.output_kind == 'awkward' else raw

                    def reference_call():
                        return reference(*args, **kwargs)

                    def kernel_only():
                        return kernel(*prepared)

                    expected = reference_call()
                    started = time.perf_counter_ns()
                    actual = unchecked()  # includes first compilation
                    first_use_ns = time.perf_counter_ns() - started
                    equal, reason = compare_results(expected, actual)
                    if not equal:
                        raise ValueError(f'{row["case"]}: {reason}')
                    functions = {'reference': reference_call, 'unchecked': unchecked, 'kernel_only': kernel_only}
                    for fn in functions.values():
                        for _ in range(3):
                            fn()
                    counts = {name: calibrate(fn) for name, fn in functions.items()}
                    batches = {name: [] for name in functions}
                    for batch in range(3):
                        samples = {name: [] for name in functions}
                        for sample in range(7):
                            names = list(functions)
                            shift = (sample + batch) % len(names)
                            for name in names[shift:] + names[:shift]:
                                samples[name].append(measure(functions[name], counts[name]))
                        for name in functions:
                            batches[name].append(statistics.median(samples[name]))
                    results.append({
                        'model': row['model'], 'case': row['case'], 'proposal': proposal,
                        'candidate_sha256': evaluation['candidate_sha256'], 'events': len(array),
                        'case_id': case_id, 'correctness_checked_once': True,
                        'first_use_ns': first_use_ns, 'iterations': counts, 'batch_medians_ns': batches,
                        'medians_ns': {name: statistics.median(values) for name, values in batches.items()},
                    })
                    item = results[-1]
                    print(f"{item['model']} {item['case']} {len(array)} proposal {proposal}: "
                          f"reference={item['medians_ns']['reference']/1e6:.3f} ms, "
                          f"unchecked={item['medians_ns']['unchecked']/1e6:.3f} ms", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparisons', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    results = []
    for path in options.comparisons:
        results.extend(run_comparison(json.loads(path.read_text(encoding='utf-8'))))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps({'schema_version': 1, 'comparisons': [str(p) for p in options.comparisons], 'results': results}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
