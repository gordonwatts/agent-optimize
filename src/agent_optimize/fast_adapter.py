"""Host-written, unchecked input/output code stored beside each kernel."""


def adapter_source(contract):
    lines = ["import awkward as ak", "", "def prepare(arguments):", f"    layout = arguments[{contract['array_argument']!r}].layout"]
    for field in contract["collection_path"]:
        lines.append(f"    layout = layout.content({field!r})")
    columns = ", ".join(f"layout.content.content({name!r}).data" for name in contract["fields"])
    for path in contract.get("event_fields", ()):
        access = f"arguments[{contract['array_argument']!r}].layout"
        for field in path.split("."):
            access += f".content({field!r})"
        columns += f", {access}.data"
    scalars = ", ".join(f"arguments[{name!r}]" for name in contract["scalar_names"])
    lines += [f"    return layout.offsets.data, ({columns},), ({scalars}{',' if scalars else ''})", "", "def replacement(arguments, kernel):", "    raw = kernel(*prepare(arguments))"]
    lines.append("    return ak.Array(raw)" if contract["output_kind"] == "awkward" else "    return raw")
    return "\n".join(lines) + "\n"
