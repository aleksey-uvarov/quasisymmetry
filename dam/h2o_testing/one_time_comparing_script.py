import re
from pathlib import Path

def parse_file(path):
    text = Path(path).read_text()
    blocks = text.split("Input:")[1:]  # drop header before first "Input:"
    records = {}
    for block in blocks:
        params = dict(
            re.findall(r"(\w+)\s*=\s*([^\s,]+)", block.split("\n")[0])
        )
        key = (
            params["example_cluster_matrices_index"],
            params["bond_length"],
            params["hoh_angle_deg"],
        )
        parity_match = re.search(r"numbers or parities:\s*(\S+)", block)
        k_sectors_match = re.search(r"K_sectors\s*=\s*(\d+)", block)
        k_states_match = re.search(r"K_states\s*=\s*(\d+)", block)

        records[key] = {
            "parity": parity_match.group(1),
            "K_sectors": int(k_sectors_match.group(1)),
            "K_states": int(k_states_match.group(1)),
        }
    return records


variance = parse_file("variance_results.txt")
eval_eq = parse_file("eval_eq_results.txt")

common_keys = set(variance) & set(eval_eq)
missing_v = set(eval_eq) - set(variance)
missing_e = set(variance) - set(eval_eq)
if missing_v or missing_e:
    print(f"Warning: {len(missing_v)} keys only in eval_eq, {len(missing_e)} keys only in variance")

parities = ["both", "N", "P"]

main_counts = {p: {"win": 0, "lose": 0} for p in parities}

other_labels = [
    "sectors: variance wins, states: eval_eq wins",
    "sectors: eval_eq wins, states: variance wins",
    "sectors: tie, states: tie",
    "sectors: variance wins, states: tie",
    "sectors: eval_eq wins, states: tie",
    "sectors: tie, states: variance wins",
    "sectors: tie, states: eval_eq wins",
]
other_counts = {p: {label: 0 for label in other_labels} for p in parities}


def classify_other(sectors_cmp, states_cmp):
    if sectors_cmp > 0 and states_cmp < 0:
        return other_labels[0]
    if sectors_cmp < 0 and states_cmp > 0:
        return other_labels[1]
    if sectors_cmp == 0 and states_cmp == 0:
        return other_labels[2]
    if sectors_cmp > 0 and states_cmp == 0:
        return other_labels[3]
    if sectors_cmp < 0 and states_cmp == 0:
        return other_labels[4]
    if sectors_cmp == 0 and states_cmp > 0:
        return other_labels[5]
    if sectors_cmp == 0 and states_cmp < 0:
        return other_labels[6]


for key in common_keys:
    v = variance[key]
    e = eval_eq[key]
    parity = v["parity"]

    # Inverted: positive now means variance is LOWER (i.e. variance wins)
    sectors_cmp = (v["K_sectors"] < e["K_sectors"]) - (v["K_sectors"] > e["K_sectors"])
    states_cmp = (v["K_states"] < e["K_states"]) - (v["K_states"] > e["K_states"])

    if sectors_cmp > 0 and states_cmp > 0:
        main_counts[parity]["win"] += 1
    elif sectors_cmp < 0 and states_cmp < 0:
        main_counts[parity]["lose"] += 1
    else:
        label = classify_other(sectors_cmp, states_cmp)
        other_counts[parity][label] += 1


for parity in parities:
    print(f"\n--- number, parity, or both = {parity} ---")
    print(f"variance wins both (lower K): {main_counts[parity]['win']}")
    print(f"eval_eq wins both (lower K): {main_counts[parity]['lose']}")
    for label in other_labels:
        print(f"{label}: {other_counts[parity][label]}")