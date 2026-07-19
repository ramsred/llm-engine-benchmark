from __future__ import annotations

import random
import string
from typing import Any, Iterable


RULER_GENERATOR_VERSION = 1


_WORDS = (
    "amber birch cedar delta ember falcon garden harbor island juniper kernel lantern "
    "meadow nectar orbit pebble quartz river saffron timber upland velvet willow xenon "
    "yarrow zephyr atlas beacon copper drift echo fern glacier hollow ivory jasmine "
    "keystone lagoon maple north opal prairie quill ridge summit thicket umber valley "
    "wander xylem yellow zenith acorn brook canyon dune elm forest grove heather inlet "
    "jade knoll lichen moss night ocean pine quest reed stone tide union vista wheat "
    "yearling zodiac apricot bridge cloud dawn engine field granite horizon ink jet "
    "kestrel lake mineral novel olive path query rain signal tower unit vector wind "
    "axis bloom circuit document element frame graph history index junction key logic "
    "matrix node object pattern queue record schema thread update value window yield zone"
).split()


def generate_ruler_records(seed: int) -> list[dict[str, Any]]:
    """Generate 40 deterministic, neutral RULER-style canonical records.

    RULER is itself a configurable synthetic benchmark. These records preserve
    its four task families while avoiding any runtime-specific chat template.
    The upstream repository revision is captured separately in the lock file.
    """

    records: list[dict[str, Any]] = []
    for index in range(10):
        records.append(_needle_record(seed, index))
    for index in range(10):
        records.append(_variable_record(seed, index))
    for index in range(10):
        records.append(_aggregation_record(seed, index))
    for index in range(10):
        records.append(_qa_record(seed, index))
    return records


def _noise_lines(rng: random.Random, count: int, *, prefix: str) -> list[str]:
    lines: list[str] = []
    for line_index in range(count):
        words = rng.sample(_WORDS, k=10)
        serial = rng.randrange(10_000_000, 99_999_999)
        lines.append(
            f"{prefix}-{line_index:04d}: {' '.join(words)}; observation={serial}."
        )
    return lines


def _needle_record(seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(seed * 100_003 + index)
    key = f"KEY-{index:02d}-{_token(rng, 8)}"
    value = f"VALUE-{rng.randrange(1000000, 9999999)}"
    lines = _noise_lines(rng, 700, prefix=f"R{index:02d}")
    insertion = rng.randrange(50, len(lines) - 50)
    lines.insert(
        insertion,
        f"Authoritative registry entry: the value associated with {key} is {value}.",
    )
    instruction = (
        "Using only the registry above, return the exact value associated with "
        f"{key}. Return only the value."
    )
    return _record(
        sample_id=f"ruler-retrieval-{index:04d}",
        task="needle_retrieval",
        context="\n".join(lines),
        instruction=instruction,
        answer=value,
        metadata={"category": "needle/retrieval", "needle_position": insertion},
    )


def _variable_record(seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(seed * 200_003 + index)
    root_value = f"ROOT-{rng.randrange(100000, 999999)}"
    variables = [f"var_{index}_{letter}" for letter in string.ascii_lowercase[:9]]
    chain = [f"{variables[0]} := {root_value}"]
    for previous, current in zip(variables, variables[1:]):
        chain.append(f"{current} := value_of({previous})")

    lines = _noise_lines(rng, 620, prefix=f"V{index:02d}")
    positions = sorted(rng.sample(range(20, len(lines) - 20), k=len(chain)))
    for offset, (position, statement) in enumerate(zip(positions, chain)):
        lines.insert(position + offset, f"Binding statement: {statement}.")
    instruction = (
        f"Trace the binding chain and return the final literal value of {variables[-1]}. "
        "Return only the literal value."
    )
    return _record(
        sample_id=f"ruler-multihop-{index:04d}",
        task="variable_tracking",
        context="\n".join(lines),
        instruction=instruction,
        answer=root_value,
        metadata={"category": "variable tracking or multi-hop", "hops": len(chain) - 1},
    )


def _aggregation_record(seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(seed * 300_007 + index)
    targets = rng.sample(_WORDS, k=4)
    counts = {
        targets[0]: 71 + index,
        targets[1]: 59 + index,
        targets[2]: 47 + index,
        targets[3]: 31 + index,
    }
    tokens: list[str] = []
    for word, count in counts.items():
        tokens.extend([word] * count)
    noise_words = [word for word in _WORDS if word not in counts]
    for _ in range(1600):
        tokens.append(rng.choice(noise_words))
    rng.shuffle(tokens)
    lines = [" ".join(tokens[offset : offset + 24]) for offset in range(0, len(tokens), 24)]
    instruction = (
        "Count the occurrences of the four tracked words in the corpus: "
        + ", ".join(targets)
        + ". Return the tracked word with the highest count. Return only that word."
    )
    answer = max(counts, key=counts.get)
    return _record(
        sample_id=f"ruler-aggregation-{index:04d}",
        task="aggregation_counting",
        context="\n".join(lines),
        instruction=instruction,
        answer=answer,
        metadata={"category": "aggregation/counting", "tracked_counts": counts},
    )


def _qa_record(seed: int, index: int) -> dict[str, Any]:
    rng = random.Random(seed * 400_009 + index)
    entity = f"Project {_token(rng, 7).title()}"
    launch_year = rng.randrange(1980, 2025)
    location = rng.choice(
        ["North Harbor", "Cedar Valley", "Quartz Ridge", "Amber Coast", "Willow Basin"]
    )
    paragraphs = _noise_lines(rng, 520, prefix=f"Q{index:02d}")
    fact = (
        f"Verified archive note: {entity} began operation in {launch_year} at {location}. "
        "This note supersedes all speculative mentions."
    )
    paragraphs.insert(rng.randrange(40, 480), fact)
    instruction = (
        f"According to the verified archive note, in what year did {entity} begin operation? "
        "Return only the four-digit year."
    )
    return _record(
        sample_id=f"ruler-qa-{index:04d}",
        task="synthetic_qa",
        context="\n".join(paragraphs),
        instruction=instruction,
        answer=str(launch_year),
        metadata={"category": "QA", "entity": entity, "location": location},
    )


def _record(
    *,
    sample_id: str,
    task: str,
    context: str,
    instruction: str,
    answer: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source": "ruler",
        "task": task,
        "context": context,
        "instruction": instruction,
        "answer": answer,
        "metadata": {
            **metadata,
            "generator": "neutral_builtin_ruler_task_families",
        },
    }


def _token(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))
