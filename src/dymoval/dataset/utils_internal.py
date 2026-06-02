from typing import Any


def list_to_structured_tuple(
    tpl: list[tuple[str, ...]],
    lst: list[Any],
) -> list[tuple[Any, ...]]:
    result = []
    idx = 0

    for group in tpl:
        size = len(group)
        result.append(tuple(lst[idx : idx + size]))
        idx += size

    return result
