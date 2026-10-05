"""Compare the actual phase instructions, including implicit baseline defaults."""

import json

from baseline.grasp import PHASES, phase_prompt


def prompt_identity(request):
    overrides = request.get("phase_prompts") or {}
    texts = [overrides.get(phase) or phase_prompt(request.get("prompt"), phase,
                                               request.get("prompt_profile", "focused"))
             for phase, _ in PHASES]
    # Formatting and capitalization do not create a new held-out instruction.
    return json.dumps([" ".join(text.split()).casefold() for text in texts])


def own_prompt(owners, request, split):
    identity = prompt_identity(request)
    if identity in owners and owners[identity] != split:
        raise ValueError("Effective prompt instructions leak across parent splits")
    owners[identity] = split
    return identity
