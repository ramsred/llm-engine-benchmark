#!/usr/bin/env python3
"""Opt-in context characterization; does not invoke the canonical preparation pipeline."""

from llm_engine_benchmark.context_scaling import main

if __name__ == "__main__":
    raise SystemExit(main())
