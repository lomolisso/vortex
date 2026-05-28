#!/usr/bin/env python3
"""
toml_get.py — print a single scalar value out of a TOML file.

Usage: toml_get.py <path/to/file.toml> <dotted.key>
       e.g. toml_get.py fpu.toml clock.target_freq_mhz

Used by hw/syn/synopsys/Makefile to read PnR TOMLs as the single source of
truth for shared knobs (target_freq_mhz, max_fanout, etc.) without
duplicating values across flows.
"""
import sys
import toml


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: toml_get.py <file.toml> <dotted.key>")
    path, dotted = sys.argv[1], sys.argv[2]
    val = toml.load(path)
    for key in dotted.split('.'):
        if key not in val:
            sys.exit("ERROR: key '{}' not found in {}".format(dotted, path))
        val = val[key]
    print(val)


if __name__ == '__main__':
    main()
