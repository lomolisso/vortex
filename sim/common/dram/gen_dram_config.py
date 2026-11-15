#!/usr/bin/env python3
# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates vx_dram_config.h: the Ramulator 2.1 configuration of one DramSim memory system.

Ramulator 2.1's C++ loads only fully resolved configs (timing constraints are computed in
Python), so the config is built here with Ramulator's own Python components from the resolved
VX_config knobs (the -D flags of gen_config.py --cflags) and serialized with its exporter.
Needs PYTHONPATH=<ramulator>/python and Python >= 3.10.
"""

import argparse
import math
import shlex
import sys
from fractions import Fraction

import ramulator
from ramulator.components import Component
from ramulator.dram import HBM2, HBM3
from ramulator.export import dict_to_yaml
from ramulator.param import Param


class VxHBM3(ramulator.controller.HBM12):
    impl = "VxHBM3"
    rw_policy = Param(str, default="AGE")
    qd_rd = Param(int, default=0)
    qd_wr = Param(int, default=0)
    t_turn_ticks = Param(int, default=0)
    t_age_ticks = Param(int, default=0)
    cmd_buses = Param(int, default=1)
    pcs_per_cmd_bus = Param(int, default=1)
    occ_rd1_milliticks = Param(int, default=0)
    occ_rd2_milliticks = Param(int, default=0)
    occ_rd3_milliticks = Param(int, default=0)
    occ_rd4_milliticks = Param(int, default=0)
    occ_wr1_milliticks = Param(int, default=0)
    occ_wr2_milliticks = Param(int, default=0)
    occ_wr3_milliticks = Param(int, default=0)
    occ_wr4_milliticks = Param(int, default=0)


class VxHBM3Device(HBM3):
    """HBM3 whose command-bus occupancy the VxHBM3 controller keeps, one row and one column bus per
    64 bits (JESD238): Ramulator's occupancy is one bus pair per channel and caps a wider channel."""

    @classmethod
    def _generate_bus_constraints(cls, cmd_idx, cmd_cycles, tick_mult):
        return []


class VxClosedShared(Component):
    impl = "VxClosedShared"


class ConfigError(Exception):
    pass


def parse_cflags(text):
    knobs = {}
    for tok in shlex.split(text):
        if not tok.startswith("-D"):
            continue
        name, eq, value = tok[2:].partition("=")
        if not eq:
            knobs.setdefault(name, 1)
            continue
        try:
            knobs[name] = int(value, 0)
        except ValueError:
            knobs[name] = value
    return knobs


class Knobs:
    def __init__(self, values):
        self.values = values

    def __call__(self, name):
        key = "VX_CFG_" + name
        if key not in self.values:
            raise ConfigError(f"{key} is not in the resolved configuration")
        return self.values[key]


def require(cond, msg):
    if not cond:
        raise ConfigError(msg)


def log2_exact(x, what):
    require(x >= 1 and (x & (x - 1)) == 0, f"{what} = {x} must be a power of two")
    return x.bit_length() - 1


def cycles(ps, tck_ps):
    return -(-ps // tck_ps)


def hbm3_timings(k, tck_ps, bgs, banks_per_bg):
    """Every HBM3 timing of Ramulator 2.1, in CK, from the ps knobs at our tCK.

    2.1 resolves secondary timings at the preset's tCK before applying overrides, so each one is
    given here: from its knob, or from 2.1's own helper evaluated at our tCK and organization.
    """
    t = {
        "rate": k("DRAM_DATA_RATE_MTS"),
        "tCK_ps": tck_ps,
        "nBL": k("DRAM_NCK_BL"),
        "nCCDS": k("DRAM_NCK_CCDS"),
        "nPPD": k("DRAM_NCK_PPD"),
        "nRC": cycles(k("DRAM_T_RC_PS"), tck_ps),
        "nRP": cycles(k("DRAM_T_RP_PS"), tck_ps),
        "nRCDRD": cycles(k("DRAM_T_RCDRD_PS"), tck_ps),
        "nRCDWR": cycles(k("DRAM_T_RCDWR_PS"), tck_ps),
        "nCL": cycles(k("DRAM_T_CL_PS"), tck_ps),
        "nCWL": cycles(k("DRAM_T_CWL_PS"), tck_ps),
        "nWR": cycles(k("DRAM_T_WR_PS"), tck_ps),
        "nRTP": cycles(k("DRAM_T_RTP_PS"), tck_ps),
        "nRRDS": cycles(k("DRAM_T_RRDS_PS"), tck_ps),
        "nRRDL": cycles(k("DRAM_T_RRDL_PS"), tck_ps),
        "nFAW": cycles(k("DRAM_T_FAW_PS"), tck_ps),
        "nWTRS": cycles(k("DRAM_T_WTRS_PS"), tck_ps),
        "nWTRL": cycles(k("DRAM_T_WTRL_PS"), tck_ps),
        "nRFC": cycles(k("DRAM_T_RFC_PS"), tck_ps),
        "nRFCpb": cycles(k("DRAM_T_RFCPB_PS"), tck_ps),
        "nRREFD": cycles(k("DRAM_T_RREFD_PS"), tck_ps),
        # tREFI is a maximum interval: rounded down, as 2.1's _resolve_nREFI does.
        "nREFI": k("DRAM_T_REFI_PS") // tck_ps,
    }
    # A row cycle of exactly nRC: separate ceilings of tRAS and tRP could add one CK.
    t["nRAS"] = t["nRC"] - t["nRP"]
    t["nCCDL"] = t["nCCDS"]
    t["nCCDR"] = HBM3._resolve_nCCDR(1, t["nCCDS"])
    t["nRTW"] = HBM3._resolve_nRTW(t, tck_ps)
    t["nREFIpb"] = HBM3._resolve_nREFIpb(tck_ps, banks_per_bg, bgs, 1)
    t["nRFMab"] = t["nRFC"]
    t["nRFMpb"] = t["nRFCpb"]

    missing = set(HBM3.timing_params) - set(t)
    require(not missing, f"HBM3 timings without a value: {sorted(missing)}")
    require(t["nRAS"] >= t["nRCDRD"], f"nRAS {t['nRAS']} < nRCDRD {t['nRCDRD']}: tRC - tRP must cover tRCDRD")
    require(t["nRAS"] + t["nRP"] == t["nRC"], "nRAS + nRP != nRC")
    return t


def occupancy_milliticks(k, direction, sectors, atom_bytes):
    """Visit interval of a request of `sectors` atoms: the H100 per-slice interval scaled by 1/duty."""
    if direction == "rd":
        rate = k("H100_SLICE_RD_MBPS")
        floor_ps = k("H100_DRAM_VISIT_RD_PS")
    else:
        rate = k("H100_SLICE_WR_MBPS")
        floor_ps = k("H100_DRAM_VISIT_WR1_PS") if sectors == 1 else k("H100_DRAM_VISIT_WR2_PS")
    data_ps = Fraction(sectors * atom_bytes * 10**6, rate)
    occ_ps = max(Fraction(floor_ps), data_ps)
    scale = Fraction(k("DRAM_SERVICE_SCALE_NUM"), k("DRAM_SERVICE_SCALE_DEN"))
    mt = occ_ps * scale * k("DRAM_CLOCK_MHZ") / 1000
    return int(mt + Fraction(1, 2))


def build_h100(k):
    channels = k("DRAM_CHANNELS")
    parts = k("NUM_PARTITIONS")
    require(channels % parts == 0 and channels >= parts,
            f"DRAM_CHANNELS {channels} must be a multiple of NUM_PARTITIONS {parts}")
    per_ms = channels // parts
    log2_exact(per_ms, "channels per partition")

    dram_mhz = k("DRAM_CLOCK_MHZ")
    require(2 * dram_mhz == k("DRAM_DATA_RATE_MTS"), "DRAM_DATA_RATE_MTS must be 2 x DRAM_CLOCK_MHZ")
    atom = k("DRAM_ATOM_BYTES")
    require(atom == HBM3.data_payload_bytes, f"DRAM_ATOM_BYTES {atom} != HBM3 transaction {HBM3.data_payload_bytes} B")
    pcs, bgs, bpg = k("DRAM_PCS"), k("DRAM_BGS"), k("DRAM_BANKS_PER_BG")
    require(pcs * bgs * bpg == k("DRAM_BANKS"), "DRAM_PCS x DRAM_BGS x DRAM_BANKS_PER_BG != DRAM_BANKS")
    rows = k("DRAM_ROWS_PER_BANK")
    dq = HBM3.org_presets["HBM3_16Gb_4hi"]["dq"]
    column = k("DRAM_ROW_BYTES") * 8 // dq
    require(column // HBM3.internal_prefetch_size == k("DRAM_ROW_BYTES") // atom,
            "transactions per row != DRAM_ROW_BYTES / DRAM_ATOM_BYTES")

    # Ramulator ticks at the DRAM clock = half CK; tCK rounded up to whole ps.
    tck_ps = -(-2 * 10**6 // dram_mhz)
    t = hbm3_timings(k, tck_ps, bgs, bpg)

    dram = VxHBM3Device(org_preset="HBM3_16Gb_4hi", timing_preset="HBM3_6400Mbps",
                pseudochannel=pcs, sid=1, bankgroup=bgs, bank=bpg, row=rows, column=column,
                channel_width=k("DRAM_CHANNEL_BITS"), **t)

    sched = {"FRFCFS": ramulator.scheduler.FRFCFS, "ROWHIT": ramulator.scheduler.FRFCFSRowHit}[k("DRAM_SCHED")]
    refresh = {"PERBANK": ramulator.refresh_manager.HBM34PerBankRefresh,
               "ALLBANK": ramulator.refresh_manager.AllBank,
               "OFF": ramulator.refresh_manager.NoRefresh}[k("DRAM_REFRESH")]
    page = k("DRAM_PAGE_POLICY")
    if page == "SHARED":
        row_policy = VxClosedShared()
    elif page == "CAP":
        row_policy = ramulator.row_policy.ClosedCAP(cap=k("DRAM_LINE_BYTES") // atom)
    else:
        row_policy = ramulator.row_policy.Open()

    qd_rd, qd_wr = k("DRAM_QD_RD"), k("DRAM_QD_WR")
    require(qd_rd >= 1 and qd_wr >= 1, "DRAM_QD_RD and DRAM_QD_WR must be >= 1")
    max_k = k("DRAM_LINE_BYTES") // atom
    t_turn = cycles(k("DRAM_T_TURN_PS") * dram_mhz, 10**6)
    t_age = cycles(k("DRAM_T_AGE_PS") * dram_mhz, 10**6)
    pcs_per_bus = k("DRAM_PCS_PER_64B")
    require(pcs % pcs_per_bus == 0, "DRAM_PCS must be a multiple of DRAM_PCS_PER_64B")
    occ = {f"occ_{d}{n}_milliticks": occupancy_milliticks(k, d, n, atom) for d in ("rd", "wr") for n in range(1, 5)}

    def controller():
        # Transactions of started requests stay buffered until issued: room for a second queue's worth.
        return VxHBM3(read_buffer_size=2 * qd_rd * max_k, write_buffer_size=2 * qd_wr * max_k,
                      rw_policy=k("DRAM_RW_POLICY"), qd_rd=qd_rd, qd_wr=qd_wr,
                      t_turn_ticks=t_turn, t_age_ticks=t_age, **occ,
                      cmd_buses=pcs // pcs_per_bus, pcs_per_cmd_bus=pcs_per_bus,
                      scheduler=sched(), refresh_manager=refresh(), row_policy=row_policy,
                      addr_mapper=ramulator.addr_mapper.ChRaBaRoCo(), dram=dram)

    tx_per_row = column // HBM3.internal_prefetch_size
    intra_bits = (log2_exact(pcs, "DRAM_PCS") + log2_exact(bgs, "DRAM_BGS") + log2_exact(bpg, "DRAM_BANKS_PER_BG")
                  + log2_exact(rows, "DRAM_ROWS_PER_BANK") + log2_exact(tx_per_row, "transactions per row")
                  + log2_exact(atom, "DRAM_ATOM_BYTES"))
    ms = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=intra_bits - log2_exact(atom, "atom")),
        controllers=[controller() for _ in range(per_ms)])

    cfg = dram.to_config()
    ticks = {name: cfg["timing"][i] for i, name in enumerate(HBM3.timing_params)}
    info = {
        "model": "H100", "num_ms": parts, "channels_per_ms": per_ms,
        "pcs": pcs, "sids": 1, "bgs": bgs, "banks_per_bg": bpg, "rows": rows, "tx_per_row": tx_per_row,
        "tx_bytes": atom, "intra_bits": intra_bits,
        "read_latency": cfg["read_latency"], "write_done": ticks["nCWL"] + ticks["nBL"],
        "ticks": ticks, "ck": t, "tck_ps": tck_ps, "t_turn": t_turn, "t_age": t_age, "occ": occ,
    }
    return ms, info


def build_legacy(k):
    channels = k("DRAM_CHANNELS")
    log2_exact(channels, "DRAM_CHANNELS")
    dram = HBM2(org_preset="HBM2_8Gb", timing_preset="HBM2_2000Mbps")
    org = HBM2.org_presets["HBM2_8Gb"]

    def controller():
        return ramulator.controller.HBM12(scheduler=ramulator.scheduler.FRFCFS(),
                                          refresh_manager=ramulator.refresh_manager.AllBank(),
                                          row_policy=ramulator.row_policy.Open(),
                                          addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(), dram=dram)

    tx = HBM2.data_payload_bytes
    tx_per_row = org["column"] // HBM2.internal_prefetch_size
    intra_bits = sum(log2_exact(org[n], n) for n in ("pseudochannel", "sid", "bankgroup", "bank", "row")) \
        + log2_exact(tx_per_row, "transactions per row") + log2_exact(tx, "tx bytes")
    ms = ramulator.memory_system.GenericDRAM(
        clock_ratio=1,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(interleave_bits=intra_bits - log2_exact(tx, "tx")),
        controllers=[controller() for _ in range(channels)])

    cfg = dram.to_config()
    ticks = {name: cfg["timing"][i] for i, name in enumerate(HBM2.timing_params)}
    info = {
        "model": "HBM2", "num_ms": 1, "channels_per_ms": channels,
        "pcs": org["pseudochannel"], "sids": org["sid"], "bgs": org["bankgroup"], "banks_per_bg": org["bank"],
        "rows": org["row"], "tx_per_row": tx_per_row, "tx_bytes": tx, "intra_bits": intra_bits,
        "read_latency": cfg["read_latency"], "write_done": ticks["nCWL"] + ticks["nBL"],
        "ticks": ticks, "ck": None, "tck_ps": ticks["tCK_ps"], "t_turn": 0, "t_age": 0, "occ": {},
    }
    return ms, info


def emit(ms, info, cflags_text):
    yaml = dict_to_yaml({"frontend": ramulator.frontend.External(clock_ratio=1).to_config(),
                         "memory_system": ms.to_config()})
    out = []
    out.append("// Generated by sim/common/dram/gen_dram_config.py; do not edit.")
    out.append("#pragma once")
    out.append("")
    out.append(f"#define VX_DRAM_MODEL_{info['model']} 1")
    out.append(f"#define VX_DRAM_NUM_MS {info['num_ms']}")
    out.append(f"#define VX_DRAM_CHANNELS_PER_MS {info['channels_per_ms']}")
    out.append(f"#define VX_DRAM_PCS {info['pcs']}")
    out.append(f"#define VX_DRAM_SIDS {info['sids']}")
    out.append(f"#define VX_DRAM_BGS {info['bgs']}")
    out.append(f"#define VX_DRAM_BANKS_PER_BG {info['banks_per_bg']}")
    out.append(f"#define VX_DRAM_ROWS {info['rows']}")
    out.append(f"#define VX_DRAM_TX_PER_ROW {info['tx_per_row']}")
    out.append(f"#define VX_DRAM_TX_BYTES {info['tx_bytes']}")
    out.append(f"#define VX_DRAM_INTRA_CHANNEL_BITS {info['intra_bits']}")
    out.append(f"#define VX_DRAM_READ_LATENCY_TICKS {info['read_latency']}")
    out.append(f"#define VX_DRAM_WRITE_DONE_TICKS {info['write_done']}")
    out.append(f"#define VX_DRAM_T_TURN_TICKS {info['t_turn']}")
    out.append(f"#define VX_DRAM_T_AGE_TICKS {info['t_age']}")
    for name, value in info["occ"].items():
        out.append(f"#define VX_DRAM_{name.upper()} {value}")
    out.append("")
    out.append(f"// Timings in Ramulator ticks (one tick = one DRAM clock; tCK = {info['tck_ps']} ps).")
    for name, value in info["ticks"].items():
        if name in ("rate", "tCK_ps"):
            continue
        out.append(f"#define VX_DRAM_{name.upper()}_TICKS {value}")
    if info["ck"] is not None:
        out.append("")
        out.append("// Review: every HBM3 timing in CK (tick = CK / 2) and its source.")
        sources = {
            "nRAS": "nRC - nRP", "nCCDL": "nCCDS (bank-group neutral)", "nCCDR": "2.1 _resolve_nCCDR, 1 SID",
            "nRTW": "2.1 _resolve_nRTW at our tCK", "nREFIpb": "2.1 _resolve_nREFIpb at our tCK and org",
            "nRFMab": "nRFC", "nRFMpb": "nRFCpb", "nREFI": "T_REFI_PS rounded down",
            "nBL": "NCK_BL", "nCCDS": "NCK_CCDS", "nPPD": "NCK_PPD",
        }
        for name in HBM3.timing_params:
            if name in ("rate", "tCK_ps"):
                continue
            src = sources.get(name, "ps knob rounded up")
            out.append(f"//   {name:8s} {info['ck'][name]:6d} CK  {src}")
    out.append("")
    out.append("#define VX_DRAM_RAMULATOR_YAML \\")
    lines = yaml.splitlines()
    for i, line in enumerate(lines):
        esc = line.replace("\\", "\\\\").replace('"', '\\"')
        tail = " \\" if i + 1 < len(lines) else ""
        out.append(f'  "{esc}\\n"{tail}')
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cflags", required=True, help="resolved -D flags (gen_config.py --cflags output)")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    k = Knobs(parse_cflags(args.cflags))
    try:
        model = k("DRAM_MODEL")
        if model == "H100":
            ms, info = build_h100(k)
        elif model == "HBM2":
            ms, info = build_legacy(k)
        else:
            raise ConfigError(f"unknown VX_CFG_DRAM_MODEL {model}")
        text = emit(ms, info, args.cflags)
    except (ConfigError, ValueError) as e:
        print(f"gen_dram_config.py: error: {e}", file=sys.stderr)
        return 1
    with open(args.output, "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
