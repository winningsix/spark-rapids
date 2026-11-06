#!/bin/bash
# 1 Hz DCGM sampling with an epoch prefix on every line, for analyze-dcgm-v1.py.
# Fields: 203 GPU util, 1002 SM active, 1003 SM occupancy, 1005 DRAM active,
#         1009/1010 PCIe tx/rx bytes, 155 power.
# Usage: dcgm-logger-v1.sh <output.txt>   (stop with: pkill -f 'dcgmi dmon -e 203')
OUT=${1:?usage: $0 <output.txt>}
dcgmi dmon -e 203,1002,1003,1005,1009,1010,155 -d 1000 | while IFS= read -r l; do
  printf '%s %s\n' "$(date +%s.%N)" "$l"
done >> "$OUT"
