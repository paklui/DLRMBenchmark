#!/usr/bin/env python3
"""patch_fbgemm_rocmsetup.py <RocmSetup.cmake> — FBGEMM d651cb36 lists two HIP warning flags as quoted strings (" \"-Wno-#pragma-messages\" ").
With clang as the host C++ compiler (needed on ROCm 10.3, see build_dlrm_rocm103.sh) they reach clang++ with literal quotes and are taken as file names.
Replace them by the plain flags (they only silence warnings)."""
import sys
p = sys.argv[1]; s = open(p).read()
for old, new in [('" \\"-Wno-#pragma-messages\\" "', "-Wno-#pragma-messages"), ('" \\"-Wno-#warnings\\" "', "-Wno-#warnings")]:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
open(p, "w").write(s); print("patched", p)
