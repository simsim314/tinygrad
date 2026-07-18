#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from extra.dfloat_attestation import verify_artifact


def main():
  parser=argparse.ArgumentParser(description="Verify deterministic DF attestation JSON chains and aggregates")
  parser.add_argument("artifact",type=Path)
  args=parser.parse_args()
  document=json.loads(args.artifact.read_text(encoding="utf-8"))
  verify_artifact(document)
  print(f"verified {args.artifact}")
  print(f"run_root {document['run_root']}")
  print(f"document_sha256 {document['document_sha256']}")


if __name__ == "__main__": main()
