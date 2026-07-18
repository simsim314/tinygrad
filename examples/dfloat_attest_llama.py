#!/usr/bin/env python3
"""Generate a greedy Llama response and checkpoint a deterministic DF attestation."""
from __future__ import annotations

import argparse, hashlib, io, json, os, sys, time, typing
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if not hasattr(typing,"Self"):
  from typing_extensions import Self
  setattr(typing,"Self",Self)

# The GGUF loading path consults Device.DEFAULT during tinygrad import. Apply an
# explicit CLI device early so model weights and input tensors cannot diverge.
if "--cpu" in sys.argv: os.environ["DEV"]="CPU"
for index,argument in enumerate(sys.argv):
  if argument == "--device" and index+1 < len(sys.argv): os.environ["DEV"]=sys.argv[index+1]
  elif argument.startswith("--device="): os.environ["DEV"]=argument.split("=",1)[1]

from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import get_state_dict

from examples.llama3 import Tokenizer, build_transformer
from extra.dfloat_attestation import AttestationSession, verify_artifact
from extra.dfloat_attestation_fast import FastAttestationSession
from extra.dfloat_attested_llama import (attest_dense_llama_forward,attest_greedy_selection,
  fast_attest_dense_llama_forward,fast_attest_greedy_selection)


def file_sha256(path:Path) -> str:
  digest=hashlib.sha256()
  with path.open("rb") as stream:
    while chunk:=stream.read(io.DEFAULT_BUFFER_SIZE): digest.update(chunk)
  return digest.hexdigest()


def atomic_write(path:Path, data:str):
  temporary=path.with_suffix(path.suffix+".tmp")
  temporary.write_text(data,encoding="utf-8")
  os.replace(temporary,path)


def save_checkpoint(output_dir:Path, session:AttestationSession, metadata:dict[str,object], generated_text:str):
  artifact=session.artifact(metadata,generated_text)
  verify_artifact(artifact)
  atomic_write(output_dir/"attestation.json",json.dumps(artifact,sort_keys=True,separators=(",",":"))+"\n")
  atomic_write(output_dir/"attestation.txt",session.text_artifact(artifact))
  atomic_write(output_dir/"story.txt",generated_text+"\n")
  return artifact


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model",type=Path,required=True)
  parser.add_argument("--tokenizer",type=Path)
  parser.add_argument("--prompt",default="tell me a story about paris")
  parser.add_argument("--max-tokens",type=int,default=64)
  parser.add_argument("--output-dir",type=Path,required=True)
  parser.add_argument("--fast-attestation",action="store_true",help="Attest each transformer-layer output with GPU XOR/SHA-256")
  parser.add_argument("--no-layer-snapshot-copy",action="store_true",help="Retain layer outputs directly instead of lazy immutable clones")
  device=parser.add_mutually_exclusive_group()
  device.add_argument("--device",dest="device")
  device.add_argument("--cpu",action="store_const",const="CPU",dest="device")
  parser.set_defaults(device=Device.DEFAULT)
  args=parser.parse_args()
  if args.max_tokens < 1: raise ValueError("--max-tokens must be positive")
  tokenizer_path=args.tokenizer or args.model.parent/"tokenizer.model"
  tokenizer=Tokenizer(str(tokenizer_path))
  prompt_tokens=[tokenizer.bos_id,*tokenizer.encode(args.prompt)]
  required_context=len(prompt_tokens)+args.max_tokens

  device_option="--cpu" if args.device == "CPU" else f"--device {args.device}"
  attestation_option=" --fast-attestation" if args.fast_attestation else ""
  snapshot_option=" --no-layer-snapshot-copy" if args.no_layer_snapshot_copy else ""
  command=(f"python examples/dfloat_attest_llama.py {device_option}{attestation_option}{snapshot_option} --model {args.model} "
           f"--tokenizer {tokenizer_path} --prompt {json.dumps(args.prompt)} --max-tokens {args.max_tokens} "
           f"--output-dir {args.output_dir}")
  print(f"model: {args.model}",flush=True)
  print(f"output: {args.output_dir}",flush=True)
  print("loading canonical FP16 model",flush=True)
  model=build_transformer(args.model,model_size="1B",device=args.device,max_context=required_context,
                          dfloat=True,low_memory=True)
  persistent=get_state_dict(model)
  storage={name:value.dtype for name,value in persistent.items() if name != "freqs_cis" and "cache_kv" not in name}
  unexpected={name:str(dtype) for name,dtype in storage.items() if dtype != dtypes.float16}
  if unexpected: raise RuntimeError(f"non-FP16 persistent model storage: {unexpected}")
  wrong_device={name:value.device for name,value in persistent.items() if value.device != args.device}
  if wrong_device: raise RuntimeError(f"persistent tensors not on requested {args.device} device: {wrong_device}")
  metadata:dict[str,object]={"model_file":args.model.name,"model_file_sha256":file_sha256(args.model),
    "tokenizer_file":tokenizer_path.name,"tokenizer_sha256":file_sha256(tokenizer_path),
    "prompt":args.prompt,"prompt_tokens":prompt_tokens,"max_generated_tokens":args.max_tokens,
    "selection":"greedy","arithmetic":"DF16-DF32-V1","storage":"canonical-fp16-v1",
    "attestation":"transformer-layer-output-xor-sha256-v3" if args.fast_attestation else "full-merkle-v1",
    "layer_snapshot_copy":bool(args.fast_attestation and not args.no_layer_snapshot_copy)}

  session=FastAttestationSession() if args.fast_attestation else AttestationSession()
  forward=fast_attest_dense_llama_forward if args.fast_attestation else attest_dense_llama_forward
  select=fast_attest_greedy_selection if args.fast_attestation else attest_greedy_selection
  selected_token=None
  generated_tokens:list[int]=[]
  started=time.monotonic()
  # Prompt tokens are teacher-forced inputs. Their logits and predicted tokens are still witnessed;
  # the complete actual prompt token list is separately bound by metadata.
  for position,input_token in enumerate(prompt_tokens):
    print(f"attesting prompt {position+1}/{len(prompt_tokens)} position={position}",flush=True)
    logits,recorder=forward(model,Tensor([[input_token]],dtype=dtypes.int32,device=args.device),
                            step=position,start_pos=position,session=session,
                            **({"snapshot_copy":not args.no_layer_snapshot_copy} if args.fast_attestation else {}))
    state=b"prompt_teacher_forced" if position+1 < len(prompt_tokens) else b"generation"
    if args.fast_attestation: selected_token=select(logits,recorder)
    else:
      candidate=int(logits[:,-1,:].argmax().item())
      selected_token=select(logits,recorder,emitted_token_bytes=tokenizer.decode([candidate]).encode(),text_stop_state=state)
    if args.fast_attestation:
      print(f"timing step={position} model_graph_and_trace={recorder.graph_snapshot_seconds:.4f}s "
            f"layer_xor_sha256={recorder.layer_attestation_seconds:.4f}s",flush=True)

  assert selected_token is not None
  position=len(prompt_tokens)
  while len(generated_tokens) < args.max_tokens:
    generated_tokens.append(selected_token)
    generated_text=tokenizer.decode(generated_tokens)
    print(f"generated {len(generated_tokens)}/{args.max_tokens}: token={selected_token} "
          f"elapsed={time.monotonic()-started:.1f}s",flush=True)
    print(generated_text,flush=True)
    if selected_token in tokenizer.stop_tokens or len(generated_tokens) >= args.max_tokens: break
    logits,recorder=forward(model,Tensor([[selected_token]],dtype=dtypes.int32,device=args.device),
                            step=position,start_pos=position,session=session,
                            **({"snapshot_copy":not args.no_layer_snapshot_copy} if args.fast_attestation else {}))
    if args.fast_attestation: selected_token=select(logits,recorder)
    else:
      candidate=int(logits[:,-1,:].argmax().item())
      stop_state=b"stop" if candidate in tokenizer.stop_tokens else b"continue"
      selected_token=select(logits,recorder,emitted_token_bytes=tokenizer.decode([candidate]).encode(),text_stop_state=stop_state)
    if args.fast_attestation:
      print(f"timing step={position} model_graph_and_trace={recorder.graph_snapshot_seconds:.4f}s "
            f"layer_xor_sha256={recorder.layer_attestation_seconds:.4f}s",flush=True)
    position+=1

  # Generation is side-effect free: export device roots and write all files once, after the final token.
  args.output_dir.mkdir(parents=True,exist_ok=True)
  artifact=save_checkpoint(args.output_dir,session,metadata,generated_text)
  atomic_write(args.output_dir/"command.txt",command+"\n")
  print(f"saved {args.output_dir/'attestation.json'}",flush=True)
  print(f"document_sha256 {artifact['document_sha256']}",flush=True)


if __name__ == "__main__": main()
