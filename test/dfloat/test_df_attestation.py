import hashlib, struct, unittest

from extra.dfloat_attestation import (AttestationRecorder, ModuleChain, TensorDType, TensorRole, boundary_root, canonical_json_bytes,
  canonical_tensor_header, document_root, ordered_root, sha256_frame, tensor_commitment, tensor_merkle_root, token_root,
  verify_artifact, weight_commitment, xor_roots, ZERO_SHA256)
from extra.dfloat_attestation_schema import (LlamaV1Config, attention_scores_plan, embedding_plan, llama_block_plan,
  matmul_plan, reduction_plan, rmsnorm_plan, schema_root, softmax_plan, token_selection_plan)


class TestDFAttestationCore(unittest.TestCase):
  def test_framing_is_unambiguous_and_standard_sha256(self):
    self.assertNotEqual(sha256_frame("TEST", (b"ab", b"c")), sha256_frame("TEST", (b"a", b"bc")))
    wire = b"TEST\0" + struct.pack("<I", 2) + struct.pack("<Q", 2) + b"ab" + struct.pack("<Q", 1) + b"c"
    self.assertEqual(sha256_frame("TEST", (b"ab", b"c")), hashlib.sha256(wire).digest())
    self.assertEqual(sha256_frame("TEST", (b"ab", b"c")).hex(),"cd191a182e14b5975d12097dc9d904fd9440aafa789306607482c560cc4550da")

  def test_tensor_header_is_little_endian_and_unicode_nfc(self):
    a = canonical_tensor_header(step=0x0102030405060708, boundary=0x090a0b0c, role=TensorRole.OUTPUT,
                                name="cafe\u0301", dtype=TensorDType.DF16, shape=(2,3), byte_length=24)
    b = canonical_tensor_header(step=0x0102030405060708, boundary=0x090a0b0c, role=TensorRole.OUTPUT,
                                name="caf\u00e9", dtype=TensorDType.DF16, shape=(2,3), byte_length=24)
    self.assertEqual(a,b)
    self.assertIn(struct.pack("<Q", 0x0102030405060708), a)
    self.assertIn(struct.pack("<I", 0x090a0b0c), a)

  def test_merkle_empty_single_odd_and_multichunk(self):
    header=b"pinned-header"
    roots=[tensor_merkle_root(header, data) for data in (b"", b"a", bytes(4097), bytes(8193))]
    self.assertEqual([x.hex() for x in roots],[
      "f4703ebc9cc3e570b2776159e4a19811d39b019a284880a73d6ef56b0428aa4e",
      "18a80ee1d49ae144a546e4a16a6ba86d2c7543cd5844d38c1e446aac04232c78",
      "c4f1af04fd43ab65585480f4e481a5a47ebbf600fd1883371f6375a7c8c88061",
      "90325152d1c723163315201fe19607f591d8e41ced630bf95cf8b8b68bbd2b9f"])
    self.assertEqual(len(set(roots)), 4)
    self.assertTrue(all(len(x) == 32 for x in roots))
    # Odd nodes use an all-zero right digest, never duplicate the left digest.
    three=tensor_merkle_root(header, bytes(8193))
    duplicated_last=tensor_merkle_root(header, bytes(8193)+bytes(4095))
    self.assertNotEqual(three,duplicated_last)

  def test_tensor_metadata_and_weight_domain_are_committed(self):
    args=dict(step=3,boundary=7,role=TensorRole.OUTPUT,name="layer.0.out",dtype=TensorDType.DF16,shape=(2,),data=b"\0"*8)
    root=tensor_commitment(**args)
    self.assertNotEqual(root,tensor_commitment(**{**args,"name":"layer.1.out"}))
    self.assertNotEqual(root,tensor_commitment(**{**args,"shape":(1,2)}))
    self.assertNotEqual(root,weight_commitment(name="layer.0.out",dtype=TensorDType.DF16,shape=(2,),data=b"\0"*8))
    self.assertEqual(root.hex(),"0a39b8dab0df7c7b7e940789a7c05b4153b4e26584927b7acd681894c7ea4232")
    with self.assertRaises(ValueError): tensor_commitment(**{**args,"data":b"\0"*7})

  def test_module_chain_commits_order_and_previous_sha(self):
    spec=hashlib.sha256(b"spec").digest()
    r1,r2=hashlib.sha256(b"one").digest(),hashlib.sha256(b"two").digest()
    a=ModuleChain(5,"layer.0.linear",spec,(r1,))
    a.add("frontier_0",r1)
    a.add("output",r2)
    b=ModuleChain(5,"layer.0.linear",spec,(r1,))
    b.add("output",r2)
    b.add("frontier_0",r1)
    self.assertNotEqual(a.finish((r2,)),b.finish((r2,)))
    self.assertEqual(a.witnesses[1].index,1)
    self.assertNotEqual(a.witnesses[0].chain,a.witnesses[1].chain)

  def test_ordered_and_xor_aggregates(self):
    roots=(hashlib.sha256(b"a").digest(),hashlib.sha256(b"b").digest())
    self.assertNotEqual(ordered_root("ORDER",roots),ordered_root("ORDER",roots[::-1]))
    self.assertEqual(ordered_root("ORDER",roots),sha256_frame("ORDER",(struct.pack("<I",2)+b"".join(roots),)))
    self.assertEqual(xor_roots(roots),xor_roots(roots[::-1]))
    self.assertEqual(xor_roots((roots[0],roots[0])),ZERO_SHA256)

  def test_boundary_and_token_order(self):
    roots=[hashlib.sha256(bytes([x])).digest() for x in range(4)]
    b0=boundary_root(index=0,name="a",input_roots=roots[:1],witness_roots=roots[1:3],output_root=roots[3])
    b1=boundary_root(index=1,name="b",input_roots=roots[:1],witness_roots=roots[1:3],output_root=roots[3])
    ordered,xor,combined=token_root(previous=ZERO_SHA256,position=9,input_token=11,selected_token=12,boundary_roots=(b0,b1))
    ordered2,xor2,combined2=token_root(previous=ZERO_SHA256,position=9,input_token=11,selected_token=12,boundary_roots=(b1,b0))
    self.assertNotEqual(ordered,ordered2)
    self.assertEqual(xor,xor2)
    self.assertNotEqual(combined,combined2)

  def test_canonical_json_and_document_hash(self):
    self.assertEqual(canonical_json_bytes({"z":1,"a":[True,"x"]}),b'{"a":[true,"x"],"z":1}')
    with self.assertRaises(TypeError): canonical_json_bytes({"temperature":0.5})
    self.assertEqual(document_root({"a":1}),document_root({"a":1}))
    with self.assertRaises(ValueError): document_root({"document_sha256":"bad"})

  def test_recorder_rejects_noncanonical_witness_order(self):
    plan=embedding_plan("embed")
    recorder=AttestationRecorder(0)
    values={name:bytes([i]) for i,name in enumerate(reversed(plan.witnesses))}
    with self.assertRaisesRegex(ValueError,"witness order mismatch"):
      recorder.record_module(plan,values,input_names=("token_ids",),output_names=("embedding_output",))

  def test_session_artifact_is_ordered_and_self_hashed(self):
    session=__import__('extra.dfloat_attestation',fromlist=['AttestationSession']).AttestationSession()
    recorder=session.recorder(0)
    plan=embedding_plan("embed")
    values={name:bytes([i]) for i,name in enumerate(plan.witnesses)}
    recorder.record_module(plan,values,input_names=("token_ids",),output_names=("embedding_output",))
    recorder.set_token_io(input_token=7,selected_token=8)
    artifact=session.artifact({"model":"small","temperature_q16":0},"hello")
    self.assertEqual(len(artifact["run_root"]),64)
    self.assertEqual(len(artifact["steps"][0]["token_combined_root"]),64)
    self.assertIn("run_sha256",session.text_artifact(artifact))
    changed=session.artifact({"model":"small","temperature_q16":0},"hello!")
    self.assertNotEqual(artifact["run_root"],changed["run_root"])
    self.assertTrue(verify_artifact(artifact))
    artifact["steps"][0]["modules"][0]["witnesses"][0]["root"]="00"*32
    with self.assertRaises(ValueError): verify_artifact(artifact)


class TestDFAttestationSchema(unittest.TestCase):
  def test_reduction_frontiers_are_formulaic(self):
    for length in (1,257,16385,128256,1_000_003):
      plan=reduction_plan(length)
      self.assertEqual(plan.padded_length,1 << ((length-1).bit_length()))
      self.assertEqual(plan.tree_depth,(length-1).bit_length())
      self.assertEqual(plan.witness_level,plan.tree_depth//2)
      self.assertEqual(plan.frontiers,(f"level_{plan.witness_level}",))

  def test_documented_module_counts(self):
    plans=(embedding_plan(),rmsnorm_plan("norm",2048),matmul_plan("small",64),matmul_plan("linear",2048),
           matmul_plan("mlp_down",8192),attention_scores_plan("scores",64),softmax_plan("softmax",512),
           token_selection_plan(128256,sampling=False),token_selection_plan(128256,sampling=True))
    self.assertTrue(all(5 <= len(x.witnesses) <= 10 for x in plans))
    self.assertEqual(len(rmsnorm_plan("norm",2048).witnesses),8)
    self.assertEqual(len(matmul_plan("small",64).witnesses),5)
    self.assertEqual(len(matmul_plan("mlp_down",8192).witnesses),5)
    self.assertEqual(len(softmax_plan("softmax",512).witnesses),10)

  def test_witness_schema_does_not_grow_with_reduction_depth(self):
    self.assertEqual(matmul_plan("wide",2048).witnesses,matmul_plan("wide",1_000_003).witnesses)
    self.assertEqual(rmsnorm_plan("norm",2048).witnesses,rmsnorm_plan("norm",1_000_003).witnesses)

  def test_witness_level_is_derived_from_tree_midpoint(self):
    for length in (1,2,3,64,2048,8192,128256,1_000_003):
      plan=reduction_plan(length)
      self.assertEqual(plan.witness_level,(length-1).bit_length()//2)

  def test_llama_block_order_and_root_are_stable(self):
    cfg=LlamaV1Config(dim=2048,hidden_dim=8192,head_dim=64,context_length=512)
    a,b=llama_block_plan(3,cfg),llama_block_plan(3,cfg)
    self.assertEqual(a,b)
    self.assertEqual(a[0].name,"layers.3.attention_norm")
    self.assertEqual(a[-1].name,"layers.3.output_residual")
    self.assertTrue(all(5 <= len(x.witnesses) <= 10 for x in a))
    metadata={"model":"llama-3.2-1b","df_arithmetic":"v1"}
    self.assertEqual(schema_root(a,metadata),schema_root(b,{"df_arithmetic":"v1","model":"llama-3.2-1b"}))
    self.assertNotEqual(schema_root(a,metadata),schema_root(llama_block_plan(4,cfg),metadata))


if __name__ == "__main__": unittest.main()
