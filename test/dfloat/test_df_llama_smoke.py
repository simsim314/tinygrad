import hashlib, math, os, unittest
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_state_dict, load_state_dict
from extra.dfloat import convert_state_dict_df16, df16_raw_bytes
from extra.models.llama import Transformer

DEVICE=os.getenv("DFLOAT_TEST_DEVICE","CUDA")

class TestDFLlamaSmoke(unittest.TestCase):
  def test_one_layer_transformer_repeatability(self):
    Tensor.manual_seed(1234)
    model=Transformer(dim=8,hidden_dim=16,n_heads=2,n_layers=1,norm_eps=1e-5,vocab_size=32,
                      n_kv_heads=2,max_context=8,jit=False,disable_kv_cache=True)
    tokens=Tensor([[1,7,3]],dtype=dtypes.int32,device=DEVICE)
    native=model.forward(tokens,0,math.nan,0,0.0,0.0,0.0).numpy()
    state=convert_state_dict_df16(get_state_dict(model),device=DEVICE)
    load_state_dict(model,state,verbose=False)
    checks=[]
    for _ in range(5):
      logits=model.forward(tokens,0,math.nan,0,0.0,0.0,0.0).realize()
      self.assertEqual(logits.dtype,dtypes.df16)
      checks.append(hashlib.sha256(df16_raw_bytes(logits)).hexdigest())
    self.assertEqual(len(set(checks)),1)
    self.assertEqual(native.argmax(axis=-1).tolist(),logits.numpy().argmax(axis=-1).tolist())
    self.assertEqual(checks[0],"d9f1e7fd2f2d2de81945a0f625b2fb3daa0992e154880c0989435284d10cfec7")

if __name__ == "__main__": unittest.main()
