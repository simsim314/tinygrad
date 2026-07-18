from typing import Self, Sequence
from tinygrad.uop import Ops
from tinygrad.dtype import DTypeLike, dtypes, sum_acc_dtype, to_dtype
from tinygrad.helpers import make_tuple, prod, all_int
from tinygrad.mixin.dtype import DTypeMixin
from tinygrad.mixin.movement import MovementMixin


class ReduceMixin(DTypeMixin, MovementMixin):
  def _rop(self, op: Ops, axis: tuple[int, ...]) -> Self:
    raise NotImplementedError

  def _reduce(self, op:Ops, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    axis = tuple(self._resolve_dim(x) for x in (range(self.ndim) if axis is None else make_tuple(axis, 1)))
    if self.ndim == 0: axis = ()
    # DF reductions have a backend-independent adjacent-pair tree.  Every level
    # is materialized, so graph fusion, warp size, tensor cores, and scheduling
    # cannot reassociate it.  Symbolic lengths retain the ordered-loop fallback.
    if self.dtype.scalar() in dtypes.dfloats and op in {Ops.ADD,Ops.MUL,Ops.MAX} and axis and all_int(self.shape):
      keep_axes=tuple(i for i in range(self.ndim) if i not in axis)
      outer=tuple(self.shape[i] for i in keep_axes)
      n=prod(self.shape[i] for i in axis)
      x=self.permute(keep_axes+axis).reshape((*outer,n))
      tree_n=1<<(n-1).bit_length()
      identity={Ops.ADD:0,Ops.MUL:1,Ops.MAX:self.dtype.min}[op]
      if tree_n != n:
        pads=((0,0),)*len(outer)+((0,tree_n-n),)
        base=x.pad(pads,value=0)
        if identity == 0: x=base
        else:
          valid=x.const_like(1).cast(dtypes.bool).pad(pads,value=0)
          x=valid.where(base,base.const_like(identity))
      while tree_n > 1:
        pair=x.reshape((*outer,tree_n//2,2))
        left,right=pair[...,0],pair[...,1]
        x={Ops.ADD:left+right,Ops.MUL:left*right,Ops.MAX:left.maximum(right)}[op].contiguous()
        tree_n//=2
      ret=x.reshape(outer)
    else: ret = self._rop(op, axis)
    return ret.reshape(tuple(1 if i in axis else s for i,s in enumerate(self.shape))) if keepdim else ret

  def sum(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Self:
    """
    Returns the sum of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.sum(axis=1).numpy())
    ```
    """
    ret = self.cast(sum_acc_dtype(self.dtype) if dtype is None else to_dtype(dtype))._reduce(Ops.ADD, axis, keepdim)
    return ret.cast(self.dtype) if dtype is None and self.dtype in (dtypes.float16, dtypes.bfloat16, dtypes.df16, *dtypes.fp8s) else ret

  def prod(self, axis:int|Sequence[int]|None=None, keepdim=False, dtype:DTypeLike|None=None) -> Self:
    """
    Returns the product of the elements of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    You can pass in `dtype` keyword argument to control the data type of the accumulation.
    If not specified, the accumulation data type is chosen based on the input tensor's data type.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([-1, -2, -3, 1, 2, 3]).reshape(2, 3)
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.prod(axis=1).numpy())
    ```
    """
    return self.cast(to_dtype(dtype) if dtype is not None else self.dtype)._reduce(Ops.MUL, axis, keepdim)

  def max(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Returns the maximum value of the tensor along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the axis along
    which the maximum is computed and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 0, 2], [5, 4, 3]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.max(axis=1, keepdim=True).numpy())
    ```
    """
    return self._reduce(Ops.MAX, axis, keepdim)

  def any(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Tests if any element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.any(axis=1, keepdim=True).numpy())
    ```
    """
    return self.bool().max(axis, keepdim)

  def all(self, axis:int|Sequence[int]|None=None, keepdim=False) -> Self:
    """
    Tests if all element evaluates to `True` along the specified axis or axes.

    You can pass in `axis` and `keepdim` keyword arguments to control the reduce axis and whether the reduced dimensions are retained.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[True, True], [True, False], [False, False]])
    print(t.numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=0).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.all(axis=1, keepdim=True).numpy())
    ```
    """
    return self.bool().prod(axis, keepdim)
