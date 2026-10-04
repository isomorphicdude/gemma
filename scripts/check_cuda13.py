"""Check CUDA forward compatibility and the checkout's JAX GPU environment."""

import ctypes
import importlib.metadata
import os
from pathlib import Path


def main():
  # Force an actual GPU check, including when the calling shell selects CPU.
  os.environ['JAX_PLATFORMS'] = 'cuda'
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
  os.environ.setdefault('NCCL_NVLS_ENABLE', '0')
  os.environ.setdefault('NCCL_CUMEM_ENABLE', '0')

  print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', '<unset>'))
  print(
      'CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
  )
  driver = ctypes.CDLL('libcuda.so.1')
  version = ctypes.c_int()
  result = driver.cuDriverGetVersion(ctypes.byref(version))
  if result:
    raise RuntimeError(f'cuDriverGetVersion returned CUDA error {result}')
  print('CUDA driver API:', version.value)
  if version.value < 13000:
    raise RuntimeError(
        'The installed CUDA 13 wheels need newer driver libraries.'
    )

  for line in Path('/proc/self/maps').read_text().splitlines():
    if 'libcuda.so.' in line:
      print('Loaded libcuda:', line.split()[-1])
      break
  result = driver.cuInit(0)
  if result:
    error_name = ctypes.c_char_p()
    driver.cuGetErrorName(result, ctypes.byref(error_name))
    raise RuntimeError(f'cuInit failed: {error_name.value!r} ({result})')
  for name in ('jax', 'jaxlib', 'jax-cuda13-plugin', 'nvidia-cuda-runtime'):
    print(f'{name}:', importlib.metadata.version(name))

  import gemma  # pylint: disable=import-outside-toplevel
  import jax  # pylint: disable=import-outside-toplevel
  import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
  import numpy as np  # pylint: disable=import-outside-toplevel

  repo_root = Path(__file__).resolve().parent.parent
  gemma_path = Path(gemma.__file__).resolve()
  print('Gemma:', gemma_path)
  if not gemma_path.is_relative_to(repo_root / 'gemma'):
    raise RuntimeError('Gemma must import from the editable checkout.')

  devices = jax.devices()
  print('JAX devices:', devices)
  if not devices or any(device.platform != 'gpu' for device in devices):
    raise RuntimeError('JAX did not select GPU devices.')

  matrix = np.arange(256, dtype=np.float32).reshape(16, 16)
  ones = np.ones((16, 16), dtype=np.float32)
  output = jax.jit(lambda a, b: a @ b)(jnp.asarray(matrix), jnp.asarray(ones))
  output.block_until_ready()
  np.testing.assert_allclose(np.asarray(output), matrix @ ones)
  print(
      'PASS: Gemma imports from the checkout and JAX compiled a GPU matrix product.'
  )

  if len(devices) > 1:
    mesh = jax.sharding.Mesh(np.asarray(devices), ('gpu',))
    collective = jax.jit(jax.shard_map(
        lambda value: jax.lax.psum(value, 'gpu'),
        mesh=mesh,
        in_specs=jax.sharding.PartitionSpec('gpu'),
        out_specs=jax.sharding.PartitionSpec(),
    ))
    total = collective(jnp.arange(len(devices), dtype=jnp.int32))
    total.block_until_ready()
    np.testing.assert_array_equal(
        np.asarray(total), [len(devices) * (len(devices) - 1) // 2]
    )
    print(f'PASS: NCCL/JAX all-reduce across {len(devices)} GPUs.')


if __name__ == '__main__':
  main()
