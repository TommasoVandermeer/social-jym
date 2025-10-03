from jax import jit, lax
import jax.numpy as jnp
from functools import partial

### Heap implementation in JAX (not used until now, but might be useful in the future)
### Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)

@partial(jnp.vectorize, signature="(a),(a),(a),(a)->(a),(a),(a),(a)")
def merge(a, b, av, bv):
    """
    Merge two sorted key tensors `a` and `b` as well as corresponding
    int value tensors `av` and `bv`

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    n = a.shape[-1]
    ordera = jnp.searchsorted(a, b) + jnp.arange(n)
    orderb = jnp.searchsorted(b, a, side='right') + jnp.arange(n)
    out = jnp.zeros((a.shape[-1] + b.shape[-1],))
    out = out.at[ordera].set(b)
    out = out.at[orderb].set(a)
    outv = jnp.zeros((a.shape[-1] + b.shape[-1],), dtype=int)
    outv = outv.at[ordera].set(bv)
    outv = outv.at[orderb].set(av)
    return (
        out[: a.shape[-1]],
        out[a.shape[-1] :],
        outv[: a.shape[-1]],
        outv[a.shape[-1] :]
    )

@partial(jit, static_argnames=("bits"))
def make_path(index, bits):
    """
    Given a size of a node to add get the binary path to reach it.

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    mask = 2**jnp.arange(bits-1, -1, -1)
    x = jnp.bitwise_and(jnp.array(index+1).ravel(), mask) != 0
    def path(c, a):
        x = a + 2 * c
        x = jnp.minimum(x, index+1)
        return x, x

    _, x = lax.scan(path, jnp.array([1]), x[1:])
    return jnp.concatenate((jnp.array([0]), (x - 1).reshape(-1)))

@partial(jit, static_argnames=("group_size", "total_size"))
def make_heap(group_size, total_size):
    """
    Create a heap over vectors of `group_size` that
    can expand to `group_size * total_size` nodes with
    independent `batch`es.

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    size = jnp.zeros(1, dtype=int)
    key_store = jnp.full((total_size, group_size), 1.e5)
    val_store = jnp.zeros((total_size, group_size), dtype=int)
    return (key_store, val_store, size)

INF = 1.e9

@partial(jit, static_argnames=("max_size",))
def insert(key_store, val_store, size, max_size, keys, values):
    """
    Insert a batch of group_size keys `keys`  with corresponding
    integer `values`.

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    path = make_path(size, max_size)
    def insert_heapify(state, n):
        key_store, val_store, keys, values = state
        head, keys, hvalues, values = merge(
            key_store[n], keys, val_store[n], values
        )
        return (key_store.at[n].set(head), val_store.at[n].set(hvalues),
                keys, values), None

    (key_store, val_store, keys, values), _ = \
        lax.scan(insert_heapify, (key_store, val_store, keys, values), path)
    return key_store, val_store, size + 1

@partial(jit, static_argnames=("msize",))
def delete_min(heap, msize):
    """
    Delete and return the minimum key and corresponding value from the heap.

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    key_store, val_store, size = heap
    keys = key_store[0]
    values = val_store[0]
    def one():
        return key_store.at[0].set(INF), val_store.at[0].set(-1)
    def two():
        path = make_path(size - 1, msize)
        key_store2 = key_store.at[0].set(key_store[path[-1]]).at[path[-1]].set(INF)
        val_store2 = val_store.at[0].set(val_store[path[-1]]).at[path[-1]].set(-1)
        key_store3, val_store3, n = \
            lax.fori_loop(0, msize, delete_heapify, (key_store2, val_store2, 0))
        return key_store3, val_store3
    key_store, val_store = lax.cond((size == 1).all(), one, two)
    size = size - 1
    return (key_store, val_store, size), keys, values

def delete_heapify(_, state):
    """
    Heapify the node at position `n` in the heap.

    Credits: Sasha Rush (https://github.com/srush/torch-queue/blob/main/jax_queue/pq.py)
    """
    key_store, val_store, n = state
    c = jnp.stack(((n + 1) * 2 - 1, (n + 1) * 2))
    c_l,c_r = key_store[c[0]], key_store[c[1]]
    c_lv, c_rv = val_store[c[0]], val_store[c[1]]
    ins = jnp.where(c_l[-1] < c_r[-1], 0, 1)
    s, l = c[ins], c[1 - ins]
    small, k2, smallv, v2 = merge(c_l, c_r, c_lv, c_rv)
    k1, k2, v1, v2 = merge(key_store[n], small, val_store[n], smallv)
    key_store = key_store.at[l].set(k2).at[n].set(k1).at[s].set(k2)
    val_store = val_store.at[l].set(v2).at[n].set(v1).at[s].set(v2)
    return key_store, val_store, s
