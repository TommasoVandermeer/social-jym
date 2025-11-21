from jax import jit
import jax.numpy as jnp
from functools import partial

from socialjym.utils.terminations.base_termination import BaseTermination

class Timeout(BaseTermination):
    """
    Termination condition based on a timeout.
    The episode ends if the timeout is reached.
    """
    def __init__(self, time_limit: float):
        super().__init__('timeout')
        self.time_limit = time_limit

    @partial(jit, static_argnames=("self"))
    def __call__(
        self, 
        time
    ) -> tuple[bool, dict]:
        """
        Computes whether the timeout has been reached.

        args:
        - time: float. The current time.

        output:
        - timeout_reached: bool. True if the timeout has been reached, False otherwise.
        - info: dict. Additional information about the timeout status.
        """
        timeout = time >= self.time_limit
        return timeout, {}