from typing import Any, Optional

from joblib import Parallel
from tqdm.auto import tqdm


# https://stackoverflow.com/a/61900501/5036246
class ProgressParallel(Parallel):  # type: ignore
    def __init__(
        self, use_tqdm: bool = True, total: Optional[int] = None, *args: Any, **kwargs: Any
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks

        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
