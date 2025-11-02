from detectron2.engine.hooks import HookBase
from tqdm import tqdm

class TqdmHook(HookBase):
    def before_train(self):
        # total is max_iter; trainer.iter starts at start_iter
        total = self.trainer.max_iter
        self.pbar = tqdm(total=total, desc="Training", dynamic_ncols=True)
        self._last = self.trainer.iter

    def after_step(self):
        # update by how many iterations progressed (usually 1)
        cur = self.trainer.iter
        delta = cur - self._last
        if delta > 0:
            # show latest total_loss if available
            loss_hist = self.trainer.storage.history("total_loss")
            loss = loss_hist.latest() if len(loss_hist) else None
            if loss is not None:
                self.pbar.set_postfix(loss=float(loss))
            self.pbar.update(delta)
            self._last = cur

    def after_train(self):
        self.pbar.close()
