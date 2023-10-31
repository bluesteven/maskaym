import numpy as np

try:
    import torch
except ImportError:
    raise ImportError("Please install the following packages: torch.")

from maskay.module import MaskayModule

class Module(MaskayModule):
    def __init__(
        self,
        cropsize: int = 512,
        overlap: int = 32,
        device: str = "cpu",
        batchsize: int = 1,
        order: str = "BCHW",
        quiet: int = False,
    ):
        super().__init__(cropsize, overlap, device, batchsize, order, quiet)

    def inProcessing(self, tensor: np.ndarray):
        pass

    def forward(self, x):
        pass

    def outProcessing(self, tensor: np.ndarray):
        pass

    def _run(self, tensor: np.ndarray):
        # 配置日志输出到文件
        logging.basicConfig(filename='/content/log.txt', level=logging.DEBUG)
        # 输出到日志文件
        logging.debug('Variable shape: %s', tensor.shape)
        tensor = torch.Tensor(tensor)

        # Run the model
        with torch.no_grad():
            # 输出到日志文件
            logging.debug(self.device)
            if self.device == "cuda":
                tensor = tensor.cuda()
            tensor = self.forward(tensor).detach().cpu().numpy()
            torch.cuda.empty_cache()
        return tensor
