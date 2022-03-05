import pytest
import torch
from tests import skip_if_cuda_not_available

if torch.cuda.is_available():
    from pytorch_lightning import Trainer
    from pytorch_lightning.strategies import BaguaStrategy
    from tests.pytorch_lightning.boring_model import BoringModel
else:
    Trainer = None
    BaguaStrategy = None
    BoringModel = object


class TestModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 32)


class TestModel4QAdam(TestModel):
    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.05, warmup_steps=20)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@skip_if_cuda_not_available()
def test_bagua_default(tmpdir):
    torch.manual_seed(13)
    model = TestModel()
    assert torch.norm(model.layer.weight) == 3.1995556354522705
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy="bagua",
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model)
    trainer.test(model)
    assert torch.norm(model.layer.weight) == 2.4819390773773193


@pytest.mark.parametrize(
    ["algorithm", "criterion"],
    [
        ("gradient_allreduce", 2.835376262664795),
        ("bytegrad", 2.835047960281372),
        ("decentralized", 2.835376262664795),
        ("low_precision_decentralized", 2.8350701332092285),
    ],
)
@skip_if_cuda_not_available()
def test_bagua_algorithm(tmpdir, algorithm, criterion):
    torch.manual_seed(13)
    model = TestModel()
    assert torch.norm(model.layer.weight) == 3.1995556354522705
    bagua_strategy = BaguaStrategy(algorithm=algorithm)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    trainer.test(model)
    assert torch.norm(model.layer.weight) == criterion


@skip_if_cuda_not_available()
def test_bagua_async(tmpdir):
    model = TestModel()
    bagua_strategy = BaguaStrategy(
        algorithm="async", warmup_steps=10, sync_interval_ms=10
    )
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    trainer.test(model)
    assert torch.norm(model.layer.weight) < 5


@skip_if_cuda_not_available()
def test_qadam(tmpdir):
    torch.manual_seed(13)
    model = TestModel4QAdam()
    assert torch.norm(model.layer.weight) == 3.1995556354522705
    bagua_strategy = BaguaStrategy(algorithm="qadam")
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    trainer.test(model)
    assert torch.norm(model.layer.weight) == 6.891299724578857
