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

    def test_epoch_end(self, outputs) -> None:
        mean_y = torch.stack([x["y"] for x in outputs]).mean()
        self.log("mean_y", mean_y)


class TestModel4QAdam(TestModel):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.05, warmup_steps=20)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@skip_if_cuda_not_available()
def test_bagua_default(tmpdir):
    torch.manual_seed(13)
    model = TestModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy="bagua",
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] == 0.6336451768875122


@pytest.mark.parametrize(
    ["algorithm", "test_loss"],
    [
        ("gradient_allreduce", 0.9828455448150635),
        ("bytegrad", 0.9484740495681763),
        ("decentralized", 0.9747325778007507),
        ("low_precision_decentralized", 0.864328145980835),
    ],
)
@skip_if_cuda_not_available()
def test_bagua_algorithm(tmpdir, algorithm, test_loss):
    model = TestModel()
    bagua_strategy = BaguaStrategy(algorithm=algorithm)
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] == test_loss


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
    ret = trainer.test(model)
    assert ret[0]["mean_y"] < 2


@skip_if_cuda_not_available()
def test_qadam(tmpdir):
    model = TestModel4QAdam()
    bagua_strategy = BaguaStrategy(algorithm="qadam")
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        strategy=bagua_strategy,
        accelerator="gpu",
        devices=2,
    )
    trainer.fit(model)
    ret = trainer.test(model)
    assert ret[0]["mean_y"] == 1.8056042194366455
