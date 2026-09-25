import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import calculate_naulc, compute_to_target, find_pareto_frontier, tokens_to_target
from data_accounting import instruction_split_specs
from evaluate import load_finetuned_model
from metrics import EfficiencyTracker
from train import evaluate_validation
import train as training
from weight_analysis import extract_adapter_update


class AnalysisTests(unittest.TestCase):
    def test_training_and_validation_slices_do_not_overlap(self):
        self.assertEqual(instruction_split_specs(1000, 50),
                         ("train[:1000]", "train[1000:1050]"))

    def test_targets_use_validation_loss_only(self):
        history = [
            {"loss": 0.2, "val_loss": None, "tokens_seen": 10},
            {"loss": 0.1, "val_loss": 1.4, "tokens_seen": 20},
            {"loss": 0.1, "val_loss": 0.9, "tokens_seen": 30, "estimated_flops": 123},
        ]
        self.assertEqual(tokens_to_target(history, 1.0), 30)
        self.assertEqual(compute_to_target(history, 1.0), 123)

    def test_constant_validation_curve_has_correct_mean(self):
        history = [
            {"tokens_seen": 10, "val_loss": 1.0},
            {"tokens_seen": 20, "val_loss": 1.0},
        ]
        self.assertEqual(calculate_naulc(history), 1.0)

    def test_mean_loss_interpolates_at_common_token_budget(self):
        history = [
            {"tokens_seen": 0, "val_loss": 1.0},
            {"tokens_seen": 10, "val_loss": 3.0},
        ]
        self.assertEqual(calculate_naulc(history, token_budget=5), 1.5)

    def test_pareto_maximizes_quality(self):
        runs = [{"compute": 1, "quality": 0.5}, {"compute": 2, "quality": 0.9}]
        self.assertEqual(find_pareto_frontier(runs, "compute", "quality"), runs)


class TrackerTests(unittest.TestCase):
    def test_log_directory_and_run_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = EfficiencyTracker("run", log_dir=directory)
            tracker.start()
            tracker.log_step(1, 1.0, 1e-4, 7, 2, val_loss=0.8, eval_tokens=3,
                             batch_trained_tokens=4)
            path = Path(directory) / "run.jsonl"
            record = json.loads(path.read_text().strip())
            self.assertEqual(record["trained_tokens_seen"], 4)
            self.assertEqual(record["val_loss"], 0.8)
            EfficiencyTracker("run", log_dir=directory)
            self.assertEqual(path.read_text(), "")


class EvaluationTests(unittest.TestCase):
    def test_fft_checkpoint_loads_as_full_model(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            sentinel = object()
            with patch("evaluate.AutoModelForCausalLM.from_pretrained", return_value=sentinel) as load:
                loaded = load_finetuned_model("base", checkpoint, torch.float32, "cpu")
            self.assertIs(loaded, sentinel)
            self.assertEqual(load.call_args.args[0], checkpoint)

    def test_validation_loss_weights_response_tokens(self):
        class Model:
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def __call__(self, **batch):
                return SimpleNamespace(loss=batch["mock_loss"])

        class Accelerator:
            device = torch.device("cpu")
            num_processes = 1

            @staticmethod
            def reduce(value, reduction):
                return value

        batches = [
            {"labels": torch.tensor([[-100, 1]]), "mock_loss": torch.tensor(1.0)},
            {"labels": torch.tensor([[-100, 1, 2]]), "mock_loss": torch.tensor(2.5)},
        ]
        model = Model()
        loss, tokens = evaluate_validation(model, batches, Accelerator())
        self.assertTrue(math.isclose(loss, 2.0))
        self.assertEqual(tokens, 3)
        self.assertTrue(model.training)


class TrainingLoopTests(unittest.TestCase):
    def test_reaches_max_steps_and_logs_held_out_loss(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = SimpleNamespace(pad_token_id=None)

            def to(self, *args, **kwargs):
                return self

            def forward(self, input_ids, attention_mask, labels):
                return SimpleNamespace(loss=(self.weight - 0.5).square() + input_ids.float().mean() * 0)

            def save_pretrained(self, path, **kwargs):
                pass

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"
            pad_token_id = 0

            def save_pretrained(self, path):
                pass

        class Dataset:
            def __init__(self, raw_data, tokenizer, max_length):
                self.rows = raw_data

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return {"input_ids": torch.tensor([1, 2]),
                        "attention_mask": torch.tensor([1, 1]),
                        "labels": torch.tensor([-100, 2])}

        class Accelerator:
            device = torch.device("cpu")
            is_main_process = True
            local_process_index = 0
            num_processes = 1
            distributed_type = "NO"

            def prepare(self, *items):
                return items

            def backward(self, loss):
                loss.backward()

            def clip_grad_norm_(self, parameters, max_norm):
                return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

            def reduce(self, value, reduction):
                return value

            def wait_for_everyone(self):
                pass

            def unwrap_model(self, model):
                return model

        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                experiment=SimpleNamespace(name="tiny_fft", seed=42),
                model=SimpleNamespace(name="tiny", dtype="float32"),
                method=SimpleNamespace(name="fft"),
                data=SimpleNamespace(dataset_name="fake", train_samples=2, val_samples=1,
                                     max_length=4, batch_size=1),
                training=SimpleNamespace(learning_rate=0.01, weight_decay=0.0,
                                         warmup_steps=0, max_steps=5, eval_every_steps=2),
                tracking=SimpleNamespace(log_dir=directory,
                                         checkpoint_dir=str(Path(directory) / "checkpoint")),
            )
            with patch.object(training, "Accelerator", Accelerator), \
                 patch.object(training, "set_seed") as set_seed, \
                 patch.object(training.AutoModelForCausalLM, "from_pretrained", return_value=Model()), \
                 patch.object(training.AutoTokenizer, "from_pretrained", return_value=Tokenizer()), \
                 patch.object(training, "InstructionDataset", Dataset), \
                 patch.object(training, "load_dataset", side_effect=[[0, 1], [2]]) as load, \
                 patch.object(training.torch.cuda, "is_available", return_value=False):
                training.train(config)
            self.assertEqual([call.kwargs["split"] for call in load.call_args_list],
                             ["train[:2]", "train[2:3]"])
            set_seed.assert_called_once_with(42, device_specific=True)
            records = [json.loads(line) for line in (Path(directory) / "tiny_fft.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 6)
            self.assertEqual(records[0]["step"], 0)
            self.assertEqual(records[0]["tokens_seen"], 0)
            self.assertEqual(records[-1]["step"], 5)
            self.assertEqual(records[-1]["tokens_seen"], 10)
            self.assertEqual([row["step"] for row in records if row["val_loss"] is not None], [0, 2, 4, 5])


class WeightAnalysisTests(unittest.TestCase):
    def test_uses_effective_merged_weight_and_restores_base(self):
        class Layer:
            merged = False

            def __init__(self):
                self.base_layer = SimpleNamespace(weight=torch.tensor([[2.0]]))
                self.lora_A = object()

            def get_base_layer(self):
                return self.base_layer

            def merge(self):
                self.base_layer.weight.add_(3.0)
                self.merged = True

            def unmerge(self):
                self.base_layer.weight.sub_(3.0)
                self.merged = False

        layer = Layer()
        model = SimpleNamespace(named_modules=lambda: iter([("target", layer)]))
        delta, original = extract_adapter_update(model, "target")
        self.assertEqual(delta.item(), 3.0)
        self.assertEqual(original.item(), 2.0)
        self.assertEqual(layer.base_layer.weight.item(), 2.0)


if __name__ == "__main__":
    unittest.main()
