from leaf_disease.apis.utils import collate_fn
import torch


class TestCollate:
    def test_int_y_with_weight(self):
        batch = [(torch.ones(3, 5, 5), 1, 1.0) for _ in range(5)]
        img, target, weight = collate_fn(batch)
        assert img.shape == (5, 3, 5, 5)
        assert target.shape == (5,)
        assert weight.shape == (5,)

    def test_int_y_no_weight(self):
        batch = [(torch.ones(3, 5, 5), 1, None) for _ in range(5)]
        img, target, weight = collate_fn(batch)
        assert img.shape == (5, 3, 5, 5)
        assert target.shape == (5,)
        assert weight is None

    def test_tensor_y_no_weight(self):
        batch = [(torch.ones(3, 5, 5), torch.ones(5) * 0.2, None) for _ in range(5)]
        img, target, weight = collate_fn(batch)
        assert img.shape == (5, 3, 5, 5)
        assert target.shape == (5, 5)
        assert weight is None