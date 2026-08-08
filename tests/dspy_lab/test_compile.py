from unittest.mock import MagicMock, patch

from dspy_lab.optimizers.compile import optimize


def test_optimize_passes_parser_metric_to_miprov2_by_default():
    fake_program = MagicMock()
    fake_examples = [MagicMock()]

    with patch("dspy_lab.optimizers.compile.dspy.MIPROv2") as mock_mipro_cls:
        mock_optimizer = MagicMock()
        mock_mipro_cls.return_value = mock_optimizer

        optimize(fake_program, fake_examples)

        from eval.scorer import parser_metric
        _, kwargs = mock_mipro_cls.call_args
        assert kwargs["metric"] is parser_metric
        mock_optimizer.compile.assert_called_once_with(fake_program, trainset=fake_examples)
