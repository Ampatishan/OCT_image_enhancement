import torch
#from losses.  import   

class MetricEvaluator:
    def __init__(self, metrics_list=None):
        self.metrics_list = metrics_list if metrics_list else ["l1", "l2", "ssim"]

        # Initialize the losses
        self.l1_loss = None


    def compute(self, pred, target, step=None):
        """
        Compute metrics based on the given list of metrics and return them.
        Args:
            pred: The predicted output from the model.
            target: The ground truth (true values).
            step: The current training step or epoch for logging purposes.
        Returns:
            A dictionary containing the calculated metrics.
        """
        results = {}

        # Calculate the metrics requested
        for metric in self.metrics_list:
            if metric == "l1":
                l1 = self.l1_loss(pred, target).item()
                results["L1 Loss"] = l1

            else:
                raise ValueError(f"Unknown metric '{metric}'")

        return results

    def calculate_metrics(self, pred, target, step=None):
        """
        Return metrics in a human-readable format.
        Args:
            pred: The predicted output from the model.
            target: The ground truth (true values).
            step: The current training step or epoch for logging purposes.
        """
        metrics = self.compute(pred, target, step)
        return metrics
