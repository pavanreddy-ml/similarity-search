from ise.psf import PostSearchFilter
import numpy as np

class KLDivergenceFilter(PostSearchFilter):

    def filter(self, sample, search_results):
        results_with_scores = [
            (result, self._kl_divergence(sample, result)) for result in search_results
        ]
        sorted_results = sorted(results_with_scores, key=lambda x: x[1])
        return [result[0] for result in sorted_results] 

    def _kl_divergence(self, p, q):
            """Calculate the KL divergence between two distributions."""
            p = np.asarray(p, dtype=np.float64)
            q = np.asarray(q, dtype=np.float64)
            p = p / np.sum(p)
            q = q / np.sum(q)
            return np.sum(np.where(p != 0, p * np.log(p / q), 0))
