def entropy(samples, bin_count, bin_start, bin_end):
    hist_counts = samples.histc(bins=bin_count, min=bin_start, max=bin_end)
    discrete_probabilities = hist_counts / hist_counts.sum()
    discrete_probabilities = discrete_probabilities[
        discrete_probabilities.nonzero(as_tuple=True)
    ]
    return -(discrete_probabilities * discrete_probabilities.log()).sum()


