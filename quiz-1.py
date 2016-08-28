def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    powered = np.exp(x)
    return np.divide(powered, np.sum(powered, axis=0))

