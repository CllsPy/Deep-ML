def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """

    fn = sum(x != y for x, y in zip(y_true, y_pred))
    fp =  fn = sum(x != y for x, y in zip(y_true, y_pred))
    tp = sum((x == y) for  x, y in zip(y_true, y_pred))

    f1_micro = (2 * tp) / (2*tp + fp + fn)

    return f1_micro
