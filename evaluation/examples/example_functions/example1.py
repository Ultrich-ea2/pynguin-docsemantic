def constrained_math(x, y, threshold):
    """Mathematical operations with explicit constraints.

    Args:
        x (float): Input value, must be positive (x > 0)
        y (int): Second value, range from 1 to 100
        threshold (float): Threshold value, non-negative and <= 1.0

    Returns:
        float: Computed result

    Constraints:
        - x must be greater than zero
        - y must be between 1 and 100 inclusive
        - threshold >= 0 and threshold <= 1.0
    """
    if x > threshold:
        return x * y
    else:
        return x + y


def string_validator(text, min_length, encoding):
    """Validate and process strings with constraints.

    Args:
        text (str): Input string, non-empty
        min_length (int): Minimum length, positive integer (min_length > 0)
        encoding (str): Character encoding, defaults to 'utf-8'

    Returns:
        bool: True if valid

    Preconditions:
        - text must not be empty
        - min_length must be a positive integer
        - encoding must be a valid encoding name
    """
    if len(text) >= min_length:
        try:
            text.encode(encoding)
            return True
        except UnicodeError:
            return False
    return False


def range_processor(values, lower_bound, upper_bound):
    """Process values within specified ranges.

    Args:
        values (list): List of numbers
        lower_bound (float): Lower boundary, range -100 to 100
        upper_bound (float): Upper boundary, must be >= lower_bound

    Returns:
        list: Filtered values

    Requirements:
        - All values must be numeric
        - lower_bound between -100 and 100
        - upper_bound >= lower_bound
    """
    return [v for v in values if lower_bound <= v <= upper_bound]
