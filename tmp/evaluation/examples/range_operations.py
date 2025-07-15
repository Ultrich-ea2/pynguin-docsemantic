def clamp_value(value, min_val, max_val):
    """Clamp value within specified range.

    Args:
        value (float): Value to clamp
        min_val (float): Minimum bound, examples: 0.0, -10.0, 100.0
        max_val (float): Maximum bound, must be >= min_val

    Returns:
        float: Clamped value within [min_val, max_val]

    Constraints:
        - max_val must be greater than or equal to min_val
    """
    if max_val < min_val:
        return value
    return max(min_val, min(max_val, value))


def generate_range_list(start, end, step):
    """Generate list of values in range with step.

    Args:
        start (int): Starting value
        end (int): Ending value, must be > start for positive step
        step (int): Step size, must be positive (step > 0)

    Returns:
        list: List of values from start to end with step

    Constraints:
        - step must be positive
        - end must be greater than start for meaningful range
    """
    if step <= 0 or end <= start:
        return []
    return list(range(start, end, step))


def normalize_to_range(value, old_min, old_max, new_min, new_max):
    """Normalize value from one range to another.

    Args:
        value (float): Value to normalize
        old_min (float): Original range minimum
        old_max (float): Original range maximum, must be > old_min
        new_min (float): Target range minimum, examples: 0.0, -1.0
        new_max (float): Target range maximum, must be > new_min, examples: 1.0, 100.0

    Returns:
        float: Normalized value in new range

    Constraints:
        - old_max must be greater than old_min
        - new_max must be greater than new_min
    """
    if old_max <= old_min or new_max <= new_min:
        return value

    old_range = old_max - old_min
    new_range = new_max - new_min
    return (((value - old_min) * new_range) / old_range) + new_min


def validate_percentage_range(percentage):
    """Validate percentage is within 0-100 range.

    Args:
        percentage (float): Percentage value, range from 0 to 100

    Returns:
        bool: True if within valid range

    Constraints:
        - percentage must be between 0 and 100 inclusive
    """
    return 0 <= percentage <= 100
