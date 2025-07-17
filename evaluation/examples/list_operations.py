def safe_list_access(items, index):
    """Access list element with bounds checking.

    Args:
        items (list): Input list, must not be empty
        index (int): List index, range from 0 to len(items)-1

    Returns:
        object: Element at specified index or None if invalid

    Constraints:
        - items must not be empty
        - index must be within valid range [0, len(items)-1]
    """
    if 0 <= index < len(items):
        return items[index]
    return None


def filter_positive_numbers(numbers):
    """Filter list to keep only positive numbers.

    Args:
        numbers (list): List of numbers, can contain int or float

    Returns:
        list: List containing only positive numbers

    Constraints:
        - all elements in numbers must be numeric (int or float)
    """
    return [n for n in numbers if isinstance(n, (int, float)) and n > 0]


def calculate_average(values):
    """Calculate average of numeric values.

    Args:
        values (list): List of numbers, must not be empty

    Returns:
        float: Average of the values

    Constraints:
        - values must not be empty
        - all elements must be numeric
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def find_max_value(data):
    """Find maximum value in non-empty list.

    Args:
        data (list): List of comparable values, must not be empty

    Returns:
        object: Maximum value from the list

    Constraints:
        - data must not be empty
        - all elements must be comparable
    """
    if not data:
        return None
    return max(data)
