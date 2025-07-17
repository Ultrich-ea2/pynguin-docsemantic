def fibonacci_nth(n: int) -> int:
    """
    Calculate the nth Fibonacci number.

    Preconditions:
    - n >= 0
    - n <= 30

    Args:
        n (int): Position in Fibonacci sequence (0-indexed).

    Returns:
        int: The nth Fibonacci number.

    Examples:
        >>> fibonacci_nth(0)
        0
        >>> fibonacci_nth(1)
        1
        >>> fibonacci_nth(5)
        5
        >>> fibonacci_nth(10)
        55
    """
    if n < 0 or n > 30:
        raise ValueError("n must be between 0 and 30")

    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

