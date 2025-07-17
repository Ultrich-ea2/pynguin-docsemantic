def is_prime_simple(n: int) -> bool:
    """
    Check if a number is prime.

    Preconditions:
    - n >= 2
    - n <= 1000

    Args:
        n (int): Integer to check for primality.

    Returns:
        bool: True if n is prime, False otherwise.

    Examples:
        >>> is_prime_simple(2)
        True
        >>> is_prime_simple(3)
        True
        >>> is_prime_simple(4)
        False
        >>> is_prime_simple(17)
        True
        >>> is_prime_simple(9)
        False
    """
    if n < 2 or n > 1000:
        raise ValueError("n must be between 2 and 1000")

    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
