def gcd(a, b):
    """Calculate the greatest common divisor of two integers.

    Args:
        a (int): First integer
        b (int): Second integer

    Returns:
        int: Greatest common divisor of a and b

    Examples:
        >>> gcd(12, 8)
        4
        >>> gcd(17, 13)
        1
        >>> gcd(100, 25)
        25
        >>> gcd(0, 5)
        5

    Preconditions:
        - a >= 0
        - b >= 0
        - not (a == 0 and b == 0)
    """
    if a == 0:
        return b
    if b == 0:
        return a

    while b != 0:
        temp = b
        b = a % b
        a = temp
    return a


def fibonacci(n):
    """Calculate the nth Fibonacci number.

    Args:
        n (int): Position in Fibonacci sequence

    Returns:
        int: The nth Fibonacci number

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(5)
        5
        >>> fibonacci(10)
        55

    Preconditions:
        - n >= 0
        - n <= 50
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def factorial(n):
    """Calculate the factorial of a non-negative integer.

    Args:
        n (int): Non-negative integer

    Returns:
        int: Factorial of n

    Examples:
        >>> factorial(0)
        1
        >>> factorial(1)
        1
        >>> factorial(5)
        120
        >>> factorial(3)
        6

    Preconditions:
        - n >= 0
        - n <= 20
    """
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def is_prime(n):
    """Check if a number is prime.

    Args:
        n (int): Number to check

    Returns:
        bool: True if n is prime, False otherwise

    Examples:
        >>> is_prime(2)
        True
        >>> is_prime(17)
        True
        >>> is_prime(4)
        False
        >>> is_prime(1)
        False

    Preconditions:
        - n >= 1
        - n <= 1000
    """
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
