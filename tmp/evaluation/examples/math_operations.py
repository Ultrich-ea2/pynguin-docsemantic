def bounded_division(dividend, divisor):
    """Perform division with clear constraints to avoid errors.

    Args:
        dividend (float): Number to be divided, any real number
        divisor (float): Divisor, must be non-zero (divisor != 0)

    Returns:
        float: Result of division

    Constraints:
        - divisor must not be zero to avoid division by zero
    """
    return dividend / divisor


def positive_square_root(number):
    """Calculate square root of positive numbers only.

    Args:
        number (float): Input number, must be positive (number > 0)

    Returns:
        float: Square root of the number

    Constraints:
        - number must be greater than zero
    """
    import math
    return math.sqrt(number)


def factorial_calculator(n):
    """Calculate factorial with clear input constraints.

    Args:
        n (int): Non-negative integer, range from 0 to 20, examples: 0, 5, 10

    Returns:
        int: Factorial of n

    Constraints:
        - n must be non-negative integer
        - n should be <= 20 for reasonable computation time
    """
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def percentage_calculator(value, percentage):
    """Calculate percentage of a value with input validation.

    Args:
        value (float): Base value, must be positive
        percentage (float): Percentage rate, range from 0 to 100, e.g. 25.0, 50.0, 75.0

    Returns:
        float: Calculated percentage

    Constraints:
        - value must be positive
        - percentage must be between 0 and 100 inclusive
    """
    return (percentage / 100) * value
