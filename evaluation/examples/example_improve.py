def calculate_discount(price: float, discount_percent: int) -> float:
    """
    Calculate discounted price for a product.

    Preconditions:
    - price > 0
    - 0 <= discount_percent <= 100

    Args:
        price (float): Original price of the product.
        discount_percent (int): Discount percentage to apply.

    Returns:
        float: The discounted price.

    Examples:
        >>> calculate_discount(100.0, 20)
        80.0
        >>> calculate_discount(50.0, 0)
        50.0
        >>> calculate_discount(200.0, 50)
        100.0
    """
    if price <= 0:
        raise ValueError("Price must be positive")
    if not (0 <= discount_percent <= 100):
        raise ValueError("Discount must be between 0 and 100")

    return price * (1 - discount_percent / 100)


def format_phone_number(phone: str) -> str:
    """
    Format a phone number to standard format.

    Preconditions:
    - len(phone) == 10
    - phone.isdigit()

    Args:
        phone (str): A 10-digit phone number string.

    Returns:
        str: Formatted phone number in (XXX) XXX-XXXX format.

    Examples:
        >>> format_phone_number("1234567890")
        '(123) 456-7890'
        >>> format_phone_number("9876543210")
        '(987) 654-3210'
    """
    if len(phone) != 10 or not phone.isdigit():
        raise ValueError("Phone number must be exactly 10 digits")

    return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"


def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate Body Mass Index.

    Preconditions:
    - weight > 0
    - height > 0

    Args:
        weight (float): Weight in kilograms.
        height (float): Height in meters.

    Returns:
        float: BMI value rounded to 2 decimal places.

    Examples:
        >>> calculate_bmi(70.0, 1.75)
        22.86
        >>> calculate_bmi(80.0, 1.80)
        24.69
    """
    if weight <= 0 or height <= 0:
        raise ValueError("Weight and height must be positive")

    bmi = weight / (height ** 2)
    return round(bmi, 2)


def grade_percentage(score: int, total: int) -> str:
    """
    Convert a score to letter grade.

    Preconditions:
    - score >= 0
    - total > 0
    - score <= total

    Args:
        score (int): Points earned.
        total (int): Total possible points.

    Returns:
        str: Letter grade (A, B, C, D, F).

    Examples:
        >>> grade_percentage(95, 100)
        'A'
        >>> grade_percentage(85, 100)
        'B'
        >>> grade_percentage(75, 100)
        'C'
        >>> grade_percentage(65, 100)
        'D'
        >>> grade_percentage(55, 100)
        'F'
    """
    if score < 0 or total <= 0 or score > total:
        raise ValueError("Invalid score or total")

    percentage = (score / total) * 100

    if percentage >= 90:
        return 'A'
    elif percentage >= 80:
        return 'B'
    elif percentage >= 70:
        return 'C'
    elif percentage >= 60:
        return 'D'
    else:
        return 'F'


def factorial_iterative(n: int) -> int:
    """
    Calculate factorial of a number iteratively.

    Preconditions:
    - n >= 0
    - n <= 20

    Args:
        n (int): Non-negative integer.

    Returns:
        int: Factorial of n.

    Examples:
        >>> factorial_iterative(0)
        1
        >>> factorial_iterative(1)
        1
        >>> factorial_iterative(5)
        120
        >>> factorial_iterative(3)
        6
    """
    if n < 0 or n > 20:
        raise ValueError("n must be between 0 and 20")

    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert Celsius to Fahrenheit.

    Preconditions:
    - celsius >= -273.15

    Args:
        celsius (float): Temperature in Celsius.

    Returns:
        float: Temperature in Fahrenheit, rounded to 1 decimal place.

    Examples:
        >>> celsius_to_fahrenheit(0.0)
        32.0
        >>> celsius_to_fahrenheit(100.0)
        212.0
        >>> celsius_to_fahrenheit(-40.0)
        -40.0
    """
    if celsius < -273.15:
        raise ValueError("Temperature cannot be below absolute zero")

    fahrenheit = (celsius * 9 / 5) + 32
    return round(fahrenheit, 1)


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


def count_vowels(text: str) -> int:
    """
    Count vowels in a text string.

    Preconditions:
    - len(text) >= 0
    - len(text) <= 1000

    Args:
        text (str): Input string to count vowels in.

    Returns:
        int: Number of vowels (a, e, i, o, u) in the text.

    Examples:
        >>> count_vowels("hello")
        2
        >>> count_vowels("aeiou")
        5
        >>> count_vowels("xyz")
        0
        >>> count_vowels("")
        0
    """
    if len(text) > 1000:
        raise ValueError("Text too long")

    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)


def power_of_two(exponent: int) -> int:
    """
    Calculate 2 raised to the given exponent.

    Preconditions:
    - exponent >= 0
    - exponent <= 30

    Args:
        exponent (int): The exponent to raise 2 to.

    Returns:
        int: 2^exponent.

    Examples:
        >>> power_of_two(0)
        1
        >>> power_of_two(3)
        8
        >>> power_of_two(10)
        1024
    """
    if exponent < 0 or exponent > 30:
        raise ValueError("Exponent must be between 0 and 30")

    return 2 ** exponent


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
