def parse_date(date_string, format_type):
    """Parse a date string according to specified format.

    Args:
        date_string (str): Date string to parse, non-empty,
                          examples: '2023-12-25', '25/12/2023', 'Dec 25, 2023'
        format_type (str): Expected format, type: str,
                          examples: 'iso', 'european', 'american'

    Returns:
        dict: Parsed date components with year, month, day

    Examples:
        >>> parse_date('2023-12-25', 'iso')
        {'year': 2023, 'month': 12, 'day': 25}
        >>> parse_date('25/12/2023', 'european')
        {'year': 2023, 'month': 12, 'day': 25}

    Preconditions:
        - len(date_string) >= 8
        - len(date_string) <= 20
        - format_type in ['iso', 'european', 'american']
    """
    import re

    if format_type == 'iso':
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_string)
        if match:
            return {'year': int(match.group(1)), 'month': int(match.group(2)), 'day': int(match.group(3))}
    elif format_type == 'european':
        match = re.match(r'(\d{2})/(\d{2})/(\d{4})', date_string)
        if match:
            return {'year': int(match.group(3)), 'month': int(match.group(2)), 'day': int(match.group(1))}

    return {'year': 0, 'month': 0, 'day': 0}


def calculate_discount(price, discount_percent, customer_type):
    """Calculate discounted price based on customer type.

    Args:
        price (float): Original price, must be positive, examples: 100.0, 50.5, 299.99
        discount_percent (float): Discount percentage, range 0.0 to 100.0,
                                 examples: 10.0, 25.0, 5.0
        customer_type (str): Customer category, type: str,
                            examples: 'regular', 'premium', 'vip'

    Returns:
        float: Final price after discount

    Examples:
        >>> calculate_discount(100.0, 10.0, 'regular')
        90.0
        >>> calculate_discount(100.0, 20.0, 'premium')
        80.0

    Function constraints:
        - price must be positive
        - discount_percent must be between 0 and 100
        - customer_type must be valid category
    """
    if customer_type == 'premium':
        discount_percent += 5  # Extra 5% for premium
    elif customer_type == 'vip':
        discount_percent += 10  # Extra 10% for VIP

    discount_amount = price * (discount_percent / 100)
    return round(price - discount_amount, 2)


def validate_password(password):
    """Validate password strength.

    Args:
        password (str): Password to validate, type: str,
                       examples: 'MyPass123!', 'weakpass', 'Str0ng@Pass'

    Returns:
        dict: Validation results with strength score and requirements

    Examples:
        >>> validate_password('MyPass123!')
        {'valid': True, 'strength': 'strong', 'score': 4}
        >>> validate_password('weak')
        {'valid': False, 'strength': 'weak', 'score': 1}

    Preconditions:
        - len(password) >= 1
        - len(password) <= 128
        - password contains only printable ASCII characters
    """
    import re

    score = 0
    if len(password) >= 8:
        score += 1
    if re.search(r'[a-z]', password):
        score += 1
    if re.search(r'[A-Z]', password):
        score += 1
    if re.search(r'[0-9]', password):
        score += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        score += 1

    strength_map = {0: 'very weak', 1: 'weak', 2: 'fair', 3: 'good', 4: 'strong', 5: 'very strong'}
    return {
        'valid': score >= 3,
        'strength': strength_map.get(score, 'unknown'),
        'score': score
    }
