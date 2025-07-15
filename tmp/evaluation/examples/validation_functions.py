def validate_age(age):
    """Validate age is within reasonable range.

    Args:
        age (int): Person's age, range from 0 to 150, examples: 25, 45, 65

    Returns:
        bool: True if valid age

    Constraints:
        - age must be non-negative
        - age must be reasonable (0 to 150)
    """
    return isinstance(age, int) and 0 <= age <= 150


def validate_grade(grade):
    """Validate academic grade.

    Args:
        grade (float): Academic grade, range from 0.0 to 100.0

    Returns:
        str: Letter grade or "Invalid"

    Constraints:
        - grade must be between 0.0 and 100.0 inclusive
    """
    if not (0.0 <= grade <= 100.0):
        return "Invalid"

    if grade >= 90:
        return "A"
    elif grade >= 80:
        return "B"
    elif grade >= 70:
        return "C"
    elif grade >= 60:
        return "D"
    else:
        return "F"


def validate_temperature(temp, scale):
    """Validate temperature based on scale.

    Args:
        temp (float): Temperature value
        scale (str): Temperature scale, examples: 'C', 'F', 'K'

    Returns:
        bool: True if temperature is physically possible

    Constraints:
        - For Celsius: temp >= -273.15
        - For Fahrenheit: temp >= -459.67
        - For Kelvin: temp >= 0
    """
    if scale == 'C':
        return temp >= -273.15
    elif scale == 'F':
        return temp >= -459.67
    elif scale == 'K':
        return temp >= 0
    return False


def validate_password_strength(password):
    """Validate password meets strength requirements.

    Args:
        password (str): Password string, minimum 8 characters

    Returns:
        dict: Validation results with strength score

    Constraints:
        - password must be at least 8 characters long
        - should contain uppercase, lowercase, digit, and special character
    """
    result = {'valid': True, 'score': 0, 'requirements': []}

    if len(password) < 8:
        result['valid'] = False
        result['requirements'].append('At least 8 characters')
    else:
        result['score'] += 1

    if any(c.isupper() for c in password):
        result['score'] += 1
    else:
        result['requirements'].append('Uppercase letter')

    if any(c.islower() for c in password):
        result['score'] += 1
    else:
        result['requirements'].append('Lowercase letter')

    if any(c.isdigit() for c in password):
        result['score'] += 1
    else:
        result['requirements'].append('Digit')

    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if any(c in special_chars for c in password):
        result['score'] += 1
    else:
        result['requirements'].append('Special character')

    result['valid'] = result['score'] >= 4
    return result
