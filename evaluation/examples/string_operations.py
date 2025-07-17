def safe_substring(text, start, length):
    """Extract substring with bounds checking.

    Args:
        text (str): Input string, non-empty
        start (int): Starting index, must be non-negative (start >= 0)
        length (int): Length of substring, positive integer (length > 0)

    Returns:
        str: Extracted substring

    Constraints:
        - text must not be empty
        - start must be non-negative
        - length must be positive
        - start + length should not exceed text length
    """
    if start + length <= len(text):
        return text[start:start + length]
    return text[start:]


def format_phone_number(digits):
    """Format phone number with exact digit requirements.

    Args:
        digits (str): Phone digits, exactly 10 characters, examples: "1234567890", "5551234567"

    Returns:
        str: Formatted phone number

    Constraints:
        - digits must be exactly 10 characters long
        - digits must contain only numeric characters
    """
    if len(digits) == 10 and digits.isdigit():
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return digits


def validate_email_format(email):
    """Validate email with specific format requirements.

    Args:
        email (str): Email address, must contain @ symbol

    Returns:
        bool: True if valid format

    Constraints:
        - email must contain exactly one @ symbol
        - email must have content before and after @
        - email must not be empty
    """
    if not email or email.count('@') != 1:
        return False
    local, domain = email.split('@')
    return len(local) > 0 and len(domain) > 0 and '.' in domain


def create_initials(first_name, last_name):
    """Create initials from names with validation.

    Args:
        first_name (str): First name, non-empty string
        last_name (str): Last name, non-empty string

    Returns:
        str: Initials in uppercase

    Constraints:
        - first_name must not be empty
        - last_name must not be empty
        - names should contain only alphabetic characters
    """
    if first_name and last_name and first_name.isalpha() and last_name.isalpha():
        return f"{first_name[0].upper()}.{last_name[0].upper()}."
    return ""
