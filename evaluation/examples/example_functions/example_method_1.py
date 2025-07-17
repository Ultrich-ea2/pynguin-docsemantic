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
