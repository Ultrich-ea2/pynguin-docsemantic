def count_vowels(text: str) -> int:
    """
    Count vowels in a text string.

    Preconditions:
    - len(text) >= 0
    - len(text) <= 1000

    Args:
        text (str): Input string to count vowels in.

    Returns:
        int: Number of vowels (a, e, i, o, u, A, E, I, O, U) in the text.

    Examples:
        >>> count_vowels("hello")
        2
        >>> count_vowels("aeiou")
        5
        >>> count_vowels("xyz")
        0
        >>> count_vowels("")
        0
        >>> count_vowels("AEIOU")
        5
        >>> count_vowels("bcdAEIOUxyz")
        5
        >>> count_vowels("123!@#")
        0
        >>> count_vowels("a" * 1000)
        1000
        >>> count_vowels("áéíóú")
        0
        >>> count_vowels("The quick brown fox jumps over the lazy dog")
        11
    """
    if len(text) > 1000:
        raise ValueError("Text too long")

    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)
