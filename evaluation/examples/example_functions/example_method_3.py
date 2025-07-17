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
