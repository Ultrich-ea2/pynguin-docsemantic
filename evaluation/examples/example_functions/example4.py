def numeric_range_example(value: int) -> bool:
    """
    Check if a value is within a specific range.

    Args:
        value (int): An integer that should be between 1 and 100 inclusive.
            Must satisfy: 1 <= value <= 100

    Returns:
        bool: True if the value is within range, False otherwise.

    Example:
        >>> numeric_range_example(50)
        True
        >>> numeric_range_example(150)
        False
    """
    return 1 <= value <= 100


def string_pattern_function(file_path: str) -> str:
    """
    Process a file path and return its contents.

    Args:
        file_path (str): A string representing a file path.
            Must satisfy: file_path.endswith('.txt')
            Must satisfy: len(file_path) > 5

    Returns:
        str: The contents of the file or an error message.

    Example:
        >>> string_pattern_function("example.txt")
        'Contents of example.txt'
        >>> string_pattern_function("invalid.pdf")
        'Error: File must be a text file'
    """
    if not file_path.endswith('.txt'):
        return "Error: File must be a text file"
    if len(file_path) <= 5:
        return "Error: Invalid file path"
    return f"Contents of {file_path}"


def combined_constraints(name: str, age: int, is_student: bool) -> str:
    """
    Generate a greeting based on personal information.

    Args:
        name (str): The person's name.
            Must satisfy: len(name) > 0
            Must satisfy: name.isalpha()
        age (int): The person's age.
            Must satisfy: 0 <= age <= 120
        is_student (bool): Whether the person is a student.
            Must satisfy: isinstance(is_student, bool)

    Returns:
        str: A personalized greeting.

    Example:
        >>> combined_constraints("Alice", 25, True)
        'Hello, Student Alice! You are 25 years old.'
        >>> combined_constraints("", 25, True)
        'Invalid name'
    """
    if not name or not name.isalpha():
        return "Invalid name"
    if age < 0 or age > 120:
        return "Invalid age"
    prefix = "Student" if is_student else "Person"
    return f"Hello, {prefix} {name}! You are {age} years old."


def value_set_constraint(day: str) -> int:
    """
    Return the day number in a week.

    Args:
        day (str): A day of the week.
            Must satisfy: day in ['Monday', 'Tuesday', 'Wednesday',
                                  'Thursday', 'Friday', 'Saturday', 'Sunday']

    Returns:
        int: The day number (1-7) or -1 if invalid.

    Example:
        >>> value_set_constraint("Monday")
        1
        >>> value_set_constraint("Funday")
        -1
    """
    days = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }
    return days.get(day, -1)
