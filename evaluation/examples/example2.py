def simple_arithmetic(a, b):
    """Simple arithmetic operations with basic branching.

    Args:
        a (int): First number
        b (int): Second number

    Returns:
        int: Result of operation based on conditions
    """
    if a > b:
        return a + b
    elif a == b:
        return a * b
    else:
        return a - b


def string_processor(text):
    """Process strings with various operations and edge cases.

    Args:
        text (str): Input string to process

    Returns:
        str: Processed string

    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")

    if len(text) < 3:
        return text.upper()
    elif text.startswith('test'):
        return text.replace('test', 'TEST')
    elif text.endswith('end'):
        return text[:-3] + 'END'
    else:
        return text.lower()


def list_analyzer(numbers):
    """Analyze a list of numbers with complex conditions.

    Args:
        numbers (list): List of integers

    Returns:
        dict: Analysis results
    """
    if not numbers:
        return {'empty': True}

    result = {
        'count': len(numbers),
        'sum': sum(numbers),
        'avg': sum(numbers) / len(numbers)
    }

    positive = [n for n in numbers if n > 0]
    negative = [n for n in numbers if n < 0]

    if positive:
        result['max_positive'] = max(positive)
    if negative:
        result['min_negative'] = min(negative)

    if len(numbers) > 10:
        result['large_list'] = True
        result['median'] = sorted(numbers)[len(numbers) // 2]

    return result


def fibonacci_calculator(n):
    """Calculate fibonacci number with memoization pattern.

    Args:
        n (int): Position in fibonacci sequence

    Returns:
        int: Fibonacci number at position n

    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    if n <= 1:
        return n

    cache = {0: 0, 1: 1}

    for i in range(2, n + 1):
        cache[i] = cache[i - 1] + cache[i - 2]

    return cache[n]


def password_validator(password):
    """Validate password with multiple criteria.

    Args:
        password (str): Password to validate

    Returns:
        dict: Validation results with boolean flags
    """
    result = {
        'valid': True,
        'errors': []
    }

    if len(password) < 8:
        result['valid'] = False
        result['errors'].append('Too short')

    if not any(c.isupper() for c in password):
        result['valid'] = False
        result['errors'].append('No uppercase letter')

    if not any(c.islower() for c in password):
        result['valid'] = False
        result['errors'].append('No lowercase letter')

    if not any(c.isdigit() for c in password):
        result['valid'] = False
        result['errors'].append('No digit')

    special_chars = "!@#$%^&*"
    if not any(c in special_chars for c in password):
        result['valid'] = False
        result['errors'].append('No special character')

    return result


def data_structure_handler(data):
    """Handle various data structures with type checking.

    Args:
        data: Input data of various types

    Returns:
        dict: Processing results based on data type
    """
    if isinstance(data, dict):
        if not data:
            return {'type': 'dict', 'empty': True}
        return {
            'type': 'dict',
            'keys_count': len(data.keys()),
            'has_nested': any(isinstance(v, (dict, list)) for v in data.values())
        }

    elif isinstance(data, list):
        if not data:
            return {'type': 'list', 'empty': True}
        return {
            'type': 'list',
            'length': len(data),
            'all_numbers': all(isinstance(x, (int, float)) for x in data),
            'all_strings': all(isinstance(x, str) for x in data)
        }

    elif isinstance(data, str):
        return {
            'type': 'string',
            'length': len(data),
            'is_numeric': data.isdigit(),
            'is_alpha': data.isalpha()
        }

    elif isinstance(data, (int, float)):
        return {
            'type': 'number',
            'value': data,
            'is_positive': data > 0,
            'is_integer': isinstance(data, int)
        }

    else:
        return {'type': 'unknown', 'class': data.__class__.__name__}


def exception_generator(mode):
    """Generate different types of exceptions for testing error handling.

    Args:
        mode (str): Type of exception to generate

    Returns:
        str: Success message if no exception

    Raises:
        Various exceptions based on mode
    """
    if mode == 'value_error':
        raise ValueError("This is a value error")
    elif mode == 'type_error':
        raise TypeError("This is a type error")
    elif mode == 'index_error':
        empty_list = []
        return empty_list[0]  # Will raise IndexError
    elif mode == 'key_error':
        empty_dict = {}
        return empty_dict['missing_key']  # Will raise KeyError
    elif mode == 'zero_division':
        return 1 / 0  # Will raise ZeroDivisionError
    elif mode == 'attribute_error':
        return None.missing_attribute  # Will raise AttributeError
    else:
        return "No exception generated"


def complex_calculator(operation, *args, **kwargs):
    """Complex calculator with variable arguments and keyword arguments.

    Args:
        operation (str): Type of operation to perform
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments

    Returns:
        Various types based on operation
    """
    if operation == 'sum':
        return sum(args)

    elif operation == 'product':
        result = 1
        for arg in args:
            result *= arg
        return result

    elif operation == 'stats':
        if not args:
            return None
        return {
            'min': min(args),
            'max': max(args),
            'count': len(args),
            'sum': sum(args)
        }

    elif operation == 'config':
        config = {'operation': operation}
        config.update(kwargs)
        if args:
            config['args'] = args
        return config

    elif operation == 'matrix':
        rows = kwargs.get('rows', 2)
        cols = kwargs.get('cols', 2)
        default_value = kwargs.get('default', 0)

        return [[default_value for _ in range(cols)] for _ in range(rows)]

    else:
        raise ValueError(f"Unknown operation: {operation}")


def recursive_processor(data, depth=0, max_depth=5):
    """Process nested data structures recursively.

    Args:
        data: Input data to process
        depth (int): Current recursion depth
        max_depth (int): Maximum allowed recursion depth

    Returns:
        dict: Processed data with metadata
    """
    if depth > max_depth:
        return {'error': 'Max depth exceeded', 'depth': depth}

    if isinstance(data, dict):
        result = {'type': 'dict', 'depth': depth, 'items': {}}
        for key, value in data.items():
            result['items'][key] = recursive_processor(value, depth + 1, max_depth)
        return result

    elif isinstance(data, list):
        result = {'type': 'list', 'depth': depth, 'items': []}
        for item in data:
            result['items'].append(recursive_processor(item, depth + 1, max_depth))
        return result

    else:
        return {'type': 'primitive', 'value': data, 'depth': depth}
