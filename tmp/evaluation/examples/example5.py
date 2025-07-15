def typed_processor(number, text, flag):
    """Function with explicit type information in descriptions.

    Args:
        number: Numeric value, type: float, must be positive
        text: Input string, dtype: str, non-empty
        flag: Boolean indicator, type bool

    Returns:
        dict: Processed results with type information
    """
    return {
        'number_doubled': number * 2,
        'text_upper': text.upper(),
        'flag_inverted': not flag
    }


def data_transformer(data, format_type, options):
    """Transform data between different formats.

    Args:
        data: Input data, can be string or integer
        format_type (str): Output format, string type, examples: 'json', 'xml', 'csv'
        options (dict): Configuration options, type: dict, defaults to {}

    Returns:
        str: Transformed data as string

    Global constraints:
        - All parameters must be provided
        - format_type must be supported format
    """
    if isinstance(data, int):
        data = str(data)

    if format_type == 'json':
        import json
        return json.dumps({'data': data, 'options': options})
    elif format_type == 'xml':
        return f"<data options='{options}'>{data}</data>"
    else:
        return f"{data},{options}"


def numeric_analyzer(values, operation, precision):
    """Analyze numeric data with type checking.

    Args:
        values: List of numbers, type: list, all elements must be numeric
        operation: Analysis type, dtype string, such as 'mean', 'median', 'mode'
        precision: Decimal places, type integer, range 0 to 10

    Returns:
        float: Analysis result

    Function constraints:
        - values must contain only numeric types
        - operation must be supported analysis type
        - precision must be non-negative integer
    """
    import statistics

    if operation == 'mean':
        result = statistics.mean(values)
    elif operation == 'median':
        result = statistics.median(values)
    else:
        result = max(values)  # fallback

    return round(result, precision)
