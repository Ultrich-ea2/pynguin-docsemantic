def matrix_operations(matrix: list, operation: str) -> list:
    """
    Perform various operations on a 2D matrix.

    Args:
        matrix (list): A 2D list representing a matrix.
            Must satisfy: len(matrix) > 0
            Must satisfy: all(len(row) == len(matrix[0]) for row in matrix)
            Must satisfy: all(isinstance(val, (int, float)) for row in matrix for val in row)
        operation (str): The operation to perform.
            Must satisfy: operation in ['transpose', 'diagonal', 'flatten', 'sum_rows']

    Returns:
        list: The result of the matrix operation.

    Example:
        >>> matrix_operations([[1, 2], [3, 4]], 'transpose')
        [[1, 3], [2, 4]]
        >>> matrix_operations([[1, 2, 3], [4, 5, 6]], 'sum_rows')
        [6, 15]
    """
    if not matrix or not all(len(row) == len(matrix[0]) for row in matrix):
        raise ValueError("Invalid matrix dimensions")

    if not all(isinstance(val, (int, float)) for row in matrix for val in row):
        raise TypeError("Matrix must contain only numbers")

    if operation == 'transpose':
        return [[matrix[i][j] for i in range(len(matrix))] for j in range(len(matrix[0]))]
    elif operation == 'diagonal':
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix must be square for diagonal operation")
        return [matrix[i][i] for i in range(len(matrix))]
    elif operation == 'flatten':
        return [val for row in matrix for val in row]
    elif operation == 'sum_rows':
        return [sum(row) for row in matrix]
    else:
        raise ValueError(f"Unknown operation: {operation}")


def file_processor(file_path: str, encoding: str = 'utf-8') -> dict:
    """
    Process a file and return metadata about its contents.

    Args:
        file_path (str): Path to the file to process.
            Must satisfy: file_path.endswith(('.txt', '.py', '.md', '.json'))
            Must satisfy: len(file_path) >= 5
        encoding (str): File encoding to use.
            Must satisfy: encoding in ['utf-8', 'ascii', 'latin-1']

    Returns:
        dict: Metadata about the file including line count, word count, etc.

    Example:
        >>> file_processor("example.txt")
        {'extension': 'txt', 'valid_encoding': True, 'estimated_lines': 10}
        >>> file_processor("invalid.exe")
        {'error': 'Unsupported file type'}
    """
    valid_extensions = ('.txt', '.py', '.md', '.json')
    valid_encodings = ['utf-8', 'ascii', 'latin-1']

    if not file_path.endswith(valid_extensions):
        return {'error': 'Unsupported file type'}

    if len(file_path) < 5:
        return {'error': 'Invalid file path'}

    if encoding not in valid_encodings:
        return {'error': 'Unsupported encoding'}

    extension = file_path.split('.')[-1]

    # Simulate file processing without actually reading files
    result = {
        'extension': extension,
        'encoding': encoding,
        'valid_encoding': True,
        'estimated_lines': len(file_path) * 2,  # Mock calculation
        'estimated_words': len(file_path) * 5,
        'binary_safe': extension in ['txt', 'md']
    }

    return result


def data_validator(data: dict, schema: dict) -> dict:
    """
    Validate data against a schema with type and constraint checking.

    Args:
        data (dict): The data to validate.
            Must satisfy: isinstance(data, dict)
            Must satisfy: len(data) > 0
        schema (dict): The validation schema.
            Must satisfy: isinstance(schema, dict)
            Must satisfy: all(isinstance(v, dict) for v in schema.values())

    Returns:
        dict: Validation results with errors and status.

    Example:
        >>> data_validator({'age': 25}, {'age': {'type': 'int', 'min': 0, 'max': 120}})
        {'valid': True, 'errors': []}
        >>> data_validator({'age': -5}, {'age': {'type': 'int', 'min': 0, 'max': 120}})
        {'valid': False, 'errors': ['age: value -5 below minimum 0']}
    """
    if not isinstance(data, dict) or not data:
        return {'valid': False, 'errors': ['Data must be a non-empty dictionary']}

    if not isinstance(schema, dict) or not all(isinstance(v, dict) for v in schema.values()):
        return {'valid': False, 'errors': ['Invalid schema format']}

    errors = []

    for field, constraints in schema.items():
        if field not in data:
            if constraints.get('required', False):
                errors.append(f'{field}: required field missing')
            continue

        value = data[field]
        field_type = constraints.get('type')

        # Type validation
        if field_type == 'int' and not isinstance(value, int):
            errors.append(f'{field}: expected int, got {type(value).__name__}')
            continue
        elif field_type == 'str' and not isinstance(value, str):
            errors.append(f'{field}: expected str, got {type(value).__name__}')
            continue
        elif field_type == 'float' and not isinstance(value, (int, float)):
            errors.append(f'{field}: expected number, got {type(value).__name__}')
            continue

        # Range validation for numbers
        if isinstance(value, (int, float)):
            if 'min' in constraints and value < constraints['min']:
                errors.append(f'{field}: value {value} below minimum {constraints["min"]}')
            if 'max' in constraints and value > constraints['max']:
                errors.append(f'{field}: value {value} above maximum {constraints["max"]}')

        # String length validation
        if isinstance(value, str):
            if 'min_length' in constraints and len(value) < constraints['min_length']:
                errors.append(f'{field}: length {len(value)} below minimum {constraints["min_length"]}')
            if 'max_length' in constraints and len(value) > constraints['max_length']:
                errors.append(f'{field}: length {len(value)} above maximum {constraints["max_length"]}')

    return {'valid': len(errors) == 0, 'errors': errors}


def graph_analyzer(edges: list, vertices: int) -> dict:
    """
    Analyze properties of a graph given its edges and vertex count.

    Args:
        edges (list): List of tuples representing edges (source, destination).
            Must satisfy: isinstance(edges, list)
            Must satisfy: all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges)
            Must satisfy: all(isinstance(v, int) and 0 <= v < vertices for edge in edges for v in edge)
        vertices (int): Number of vertices in the graph.
            Must satisfy: vertices > 0
            Must satisfy: vertices <= 1000

    Returns:
        dict: Graph properties including connectivity, cycles, etc.

    Example:
        >>> graph_analyzer([(0, 1), (1, 2), (2, 0)], 3)
        {'vertices': 3, 'edges': 3, 'has_cycle': True, 'max_degree': 2}
        >>> graph_analyzer([(0, 1), (1, 2)], 3)
        {'vertices': 3, 'edges': 2, 'has_cycle': False, 'max_degree': 2}
    """
    if vertices <= 0 or vertices > 1000:
        raise ValueError("Vertices must be between 1 and 1000")

    if not isinstance(edges, list):
        raise TypeError("Edges must be a list")

    if not all(isinstance(edge, tuple) and len(edge) == 2 for edge in edges):
        raise ValueError("Each edge must be a tuple of length 2")

    if not all(isinstance(v, int) and 0 <= v < vertices for edge in edges for v in edge):
        raise ValueError("All vertices in edges must be valid integers")

    # Build adjacency list
    adj_list = {i: [] for i in range(vertices)}
    degree = {i: 0 for i in range(vertices)}

    for src, dst in edges:
        adj_list[src].append(dst)
        degree[src] += 1
        degree[dst] += 1

    # Detect cycles using DFS
    visited = set()
    rec_stack = set()
    has_cycle = False

    def dfs(vertex):
        nonlocal has_cycle
        if vertex in rec_stack:
            has_cycle = True
            return
        if vertex in visited:
            return

        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in adj_list[vertex]:
            dfs(neighbor)

        rec_stack.remove(vertex)

    for v in range(vertices):
        if v not in visited:
            dfs(v)

    return {
        'vertices': vertices,
        'edges': len(edges),
        'has_cycle': has_cycle,
        'max_degree': max(degree.values()) if degree else 0,
        'min_degree': min(degree.values()) if degree else 0,
        'isolated_vertices': sum(1 for d in degree.values() if d == 0)
    }


def inventory_manager(items: dict, operation: str, item_id: str, quantity: int = 0) -> dict:
    """
    Manage inventory operations with detailed validation.

    Args:
        items (dict): Current inventory state.
            Must satisfy: isinstance(items, dict)
            Must satisfy: all(isinstance(k, str) for k in items.keys())
            Must satisfy: all(isinstance(v, dict) and 'quantity' in v and 'price' in v for v in items.values())
        operation (str): Operation to perform.
            Must satisfy: operation in ['add', 'remove', 'update', 'check', 'value']
        item_id (str): Identifier for the item.
            Must satisfy: len(item_id) > 0
            Must satisfy: item_id.isalnum()
        quantity (int): Quantity for the operation (if applicable).
            Must satisfy: quantity >= 0

    Returns:
        dict: Result of the inventory operation.

    Example:
        >>> inventory_manager({'item1': {'quantity': 10, 'price': 5.0}}, 'check', 'item1')
        {'status': 'success', 'item': 'item1', 'quantity': 10, 'price': 5.0}
        >>> inventory_manager({}, 'add', 'item2', 5)
        {'status': 'error', 'message': 'Cannot add to empty inventory without price'}
    """
    valid_operations = ['add', 'remove', 'update', 'check', 'value']

    if not isinstance(items, dict):
        return {'status': 'error', 'message': 'Items must be a dictionary'}

    if operation not in valid_operations:
        return {'status': 'error', 'message': f'Invalid operation: {operation}'}

    if not item_id or not item_id.isalnum():
        return {'status': 'error', 'message': 'Item ID must be non-empty alphanumeric'}

    if quantity < 0:
        return {'status': 'error', 'message': 'Quantity cannot be negative'}

    # Validate existing items structure
    for item_key, item_data in items.items():
        if not isinstance(item_data, dict) or 'quantity' not in item_data or 'price' not in item_data:
            return {'status': 'error', 'message': f'Invalid item data for {item_key}'}

    if operation == 'check':
        if item_id in items:
            item = items[item_id]
            return {
                'status': 'success',
                'item': item_id,
                'quantity': item['quantity'],
                'price': item['price']
            }
        else:
            return {'status': 'error', 'message': f'Item {item_id} not found'}

    elif operation == 'remove':
        if item_id not in items:
            return {'status': 'error', 'message': f'Item {item_id} not found'}

        if items[item_id]['quantity'] < quantity:
            return {'status': 'error', 'message': 'Insufficient quantity'}

        items[item_id]['quantity'] -= quantity
        return {'status': 'success', 'removed': quantity, 'remaining': items[item_id]['quantity']}

    elif operation == 'value':
        total_value = sum(item['quantity'] * item['price'] for item in items.values())
        return {'status': 'success', 'total_value': total_value, 'item_count': len(items)}

    elif operation in ['add', 'update']:
        return {'status': 'error', 'message': 'Cannot add to empty inventory without price'}


def text_analyzer(text: str, analysis_type: str, language: str = 'english') -> dict:
    """
    Perform various text analysis operations.

    Args:
        text (str): The text to analyze.
            Must satisfy: len(text) > 0
            Must satisfy: len(text) <= 10000
        analysis_type (str): Type of analysis to perform.
            Must satisfy: analysis_type in ['basic', 'sentiment', 'readability', 'keywords']
        language (str): Language of the text.
            Must satisfy: language in ['english', 'spanish', 'french', 'german']

    Returns:
        dict: Analysis results based on the analysis type.

    Example:
        >>> text_analyzer("Hello world!", "basic")
        {'word_count': 2, 'char_count': 12, 'sentence_count': 1, 'avg_word_length': 5.5}
        >>> text_analyzer("", "basic")
        {'error': 'Text cannot be empty'}
    """
    if not text:
        return {'error': 'Text cannot be empty'}

    if len(text) > 10000:
        return {'error': 'Text too long (max 10000 characters)'}

    valid_types = ['basic', 'sentiment', 'readability', 'keywords']
    valid_languages = ['english', 'spanish', 'french', 'german']

    if analysis_type not in valid_types:
        return {'error': f'Invalid analysis type: {analysis_type}'}

    if language not in valid_languages:
        return {'error': f'Unsupported language: {language}'}

    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    if sentences == 0:
        sentences = 1

    if analysis_type == 'basic':
        return {
            'word_count': len(words),
            'char_count': len(text),
            'sentence_count': sentences,
            'avg_word_length': sum(len(word.strip('.,!?')) for word in words) / len(words) if words else 0,
            'language': language
        }

    elif analysis_type == 'sentiment':
        # Mock sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst']

        text_lower = text.lower()
        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)

        if positive_score > negative_score:
            sentiment = 'positive'
        elif negative_score > positive_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return {
            'sentiment': sentiment,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'confidence': abs(positive_score - negative_score) / max(len(words), 1)
        }

    elif analysis_type == 'readability':
        avg_sentence_length = len(words) / sentences
        long_words = sum(1 for word in words if len(word.strip('.,!?')) > 6)

        # Simplified readability score
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (long_words / len(words)))

        if readability_score >= 90:
            level = 'very_easy'
        elif readability_score >= 80:
            level = 'easy'
        elif readability_score >= 70:
            level = 'fairly_easy'
        elif readability_score >= 60:
            level = 'standard'
        elif readability_score >= 50:
            level = 'fairly_difficult'
        else:
            level = 'difficult'

        return {
            'readability_score': readability_score,
            'readability_level': level,
            'avg_sentence_length': avg_sentence_length,
            'long_words_ratio': long_words / len(words) if words else 0
        }

    elif analysis_type == 'keywords':
        # Simple keyword extraction
        word_freq = {}
        for word in words:
            clean_word = word.lower().strip('.,!?')
            if len(clean_word) > 3:  # Ignore short words
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1

        # Sort by frequency and take top 5
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'keywords': [word for word, freq in keywords],
            'keyword_frequencies': dict(keywords),
            'total_unique_words': len(word_freq),
            'vocabulary_richness': len(word_freq) / len(words) if words else 0
        }


def crypto_hasher(data: str, algorithm: str, iterations: int = 1) -> dict:
    """
    Generate cryptographic hashes with various algorithms.

    Args:
        data (str): The data to hash.
            Must satisfy: len(data) > 0
            Must satisfy: len(data) <= 1000000
        algorithm (str): The hashing algorithm to use.
            Must satisfy: algorithm in ['md5', 'sha1', 'sha256', 'sha512']
        iterations (int): Number of hash iterations (for key stretching).
            Must satisfy: 1 <= iterations <= 10000

    Returns:
        dict: Hash results and metadata.

    Example:
        >>> crypto_hasher("hello", "sha256")
        {'hash': '2cf24dba4f21d4288...', 'algorithm': 'sha256', 'length': 64}
        >>> crypto_hasher("", "sha256")
        {'error': 'Data cannot be empty'}
    """
    import hashlib

    if not data:
        return {'error': 'Data cannot be empty'}

    if len(data) > 1000000:
        return {'error': 'Data too large (max 1MB)'}

    valid_algorithms = ['md5', 'sha1', 'sha256', 'sha512']
    if algorithm not in valid_algorithms:
        return {'error': f'Unsupported algorithm: {algorithm}'}

    if not (1 <= iterations <= 10000):
        return {'error': 'Iterations must be between 1 and 10000'}

    # Get the hash function
    hash_func = getattr(hashlib, algorithm)

    # Perform iterative hashing
    result = data.encode('utf-8')
    for _ in range(iterations):
        result = hash_func(result).digest()

    # Convert to hex string
    hex_hash = result.hex()

    return {
        'hash': hex_hash,
        'algorithm': algorithm,
        'iterations': iterations,
        'length': len(hex_hash),
        'input_length': len(data),
        'entropy_estimate': len(set(data)) / len(data)
    }


def network_validator(ip_address: str, port: int, protocol: str) -> dict:
    """
    Validate network configuration parameters.

    Args:
        ip_address (str): IP address to validate.
            Must satisfy: len(ip_address) >= 7  # minimum "0.0.0.0"
            Must satisfy: ip_address.count('.') == 3
            Must satisfy: all(part.isdigit() and 0 <= int(part) <= 255 for part in ip_address.split('.'))
        port (int): Port number to validate.
            Must satisfy: 1 <= port <= 65535
        protocol (str): Network protocol.
            Must satisfy: protocol.upper() in ['TCP', 'UDP', 'ICMP']

    Returns:
        dict: Validation results and network information.

    Example:
        >>> network_validator("192.168.1.1", 80, "TCP")
        {'valid': True, 'ip_class': 'C', 'port_type': 'well_known', 'protocol': 'TCP'}
        >>> network_validator("256.1.1.1", 80, "TCP")
        {'valid': False, 'error': 'Invalid IP address format'}
    """
    # Validate IP address
    if len(ip_address) < 7 or ip_address.count('.') != 3:
        return {'valid': False, 'error': 'Invalid IP address format'}

    try:
        parts = ip_address.split('.')
        if not all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            return {'valid': False, 'error': 'Invalid IP address range'}
    except ValueError:
        return {'valid': False, 'error': 'Invalid IP address format'}

    # Validate port
    if not (1 <= port <= 65535):
        return {'valid': False, 'error': 'Port must be between 1 and 65535'}

    # Validate protocol
    valid_protocols = ['TCP', 'UDP', 'ICMP']
    if protocol.upper() not in valid_protocols:
        return {'valid': False, 'error': f'Invalid protocol: {protocol}'}

    # Determine IP class
    first_octet = int(parts[0])
    if 1 <= first_octet <= 126:
        ip_class = 'A'
    elif 128 <= first_octet <= 191:
        ip_class = 'B'
    elif 192 <= first_octet <= 223:
        ip_class = 'C'
    else:
        ip_class = 'Other'

    # Determine port type
    if 1 <= port <= 1023:
        port_type = 'well_known'
    elif 1024 <= port <= 49151:
        port_type = 'registered'
    else:
        port_type = 'dynamic'

    # Check for private IP ranges
    is_private = (
        (first_octet == 10) or
        (first_octet == 172 and 16 <= int(parts[1]) <= 31) or
        (first_octet == 192 and int(parts[1]) == 168)
    )

    return {
        'valid': True,
        'ip_address': ip_address,
        'ip_class': ip_class,
        'is_private': is_private,
        'port': port,
        'port_type': port_type,
        'protocol': protocol.upper(),
        'is_secure_port': port in [22, 443, 993, 995]
    }
