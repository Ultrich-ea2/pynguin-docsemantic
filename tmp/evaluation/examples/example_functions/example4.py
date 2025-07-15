class DataPipeline:
    """
    A data processing pipeline with interconnected methods for comprehensive testing.

    This class demonstrates complex method interactions, state management, and various
    data processing patterns that test Pynguin's ability to understand semantic
    relationships between methods.
    """

    def __init__(self, name: str, max_size: int = 1000):
        """
        Initialize the data pipeline.

        Args:
            name (str): Name of the pipeline.
                Must satisfy: len(name) > 0
                Must satisfy: name.replace('_', '').replace('-', '').isalnum()
            max_size (int): Maximum number of items the pipeline can hold.
                Must satisfy: 10 <= max_size <= 10000

        Example:
            >>> pipeline = DataPipeline("test_pipeline", 100)
            >>> pipeline.get_status()['name']
            'test_pipeline'
        """
        if not name or len(name) == 0:
            raise ValueError("Pipeline name cannot be empty")

        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Pipeline name must be alphanumeric (with _ and - allowed)")

        if not (10 <= max_size <= 10000):
            raise ValueError("Max size must be between 10 and 10000")

        self._name = name
        self._max_size = max_size
        self._data = []
        self._processed_count = 0
        self._error_count = 0
        self._filters = []
        self._transformers = []
        self._validators = []
        self._metadata = {}
        self._state = "initialized"

    def add_data(self, items: list) -> dict:
        """
        Add data items to the pipeline.

        Args:
            items (list): List of data items to add.
                Must satisfy: isinstance(items, list)
                Must satisfy: len(items) > 0
                Must satisfy: len(self._data) + len(items) <= self._max_size

        Returns:
            dict: Result of the add operation.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> result = pipeline.add_data([1, 2, 3, "hello"])
            >>> result['added_count']
            4
        """
        if not isinstance(items, list):
            return {'success': False, 'error': 'Items must be a list'}

        if len(items) == 0:
            return {'success': False, 'error': 'Cannot add empty list'}

        if len(self._data) + len(items) > self._max_size:
            return {
                'success': False,
                'error': f'Would exceed max size ({self._max_size})',
                'current_size': len(self._data),
                'attempted_add': len(items)
            }

        self._data.extend(items)
        self._state = "loaded"

        return {
            'success': True,
            'added_count': len(items),
            'total_size': len(self._data),
            'remaining_capacity': self._max_size - len(self._data)
        }

    def add_filter(self, filter_type: str, **kwargs) -> dict:
        """
        Add a data filter to the pipeline.

        Args:
            filter_type (str): Type of filter to add.
                Must satisfy: filter_type in ['type', 'range', 'length', 'pattern']
            **kwargs: Filter-specific parameters.

        Returns:
            dict: Result of adding the filter.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> result = pipeline.add_filter('type', target_type='int')
            >>> result['success']
            True
        """
        valid_filters = ['type', 'range', 'length', 'pattern']

        if filter_type not in valid_filters:
            return {'success': False, 'error': f'Invalid filter type: {filter_type}'}

        filter_config = {'type': filter_type, **kwargs}

        # Validate filter configuration
        if filter_type == 'type':
            if 'target_type' not in kwargs:
                return {'success': False, 'error': 'Type filter requires target_type parameter'}
            valid_types = ['int', 'float', 'str', 'bool', 'list', 'dict']
            if kwargs['target_type'] not in valid_types:
                return {'success': False, 'error': f'Invalid target_type: {kwargs["target_type"]}'}

        elif filter_type == 'range':
            if 'min_val' not in kwargs and 'max_val' not in kwargs:
                return {'success': False, 'error': 'Range filter requires min_val or max_val'}
            if 'min_val' in kwargs and 'max_val' in kwargs:
                if kwargs['min_val'] > kwargs['max_val']:
                    return {'success': False, 'error': 'min_val cannot be greater than max_val'}

        elif filter_type == 'length':
            if 'min_length' not in kwargs and 'max_length' not in kwargs:
                return {'success': False, 'error': 'Length filter requires min_length or max_length'}

        elif filter_type == 'pattern':
            if 'regex' not in kwargs:
                return {'success': False, 'error': 'Pattern filter requires regex parameter'}

        self._filters.append(filter_config)

        return {
            'success': True,
            'filter_count': len(self._filters),
            'filter_added': filter_config
        }

    def add_transformer(self, transform_type: str, **kwargs) -> dict:
        """
        Add a data transformer to the pipeline.

        Args:
            transform_type (str): Type of transformation.
                Must satisfy: transform_type in ['uppercase', 'lowercase', 'multiply', 'round', 'format']
            **kwargs: Transformer-specific parameters.

        Returns:
            dict: Result of adding the transformer.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> result = pipeline.add_transformer('multiply', factor=2)
            >>> result['success']
            True
        """
        valid_transformers = ['uppercase', 'lowercase', 'multiply', 'round', 'format']

        if transform_type not in valid_transformers:
            return {'success': False, 'error': f'Invalid transformer type: {transform_type}'}

        transform_config = {'type': transform_type, **kwargs}

        # Validate transformer configuration
        if transform_type == 'multiply':
            if 'factor' not in kwargs:
                return {'success': False, 'error': 'Multiply transformer requires factor parameter'}
            if not isinstance(kwargs['factor'], (int, float)):
                return {'success': False, 'error': 'Factor must be a number'}

        elif transform_type == 'round':
            if 'decimals' in kwargs and not isinstance(kwargs['decimals'], int):
                return {'success': False, 'error': 'Decimals must be an integer'}

        elif transform_type == 'format':
            if 'template' not in kwargs:
                return {'success': False, 'error': 'Format transformer requires template parameter'}

        self._transformers.append(transform_config)

        return {
            'success': True,
            'transformer_count': len(self._transformers),
            'transformer_added': transform_config
        }

    def apply_filters(self) -> dict:
        """
        Apply all configured filters to the data.

        Returns:
            dict: Results of the filtering operation.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, "hello", 3.14])
            >>> pipeline.add_filter('type', target_type='int')
            >>> result = pipeline.apply_filters()
            >>> result['filtered_count']
            2
        """
        if not self._data:
            return {'success': False, 'error': 'No data to filter'}

        if not self._filters:
            return {'success': True, 'message': 'No filters configured', 'data_count': len(self._data)}

        original_count = len(self._data)
        filtered_data = self._data.copy()

        for filter_config in self._filters:
            filter_type = filter_config['type']

            if filter_type == 'type':
                target_type = filter_config['target_type']
                type_map = {
                    'int': int, 'float': float, 'str': str,
                    'bool': bool, 'list': list, 'dict': dict
                }
                filtered_data = [item for item in filtered_data if isinstance(item, type_map[target_type])]

            elif filter_type == 'range':
                min_val = filter_config.get('min_val', float('-inf'))
                max_val = filter_config.get('max_val', float('inf'))
                filtered_data = [
                    item for item in filtered_data
                    if isinstance(item, (int, float)) and min_val <= item <= max_val
                ]

            elif filter_type == 'length':
                min_length = filter_config.get('min_length', 0)
                max_length = filter_config.get('max_length', float('inf'))
                filtered_data = [
                    item for item in filtered_data
                    if hasattr(item, '__len__') and min_length <= len(item) <= max_length
                ]

        self._data = filtered_data
        filtered_count = original_count - len(filtered_data)
        self._state = "filtered"

        return {
            'success': True,
            'original_count': original_count,
            'remaining_count': len(filtered_data),
            'filtered_count': filtered_count,
            'filters_applied': len(self._filters)
        }

    def apply_transformations(self) -> dict:
        """
        Apply all configured transformations to the data.

        Returns:
            dict: Results of the transformation operation.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3])
            >>> pipeline.add_transformer('multiply', factor=2)
            >>> result = pipeline.apply_transformations()
            >>> pipeline.get_data()
            [2, 4, 6]
        """
        if not self._data:
            return {'success': False, 'error': 'No data to transform'}

        if not self._transformers:
            return {'success': True, 'message': 'No transformers configured', 'data_count': len(self._data)}

        transformed_data = []
        error_count = 0

        for item in self._data:
            current_item = item

            for transform_config in self._transformers:
                try:
                    current_item = self._apply_single_transformation(current_item, transform_config)
                except Exception as e:
                    error_count += 1
                    self._error_count += 1
                    # Keep original item if transformation fails
                    current_item = item
                    break

            transformed_data.append(current_item)

        self._data = transformed_data
        self._processed_count += len(transformed_data)
        self._state = "transformed"

        return {
            'success': True,
            'processed_count': len(transformed_data),
            'error_count': error_count,
            'transformers_applied': len(self._transformers)
        }

    def _apply_single_transformation(self, item, transform_config):
        """
        Apply a single transformation to an item.

        Args:
            item: The item to transform
            transform_config (dict): Configuration for the transformation

        Returns:
            The transformed item
        """
        transform_type = transform_config['type']

        if transform_type == 'uppercase':
            if isinstance(item, str):
                return item.upper()

        elif transform_type == 'lowercase':
            if isinstance(item, str):
                return item.lower()

        elif transform_type == 'multiply':
            if isinstance(item, (int, float)):
                return item * transform_config['factor']

        elif transform_type == 'round':
            if isinstance(item, (int, float)):
                decimals = transform_config.get('decimals', 0)
                return round(item, decimals)

        elif transform_type == 'format':
            template = transform_config['template']
            return template.format(item)

        return item

    def validate_data(self, validation_rules: dict) -> dict:
        """
        Validate data against provided rules.

        Args:
            validation_rules (dict): Rules for validation.
                Must satisfy: isinstance(validation_rules, dict)
                Must satisfy: len(validation_rules) > 0

        Returns:
            dict: Validation results.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3, 4, 5])
            >>> rules = {'min_count': 3, 'max_count': 10, 'all_numbers': True}
            >>> result = pipeline.validate_data(rules)
            >>> result['valid']
            True
        """
        if not isinstance(validation_rules, dict) or len(validation_rules) == 0:
            return {'valid': False, 'error': 'Validation rules must be a non-empty dictionary'}

        if not self._data:
            return {'valid': False, 'error': 'No data to validate'}

        validation_errors = []

        # Check count constraints
        if 'min_count' in validation_rules:
            if len(self._data) < validation_rules['min_count']:
                validation_errors.append(f'Data count {len(self._data)} below minimum {validation_rules["min_count"]}')

        if 'max_count' in validation_rules:
            if len(self._data) > validation_rules['max_count']:
                validation_errors.append(f'Data count {len(self._data)} above maximum {validation_rules["max_count"]}')

        # Check type constraints
        if 'all_numbers' in validation_rules and validation_rules['all_numbers']:
            if not all(isinstance(item, (int, float)) for item in self._data):
                validation_errors.append('Not all items are numbers')

        if 'all_strings' in validation_rules and validation_rules['all_strings']:
            if not all(isinstance(item, str) for item in self._data):
                validation_errors.append('Not all items are strings')

        # Check value constraints
        if 'min_value' in validation_rules:
            numeric_items = [item for item in self._data if isinstance(item, (int, float))]
            if numeric_items and min(numeric_items) < validation_rules['min_value']:
                validation_errors.append(
                    f'Minimum value {min(numeric_items)} below required {validation_rules["min_value"]}')

        if 'max_value' in validation_rules:
            numeric_items = [item for item in self._data if isinstance(item, (int, float))]
            if numeric_items and max(numeric_items) > validation_rules['max_value']:
                validation_errors.append(
                    f'Maximum value {max(numeric_items)} above allowed {validation_rules["max_value"]}')

        is_valid = len(validation_errors) == 0
        self._state = "validated" if is_valid else "validation_failed"

        return {
            'valid': is_valid,
            'errors': validation_errors,
            'data_count': len(self._data),
            'rules_checked': len(validation_rules)
        }

    def get_statistics(self) -> dict:
        """
        Get comprehensive statistics about the pipeline data.

        Returns:
            dict: Statistical information about the data.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3, 4, 5])
            >>> stats = pipeline.get_statistics()
            >>> stats['numeric_count']
            5
        """
        if not self._data:
            return {'error': 'No data available for statistics'}

        stats = {
            'total_count': len(self._data),
            'numeric_count': sum(1 for item in self._data if isinstance(item, (int, float))),
            'string_count': sum(1 for item in self._data if isinstance(item, str)),
            'boolean_count': sum(1 for item in self._data if isinstance(item, bool)),
            'list_count': sum(1 for item in self._data if isinstance(item, list)),
            'dict_count': sum(1 for item in self._data if isinstance(item, dict)),
            'none_count': sum(1 for item in self._data if item is None)
        }

        # Numeric statistics
        numeric_items = [item for item in self._data if isinstance(item, (int, float)) and not isinstance(item, bool)]
        if numeric_items:
            stats.update({
                'numeric_min': min(numeric_items),
                'numeric_max': max(numeric_items),
                'numeric_mean': sum(numeric_items) / len(numeric_items),
                'numeric_sum': sum(numeric_items)
            })

        # String statistics
        string_items = [item for item in self._data if isinstance(item, str)]
        if string_items:
            stats.update({
                'string_lengths': [len(s) for s in string_items],
                'avg_string_length': sum(len(s) for s in string_items) / len(string_items),
                'longest_string': max(string_items, key=len),
                'shortest_string': min(string_items, key=len)
            })

        return stats

    def process_pipeline(self, validation_rules: dict = None) -> dict:
        """
        Execute the complete pipeline: filter, transform, and optionally validate.

        Args:
            validation_rules (dict, optional): Rules for final validation.

        Returns:
            dict: Complete pipeline execution results.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3, "hello", 4.5])
            >>> pipeline.add_filter('type', target_type='int')
            >>> pipeline.add_transformer('multiply', factor=2)
            >>> result = pipeline.process_pipeline({'min_count': 1})
            >>> result['success']
            True
        """
        if not self._data:
            return {'success': False, 'error': 'No data to process'}

        results = {
            'success': True,
            'pipeline_name': self._name,
            'initial_data_count': len(self._data),
            'steps': []
        }

        # Apply filters
        if self._filters:
            filter_result = self.apply_filters()
            results['steps'].append(('filter', filter_result))
            if not filter_result['success']:
                results['success'] = False
                return results

        # Apply transformations
        if self._transformers:
            transform_result = self.apply_transformations()
            results['steps'].append(('transform', transform_result))
            if not transform_result['success']:
                results['success'] = False
                return results

        # Apply validation if rules provided
        if validation_rules:
            validation_result = self.validate_data(validation_rules)
            results['steps'].append(('validate', validation_result))
            if not validation_result['valid']:
                results['success'] = False

        results.update({
            'final_data_count': len(self._data),
            'total_processed': self._processed_count,
            'total_errors': self._error_count,
            'final_state': self._state
        })

        return results

    def get_data(self) -> list:
        """
        Get the current data in the pipeline.

        Returns:
            list: Current data items.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3])
            >>> pipeline.get_data()
            [1, 2, 3]
        """
        return self._data.copy()

    def get_status(self) -> dict:
        """
        Get the current status of the pipeline.

        Returns:
            dict: Pipeline status information.

        Example:
            >>> pipeline = DataPipeline("test_pipeline", 100)
            >>> status = pipeline.get_status()
            >>> status['name']
            'test_pipeline'
        """
        return {
            'name': self._name,
            'state': self._state,
            'data_count': len(self._data),
            'max_size': self._max_size,
            'remaining_capacity': self._max_size - len(self._data),
            'filter_count': len(self._filters),
            'transformer_count': len(self._transformers),
            'processed_count': self._processed_count,
            'error_count': self._error_count,
            'metadata': self._metadata.copy()
        }

    def clear_pipeline(self) -> dict:
        """
        Clear all data and reset the pipeline state.

        Returns:
            dict: Result of the clear operation.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_data([1, 2, 3])
            >>> result = pipeline.clear_pipeline()
            >>> result['cleared_items']
            3
        """
        cleared_items = len(self._data)
        self._data.clear()
        self._processed_count = 0
        self._error_count = 0
        self._state = "cleared"

        return {
            'success': True,
            'cleared_items': cleared_items,
            'state': self._state
        }

    def export_configuration(self) -> dict:
        """
        Export the current pipeline configuration.

        Returns:
            dict: Pipeline configuration that can be used to recreate the pipeline.

        Example:
            >>> pipeline = DataPipeline("test", 100)
            >>> pipeline.add_filter('type', target_type='int')
            >>> config = pipeline.export_configuration()
            >>> len(config['filters'])
            1
        """
        return {
            'name': self._name,
            'max_size': self._max_size,
            'filters': self._filters.copy(),
            'transformers': self._transformers.copy(),
            'metadata': self._metadata.copy(),
            'current_state': self._state,
            'data_count': len(self._data)
        }
