def reverse_string(s):
    """Reverse a string.

    Args:
        s (str): Input string

    Returns:
        str: Reversed string

    Examples:
        >>> reverse_string("hello")
        'olleh'
        >>> reverse_string("abc")
        'cba'
        >>> reverse_string("")
        ''
        >>> reverse_string("a")
        'a'

    Preconditions:
        - len(s) <= 1000
    """
    return s[::-1]


def binary_search(arr, target):
    """Perform binary search on a sorted array.

    Args:
        arr (list): Sorted list of integers
        target (int([1, 3, 5, 7, 9], 6)
        -1
        >>> binary_search([], 5)
        -1

    Preconditions:
        - len(arr) >= 0
        - len(arr) <= 1000
        - all): Value to search for

    Returns:
        int: Index of target if found, -1 otherwise

    Examples:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 1)
        0
        >>> binary_search([1, 3, 5, 7, 9], 9)
        4
        >>> binary_search(arr[i] <= arr[i+1] for i in range(len(arr)-1)) if len(arr) > 1
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def merge_sorted_lists(list1, list2):
    """Merge two sorted lists into one sorted list.

    Args:
        list1 (list): First sorted list
        list2 (list): Second sorted list

    Returns:
        list: Merged sorted list

    Examples:
        >>> merge_sorted_lists([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
        >>> merge_sorted_lists([1, 2], [3, 4])
        [1, 2, 3, 4]
        >>> merge_sorted_lists([], [1, 2, 3])
        [1, 2, 3]
        >>> merge_sorted_lists([1, 2, 3], [])
        [1, 2, 3]

    Preconditions:
        - len(list1) <= 100
        - len(list2) <= 100
        - all(list1[i] <= list1[i+1] for i in range(len(list1)-1)) if len(list1) > 1
        - all(list2[i] <= list2[i+1] for i in range(len(list2)-1)) if len(list2) > 1
    """
    result = []
    i = j = 0

    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1

    result.extend(list1[i:])
    result.extend(list2[j:])
    return result


def count_words(text):
    """Count the number of words in a text string.

    Args:
        text (str): Input text

    Returns:
        int: Number of words

    Examples:
        >>> count_words("hello world")
        2
        >>> count_words("python is great")
        3
        >>> count_words("")
        0
        >>> count_words("   spaces   ")
        1
        >>> count_words("one")
        1

    Preconditions:
        - len(text) <= 10000
    """
    return len(text.split())


def find_max(numbers):
    """Find the maximum value in a list of numbers.

    Args:
        numbers (list): List of numeric values

    Returns:
        float: Maximum value

    Examples:
        >>> find_max([1, 5, 3, 9, 2])
        9
        >>> find_max([1.5, 2.7, 1.2])
        2.7
        >>> find_max([-1, -5, -2])
        -1
        >>> find_max([42])
        42

    Preconditions:
        - len(numbers) > 0
        - len(numbers) <= 1000
        - all(isinstance(x, (int, float)) for x in numbers)
    """
    return max(numbers)
