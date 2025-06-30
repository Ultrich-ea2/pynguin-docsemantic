"""Demo-Modul für Pynguin – generiert automatisch Tests."""

def add(x: int, y: int):
    """
    >>> add(3, 5)
    8
    
    >>> add(0, 5)
    5


    Preconditions
    x >37
    """
    if x <= 0 or y <= 0:
        raise ValueError("x und y müssen > 0 sein")
    return x + y

