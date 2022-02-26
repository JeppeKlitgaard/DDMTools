# Backport of str.removeprefix.
# TODO Remove when minimum python version is 3.9 or above
def removeprefix(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix) :]

    return string
