import textwrap

from colorama import Fore, Style, init

init(autoreset=True)


def print_dict_tree(d, indent=2, keys_only=False, wrap_width=60):
    """
    Print the keys and values of a nested dictionary in a tree-like structure with different colors.

    Parameters:
    - d: dict, the dictionary to print.
    - indent: int, the number of spaces for indentation.
    - keys_only: bool, if True, only print keys.
    - width: int, the maximum width for text wrapping.
    """

    def print_tree(d, level=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * (level * indent) + Fore.CYAN + str(key) + ":")
                print_tree(value, level + 1)
            else:
                if keys_only:
                    print(" " * (level * indent) + Fore.CYAN + str(key))
                else:
                    wrapped_value = textwrap.fill(
                        str(value),
                        width=wrap_width,
                        subsequent_indent=" " * (indent + 2),
                    )
                    print(
                        " " * (level * indent)
                        + Fore.CYAN
                        + str(key)
                        + ":"
                        + Style.RESET_ALL
                        + " "
                        + wrapped_value
                    )

        Style.RESET_ALL

    print_tree(d)


# --------------------------------- Dividers --------------------------------- #


def divider(text="", char="=", line_max=79, show=True):
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. " "Received: {}".format(char)
        )
    deco = char * (int(round((line_max - len(text))) / 2) - 2)
    text = " {} ".format(text) if text else ""
    text = f"{deco}{text}{deco}"
    if len(text) < line_max:
        text = text + char * (line_max - len(text))
    if show:
        print(text)
    return text


def plain_divider(char="=", line_max=79, show=True):
    if len(char) != 1:
        raise ValueError(
            "Divider chars need to be one character long. " "Received: {}".format(char)
        )
    deco = char * line_max
    text = f"{deco}"
    if show:
        print(text)
    return text
