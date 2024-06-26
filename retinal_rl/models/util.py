from typing import List, Union


def assert_list(
    list_candidate: Union[int, List[int]],
    len_list: int,
) -> List[int]:
    if isinstance(list_candidate, int):
        _list = [list_candidate] * len_list
    else:
        if len(list_candidate) != len_list:
            raise AssertionError(
                "The length of the list does not match the expected length: "
                + str(len(list_candidate))
                + " != "
                + str(len_list)
            )
        _list = list_candidate
    return _list
