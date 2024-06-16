def assert_list(list_candidate, len_list, dtype=int):
    if isinstance(list_candidate, dtype):
        _list = [list_candidate] * len_list
    else:
        if len(list_candidate) != len_list:
           raise AssertionError("The length of the list does not match the expected length: "+str(len(list_candidate))+" != " + str(len_list))
        _list = list_candidate
    return _list
