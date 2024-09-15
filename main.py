def test_fn(a, b, c):
    print(f'{a = }, {b = }, {c = }')


default_args = {'a': 1, 'b': 2, 'c': 3}
update_dict = {'c': 5, 'a': 6}
test_fn(**{**default_args, **update_dict})
