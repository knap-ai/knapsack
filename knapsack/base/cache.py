from threading import Lock
from cachetools.keys import hashkey


def cache_custom_key(*args, **kwargs):
    items = [v for _, v in kwargs.items()]
    items += list(args)

    modded_args = []
    for arg in args: 
        if type(arg) in [list, tuple]:
            modded_args += arg 
        else: 
            modded_args.append(arg)

    modded_kwargs = {}
    for k, v in kwargs.items():
        if type(v) in [list, tuple]:
            modded_args += v 
        else: 
            modded_kwargs[k] = v

    return hashkey(*modded_args, tuple(modded_kwargs.items()))


def clear_cache_for_collection(cache, cache_lock: Lock, collection_to_clear: str):
    with cache_lock:
        keys = cache.keys()
        keys_to_delete = []
        for key in keys:
            elems = []
            for key_elem in key:
                if type(key_elem) == tuple:
                    elems += key_elem
                else:
                    elems.append(key_elem)

            for elem in elems:
                if type(elem) == tuple and elem[0] == 'collection' and elem[1] == collection_to_clear:
                    keys_to_delete.append(key)
                    break

        for key in keys_to_delete:
            del cache[key]
