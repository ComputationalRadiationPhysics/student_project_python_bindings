from contextlib import contextmanager

@contextmanager
def sync_open(buffer, is_synced):
    if is_synced:
        buffer.read()
        yield buffer.buffer
        buffer.write()
    else:
        yield buffer
        