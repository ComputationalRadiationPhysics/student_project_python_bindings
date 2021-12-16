from contextlib import contextmanager

@contextmanager
def sync_open_rw(buffer, is_synced):
    if is_synced:
        buffer.read()
        yield buffer.buffer
        buffer.write()
    else:
        yield buffer
    
@contextmanager
def sync_open_r(buffer, is_synced):
    if is_synced:
        buffer.read()
        yield buffer.buffer
    else:
        yield buffer

@contextmanager
def sync_open_w(buffer, is_synced):
    if is_synced:
        yield buffer.buffer
        buffer.write()
    else:
        yield buffer
        