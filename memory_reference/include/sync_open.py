from contextlib import contextmanager

@contextmanager
def sync_open_rw(buffer, is_synced):
    if is_synced:
        yield buffer
    else:
        buffer.read()
        yield buffer.buffer
        buffer.write()
    
@contextmanager
def sync_open_r(buffer, is_synced):
    if is_synced:
        yield buffer
    else:
        buffer.read()
        yield buffer.buffer

@contextmanager
def sync_open_w(buffer, is_synced):
    if is_synced:
        yield buffer
    else:
        yield buffer.buffer
        buffer.write()
        