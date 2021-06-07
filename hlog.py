from absl import flags as a_flags

from contextlib import contextmanager
import logging
import threading
import time
import sys

console = logging.StreamHandler(stream=sys.stdout)
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d] %(message)s', "%H:%M:%S")
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('hlog').addHandler(console)
# Now, we can log to the root logger, or any other logger. First the root..
logger = logging.getLogger('hlog')  # Loggerr
# logger.setLevel(logging.INFO)
# logger.handlers[0].stream = sys.stdout
logger.propagate = False

print(logger.handlers)

state = threading.local()
state.path = []

@contextmanager
def task(name, timer=True):
    state.path.append(str(name))
    begin = time.time()
    yield
    end = time.time()
    if timer:
        logger.info('%s{%0.2fs}' % ('/'.join(state.path), end - begin))
    state.path.pop()

def flags():
    flags = a_flags.FLAGS.flags_into_string().split("\n")
    for flag in flags:
        logger.info("# %s", flag)
        
def group(name):
    return task(name, timer=False)

def log(value):
    if isinstance(value, float):
        value = "%0.4f" % value
    logger.info('%s %s' % ('/'.join(state.path), str(value)))

def value(name, value):
    with task(name, timer=False):
        log(value)

def loop(template, coll=None, counter=None, timer=True):
    assert not (coll is None and counter is None)
    if coll is None:
        seq = zip(counter, counter)
    elif counter is None:
        seq = enumerate(coll)
    else:
        assert len(counter) == len(coll)
        seq = zip(counter, coll)
    for i, item in seq:
        with task(template % i, timer):
            yield item

def fn(name, timer=True):
    def wrap(underlying):
        def wrapped(*args, **kwargs):
            with task(name, timer):
                result = underlying(*args, **kwargs)
            return result
        return wrapped
    return wrap
