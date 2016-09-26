import sys
import json
import time
import uuid
import beanstalkc as bean

from subprocess import check_output, STDOUT
from multiprocessing import Process

LOCALHOST = '127.0.0.1'
DEFAULT_BEANSTALKD = 14444

def listen(func):
    """
    This function is for use in external scripts in order to listen on the
    queue and perform tasks. In particular, create a function which takes the
    arguments you would pass to `launch_all`s jobs parameter, then at the
    bottom of the script add:

    if __name__ == '__main__':
        peri.test.beanstalk.listen(my_function_name)
    """
    import argparse

    parser = argparse.ArgumentParser(description="PERI remote listener")
    parser.add_argument('--port', '-p', type=int, help="beanstalkd port")
    parser.add_argument('--processes', '-n', type=int, help="number of processes to launch")
    
    args = vars(parser.parse_args())

    port = int(args.get('port') or DEFAULT_BEANSTALKD)
    proc = int(args.get('processes') or 1)

    def _listen(func):
        bsd = bean.Connection(host=LOCALHOST, port=port)
        while True:
            job = bsd.reserve()
            func(json.loads(job.body))

    for i in xrange(proc):
        Process(target=_listen, args=(func,)).start()

def launch_all(script, hosts, jobs, nprocs=1, bean_port=DEFAULT_BEANSTALKD,
        docopy=True, env=None):
    """
    Launch a group of workers spread across a group of machines which listen on
    a beanstalkd queue and perform tasks. This main process controls the others
    by launching the script through ssh and killing them when this main process
    is killed. Note that this launch happens over ssh, so it is recommended
    that you have ssh-agent running to avoid a ridiculous number of password
    entries.

    Parameters:
    -----------
    script : string
        The filename of the script to run

    hosts : list of strings
        The hosts on which to launch the workers

    nprocs : int or list of ints
        Number of worker processes to spawn on each worker, the number of
        concurrent jobs it can run

    bean_port : int
        beanstalk port which to open

    docopy : boolean
        whether to remotely copy the script to the new machine

    env : dictionary
        extra environment variables which need to be set to run
    """

    def beanstalk(bean_port=DEFAULT_BEANSTALKD):
        bean = 'beanstalkd -l {} -p {}'.format(LOCALHOST, bean_port)
        check_output(bean, shell=True)

    def copy_script(script, host):
        tmpfile = '/tmp/{}.py'.format(uuid.uuid4().hex)
        ssh = 'scp {script} {host}:{tmpfile}'.format(
            script=script, host=host, tmpfile=tmpfile
        )
        check_output(ssh, shell=True)
        return tmpfile

    def launch(script, host, nprocs=1, bean_port=DEFAULT_BEANSTALKD, env=None, index=0):
        env = env or {}
        env = 'export PYTHONPATH=$PYTHONPATH;'

        log = '> /tmp/launcher-{}.log 2>&1'.format(index)
        fwd = '-L{}:localhost:{}'.format(bean_port, bean_port)
        cmd = "{} python {} --port={} --processes={}".format(env, script, bean_port, nprocs)
        ssh = 'ssh {forward} {host} "{cmd} {log}"'.format(forward=fwd, host=host, cmd=cmd, log=log)
        check_output(ssh, shell=True)

    env = env or {}
    procs = {}

    # start up the beanstalk process first
    procs[-1] = Process(target=beanstalk, args=(bean_port,))
    procs[-1].start()
    time.sleep(1)

    nprocs = [nprocs]*len(hosts) if not hasattr(nprocs, '__iter__') else nprocs

    for i, (host, nproc) in enumerate(zip(hosts, nprocs)):
        tmpscript = script if not docopy else copy_script(script, host)
        procs[i] = Process(target=launch, args=(tmpscript, host, nproc, bean_port, env, i))
        procs[i].start()

    def signal_handler():
        print "Sending signal to flush, waiting 60 sec until forced shutdown..."
        for p in procs.values():
            p.join(timeout=60)
        sys.exit(1)

    bsd = bean.Connection(host=LOCALHOST, port=bean_port, connect_timeout=10)
    for job in jobs:
        bsd.put(json.dumps(job))

    try:
        while True:
            for i,k in procs.iteritems():
                procs[i].join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        signal_handler()    

