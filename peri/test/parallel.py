from builtins import range

import os
import sys
import json
import time
import uuid

from subprocess import check_output, STDOUT
from multiprocessing import Process

LOCALHOST = '127.0.0.1'
DEFAULT_BEANSTALKD = 14444

#=============================================================================
# Shared filesystem (NFS etc) parallel job runner
#=============================================================================
def get_next_job(start_prefix, end_prefix):
    """
    Given two globs, figure out the the difference in the file lists to
    determine which job to run next
    """
    from peri.test import analyze

    # jobs left to do
    f0 = analyze.sorted_files(start_prefix, return_num=True)
    n0 = [t[-2] for t in f0]

    # jobs that have been completed
    f1 = analyze.sorted_files(end_prefix, return_num=True)
    n1 = [t[-2] for t in f1]

    # numbers of elements yet to do
    nums = list(set(n0).difference(set(n1)))

    if len(nums) > 0:
        next_num = min(nums)
        next_file = [t[-1] for t in f0 if t[-2] == next_num][0]
        return next_file
    return None

def watch(func, file_pattern, postfix='run'):
    """
    Watch the filesystem for files that match `file_pattern` and run the ones
    that haven't been completed yet with function `func`
    """
    import argparse
    parser = argparse.ArgumentParser(description="PERI remote listener")
    parser.add_argument('--processes', '-n', type=int, help="number of processes to launch")

    args = vars(parser.parse_args())
    proc = int(args.get('processes') or 1)

    def _watch(func, file_pattern, postfix, index):
        # redirect the logs to separate files in /tmp
        sys.stdout = open('/tmp/peri-watch-{}.stdout'.format(index), "w", buffering=0)
        sys.stderr = open('/tmp/peri-watch-{}.stderr'.format(index), "w", buffering=0)

        def mark_done(f, postfix):
            return '{}-{}'.format(f, postfix)

        def next_file():
            return get_next_job(
                file_pattern,  mark_done(file_pattern, postfix)
            )

        filename = next_file()
        while filename:
            open(mark_done(filename, postfix), 'w').close()
            func(filename)
            filename = next_file()

    log.info('Launching listener processes')
    for i in range(proc):
        log.info('{}'.format(i))
        Process(target=_watch, args=(func, file_pattern, postfix, i)).start()
    log.info('.')

def launch_watchers(script, hosts, nprocs=1):
    """
    Launch the watch script on a bunch of workers which have uniform environment
    and filesystem for example via EC2 or any group of computers sharing home
    filesystem.

    Parameters:
    -----------
    script : string
        The filename of the script to run

    hosts : list of strings
        Names of the hosts on which to launch the watcher processes

    nprocs : integer
        Number of processes per host to run
    """
    def launch(script, host):
        cmd = 'python {} --processes={}'.format(script, nprocs)
        log = '> /tmp/peri-launcher.log 2>&1'
        sub = 'bash -c \'{cmd} {log}\''.format(cmd=cmd, log=log)
        ssh = sub if host == LOCALHOST else 'ssh {name} "{sub}"'.format(name=host, sub=sub)
        check_output(ssh, shell=True)

    procs = {}

    log.info('Starting job process for')
    for i, host in enumerate(hosts):
        log.info('... {}'.format(host))
        procs[i] = Process(target=launch, args=(script, host))
        procs[i].start()

    def signal_handler():
        log.info("Sending signal to flush, waiting 60 sec until forced shutdown...")
        for p in procs.values():
            p.join(timeout=60)
        sys.exit(1)

    try:
        while True:
            for i in procs.keys():
                procs[i].join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        signal_handler()

#=============================================================================
# SSH based parallel job runner
#=============================================================================
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
    import beanstalkc as bean

    parser = argparse.ArgumentParser(description="PERI remote listener")
    parser.add_argument('--port', '-p', type=int, help="beanstalkd port")
    parser.add_argument('--processes', '-n', type=int, help="number of processes to launch")

    args = vars(parser.parse_args())

    port = int(args.get('port') or DEFAULT_BEANSTALKD)
    proc = int(args.get('processes') or 1)

    def _listen(func, index):
        # redirect the logs to separate files in /tmp
        sys.stdout = open('/tmp/peri-listen-{}.stdout'.format(index), "w", buffering=0)
        sys.stderr = open('/tmp/peri-listen-{}.stderr'.format(index), "w", buffering=0)

        bsd = bean.Connection(host=LOCALHOST, port=port)
        while True:
            job = bsd.reserve()
            func(json.loads(job.body))

    log.info('Launching listener processes')
    for i in range(proc):
        log.info('{}'.format(i))
        Process(target=_listen, args=(func,i)).start()

def launch_all(script, hosts, jobs, bean_port=DEFAULT_BEANSTALKD, docopy=True):
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

    hosts : list of dictionaries or list of strings
        Each entry in the list should describe a host, containing the keys:
            "host" : either user@host or host (no default)
            "proc" : number of processes to launch (default 1)
            "fldr" : working directory (default /tmp)
            "env" : extra environment variables which need to be set
        If list of strings, is simply hostnames

    bean_port : int
        beanstalk port which to open

    docopy : boolean
        whether to remotely copy the script to the new machine
    """
    import beanstalkc as bean

    def beanstalk(bean_port=DEFAULT_BEANSTALKD):
        bean = 'beanstalkd -l {} -p {}'.format(LOCALHOST, bean_port)
        check_output(bean, shell=True)

    def copy_script(script, host):
        name = host.get('host')
        fldr = host.get('fldr', '/tmp')
        proc = host.get('proc', 1)
        env = host.get('env', {})

        tmpfile = '{}.py'.format(os.path.join(fldr, uuid.uuid4().hex))
        ssh = 'scp {script} {name}:{tmpfile}'.format(
            script=script, name=name, tmpfile=tmpfile
        )
        check_output(ssh, shell=True)
        return tmpfile

    def sshtunnel(host, bean_port=DEFAULT_BEANSTALKD):
        ssh = 'ssh {name} "ssh -N {forward} master"'.format(forward=fwd, name=name)
        check_output(ssh, shell=True)

    def launch(script, host, bean_port=DEFAULT_BEANSTALKD, index=0):
        name = host.get('host')
        fldr = host.get('fldr', '/tmp')
        proc = host.get('proc', 1)
        env = host.get('env', {})

        var = ' '.join(['{}={}:${}'.format(k, v, k) for k,v in env.items()])
        env = 'export {}; cd {};'.format(var, fldr)

        fwd = '-R{}:localhost:{}'.format(bean_port, bean_port)
        log = '> /tmp/launcher-{}.log 2>&1'.format(index)
        cmd = '{} python {} --port={} --processes={}'.format(env, script, bean_port, proc)
        ssh = 'ssh {forward} {name} "{cmd} {log}"'.format(forward=fwd, name=name, cmd=cmd, log=log)
        check_output(ssh, shell=True)

    clean_hosts = []
    for h in hosts:
        clean_hosts.append({'host': h} if isinstance(h, str) else h)
    hosts = clean_hosts

    procs = {}

    # start up the beanstalk process first
    log.info('Starting up beanstalkd ...')
    procs[-1] = Process(target=beanstalk, args=(bean_port,))
    procs[-1].start()
    time.sleep(1)

    log.info('Starting job process for ...')
    for i, host in enumerate(hosts):
        log.info('... {}'.format(host))
        o = len(hosts)
        tmpscript = script if not docopy else copy_script(script, host)
        procs[i] = Process(target=launch, args=(tmpscript, host, bean_port, i))
        procs[i].start()

    time.sleep(10)

    def signal_handler():
        log.info("Sending signal to flush, waiting 60 sec until forced shutdown...")
        for p in procs.values():
            p.join(timeout=60)
        sys.exit(1)

    log.info('Sending jobs ...')
    bsd = bean.Connection(host=LOCALHOST, port=bean_port, connect_timeout=10)
    for job in jobs:
        log.info('... {}'.format(job))
        bsd.put(json.dumps(job))

    try:
        while True:
            for i in procs.keys():
                procs[i].join(timeout=1.0)
    except (KeyboardInterrupt, SystemExit):
        signal_handler()

