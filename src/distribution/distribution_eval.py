from multiprocessing import Manager, Process, Queue, Lock

import logzero
import requests
import urllib3

from src.eval_func import eval_funcs
from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)


def eval_worker(up_queue, result_queue, gpu_index: int = None):
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)
    while True:
        task_info = up_queue.get()
        task_id = task_info["task_id"]
        task_func = task_info["task_func"]
        task_args = task_info["task_args"]
        if task_func in eval_funcs:
            try:
                result = eval_funcs[task_func](*task_args)
                result_queue.put({
                    "task_id": task_id,
                    "result": result
                })
            except Exception as e:
                print(e)
                pass


class DistributedEvaluator:
    def __init__(self, master_host="10.16.104.19:1088", task_capacity=512, task_type="cpu", gpu_list: List[int] = None,
                 logger=None, up_queue: Queue = None, result_queue: Queue = None):
        if not os.path.exists(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs")):
            os.makedirs(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs"))
        self.logger = logger if logger is not None else logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/evaluator.log")),
            name="DistributedEvaluator Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7))
        self.master_host = master_host
        self.task_capacity = task_capacity
        self.task_lock = Lock()
        self.task_type = task_type
        self.gpu_list = gpu_list
        self.manager = Manager()
        self.up_queue = up_queue if up_queue is not None else Queue()
        self.result_queue = result_queue if result_queue is not None else Queue()
        self.eval_processes = []
        self.task_ongoing: Dict = self.manager.dict()
        for p_num in range(self.task_capacity):
            if self.task_type == "cpu":
                p = Process(target=eval_worker, args=(self.up_queue, self.result_queue))
            elif self.task_type == "gpu":
                if self.gpu_list is None or len(self.gpu_list) == 0:
                    gpu_index = None
                else:
                    gpu_index = self.gpu_list[p_num % len(self.gpu_list)]
                print(f"Create Task on GPU {gpu_index}")
                p = Process(target=eval_worker, args=(self.up_queue, self.result_queue, gpu_index))
            else:
                print("ERROR TASK TYPE")
                break
            p.start()
            self.eval_processes.append(p)
        self.task_process = Process(target=self.eval_new_task, args=())
        self.result_process = Process(target=self.report_task_result, args=())
        self.task_process.start()
        self.result_process.start()
        print("EVAL START")

    def eval_new_task(self):
        fail_num = 0
        while True:
            try:
                response_code = 500
                while response_code > 299 or response_code < 200:
                    try:
                        fail_num += 1
                        if fail_num >= 10:
                            time.sleep(2)
                        elif fail_num > 100:
                            time.sleep(10)
                            fail_num = 10
                        with self.task_lock:
                            current_tasks = dict(self.task_ongoing)
                            rest_capacity = self.task_capacity - sum([value for value in current_tasks.values()])
                        if rest_capacity <= 0:
                            time.sleep(1)
                            continue
                        response = requests.post(url="http://{}/get_task".format(self.master_host),
                                                 data={
                                                     "check_val": "81600a92e8416bba7d9fada48e9402a4",
                                                     "task_type": self.task_type,
                                                     "max_cost": rest_capacity,
                                                 }, verify=False)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        pass
                task_info: Dict = pickle.loads(response.content)
                # self.logger.info(f"Get New Task {task_info['task_id']} with cost {task_info['task_cost']}")
                self.up_queue.put(task_info)
                with self.task_lock:
                    self.task_ongoing[task_info["task_id"]] = task_info["task_cost"]
                fail_num = 0
            except Exception as e:
                self.logger.error(e.with_traceback())

    def report_task_result(self):
        while True:
            try:
                result_data = self.result_queue.get()
                response_code = 500
                while response_code > 299 or response_code < 200:
                    try:
                        response = requests.post(url="http://{}/report_result".format(self.master_host),
                                                 data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                                       "task_id": result_data["task_id"]},
                                                 files={"result": pickle.dumps(result_data["result"])}, verify=False)
                        response_code = response.status_code
                    except requests.exceptions.ConnectionError as e:
                        pass
                # self.logger.info(f"Report Task Result of {result_data['task_id']}")
                with self.task_lock:
                    self.task_ongoing.pop(result_data["task_id"])
            except Exception as e:
                self.logger.error(e.with_traceback())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--master_host', type=str, default="10.16.104.19:1088", help='The Host of the Master')
    parser.add_argument('--task_capacity', type=int, default=200, help='The Host of the Master')
    parser.add_argument('--task_type', type=str, default="cpu", help='The Task Type of Evaluator')
    parser.add_argument('--gpu_list', nargs='+', type=int, default=[], help="The GPU List used by Evaluator")
    args = parser.parse_args()
    print(args.task_type, args.gpu_list)
    if args.task_capacity <= 160:
        evaluator = DistributedEvaluator(master_host=args.master_host, task_capacity=args.task_capacity,
                                         task_type=args.task_type, gpu_list=args.gpu_list)
        while True:
            time.sleep(100)
    else:
        sub_evaluator_num = (args.task_capacity // 160) + 1
        sub_cap = (args.task_capacity // sub_evaluator_num) + 1
        print(f"Run with SUB EVALUATOR: {sub_evaluator_num}x{sub_cap}")
        logger = logzero.setup_logger(
            logfile=str(Path(os.path.dirname(os.path.abspath(__file__)), "../../logs/Distribution_logs/evaluator.log")),
            name="DistributedEvaluator Log", level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
            maxBytes=int(1e7)
        )


        def new_sub_evaluator(logger):
            sub_evaluator = DistributedEvaluator(master_host=args.master_host, task_capacity=sub_cap,
                                                 task_type=args.task_type, gpu_list=args.gpu_list, logger=logger)
            while True:
                time.sleep(100)


        p_list = []
        for _ in range(sub_evaluator_num):
            p = Process(target=new_sub_evaluator, args=(logger,))
            p_list.append(p)
            p.start()
        [p.join() for p in p_list]
