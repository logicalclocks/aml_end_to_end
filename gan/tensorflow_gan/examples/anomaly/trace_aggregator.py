# import os
# import tempfile
import json
import glob

# import tensorflow as tf
# from tensorflow.contrib.layers import fully_connected as fc
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.client import timeline

class TimeLiner:
    _timeline_dict = None
    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)
    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)

many_runs_timeline = TimeLiner()
for trace_file in glob.glob("/tmp/tfgan_logdir/simple_profiler_output_dir/*.json"):
	with open(trace_file) as json_file:
	     chrome_trace = json.load(json_file)
	     chrome_trace = json.dumps(chrome_trace)
	     #chrome_trace = json.loads(chrome_trace)
	     many_runs_timeline.update_timeline(chrome_trace)


many_runs_timeline.save('/tmp/tfgan_logdir/all_steps.json')