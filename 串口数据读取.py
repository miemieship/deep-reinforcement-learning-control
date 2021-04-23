import nidaqmx
import time

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_thrmcpl_chan(physical_channel="cDAQ1Mod1/ai0:9")
    samplerate = task.timing.samp_clk_rate = 1.652
    while True:
        t1 = time.time()
        #task.ai_channels.add_ai_thrmcpl_chan(physical_channel="cDAQ1Mod1/ai0:9")
        #samplerate = task.timing.samp_clk_rate = 1.1
        T0=task.read()
        
        print(T0)
        print("The run time is :{}s.\n".format(time.time()-t1))      
        