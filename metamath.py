from pexpect import popen_spawn
import signal


class Metamath:
    def __init__(self, metamath_path):
        self.path = metamath_path
        self.process_dict = {}

    def initialize(self,id):
        self.process_dict[id] = popen_spawn.PopenSpawn(self.path)
        self.process_dict[id].expect('MM>')
        return self.process_dict[id].before.decode()

    def send(self,id,command):
        self.process_dict[id].sendline(command)
        self.process_dict[id].expect('MM(?:(?!MM).)*?>')
        return self.process_dict[id].before.decode()

    def close(self,id):
        self.process_dict[id].kill(signal.CTRL_C_EVENT)
        del self.process_dict[id]
# how to use
'''
metamath = Metamath('metamath/metamath.exe')    #path from this file to metamath.exe
print(metamath.initialize(0))
output1 = metamath.send(0,'read "metamath/demo0.mm"')   
print(output1)
def output_process(output):         # an example of creating your next input from the output
    return 'verify proof *'
output2 = metamath.send(0,output_process(output1))
print(output2)
metamath.close(0)
'''

