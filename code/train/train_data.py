import os
from multiprocessing import Pool
import sys
import glob
import time



def run_data(data_prefix, data_prefix1,result_path ,data_info, data_info1, GPU_SET, kernel_number, local_window_size,random_seed):


    cmd = "python ../model/train_model.py"
    data_path = data_prefix +"\\" + data_info + "\\"
    data_path1 = data_prefix1 + "\\" + data_info1 + "\\"
    set = cmd + " " + data_path + " " + data_path1 + " " + result_path + " " + data_info + " "+str(kernel_number) + " " + str(random_seed) +" " + str(local_window_size)+ " " +GPU_SET
    print(set)
    os.system(set)

def get_data_info(path):
    #path_list = glob.glob(path + '*/')
    path_list = glob.glob(path)
    data_list = []
    for rec in path_list:
        data_info = rec.split("\\")[-1]
        data_list.extend([data_info])
    return data_list

if __name__ == '__main__':


    GPU_SET = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    '''
    GPU_SET = 0
    start = 0
    end = 2
    '''
    # the path of data
    path = r"F:\Download\lengent\venv\\MAHyNet-main\demo\HDF5\*"
    data_prefix = r"F:\Download\lengent\venv\\MAHyNet-main\demo\HDF5"
    result_path = r"F:\Download\lengent\venv\\MAHyNet-main\demo\result_local"
    path1 = r"F:\Download\lengent\venv\\MAHyNet-main\demo\HDF5_2li\*"
    data_prefix1 = r"F:\Download\lengent\venv\MAHyNet-main\demo\HDF5_2li"

    data_list = get_data_info(path)
    data_list1 =get_data_info(path1)
    start_time =  time.time()
    pool = Pool(processes  = 1 )
    local_window_size_list = [19]#19
    #local_window_size_list = [2,5,9,15,19,25,51]
    random_seed_list = [1]
    #kernal_number_list = [8,16,32,64,128,256]
    kernal_number_list = [256]

    print("start run all models")
    for kernel_number in kernal_number_list:
        for random_seed in random_seed_list:
            for local_window_size in local_window_size_list:
                for data_info in data_list[start:end]:
                   for data_info1 in data_list1[start:end]:
                       if  data_info==data_info1:

                            print(data_info, GPU_SET, local_window_size, random_seed)
                            print(data_info1, GPU_SET, local_window_size, random_seed)
                            time.sleep(2)

                            pool.apply_async(run_data, (data_prefix, data_prefix1,result_path ,data_info, data_info1, GPU_SET, kernel_number, local_window_size, random_seed))
    pool.close()
    pool.join()
    print("all model cost ",  time.time() - start_time)



