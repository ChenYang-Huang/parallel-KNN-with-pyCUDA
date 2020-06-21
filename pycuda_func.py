import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import datetime

mod = SourceModule("""

  #include <stdint.h>

  __global__ void get_neighbor_dist(uint8_t* train, uint8_t* valid, uint32_t* dest, int offset, int train_num)
  {

    int bid = blockIdx.x;
    int x = threadIdx.x;

    uint8_t *cur_item = valid + offset * 784;
    __shared__ float dist_arr [832];

    if(x < 392){
      int difference = *(cur_item + x) - *(train + bid * 784 + x);
      dist_arr[x] = difference * difference;
      difference = *(cur_item + 392 + x) - *(train + bid * 784 + 392 + x);
      dist_arr[x + 416] = difference * difference;
    }
    else{
      dist_arr[x] = 0;
      dist_arr[x + 416] = 0;
    }
    
    for (int stride = 416; stride > 0; stride /= 2){
      __syncthreads();
      if(x < stride){
        dist_arr[x] += dist_arr[x + stride];
      }
    }

    if (x == 0){
      dest[offset * train_num + bid] = dist_arr[0];
    }

    return;
  }
  """)


def GPU_init_memory(train_data, val_data, num_val, num_train):

    train_gpu = cuda.mem_alloc(train_data.nbytes)
    cuda.memcpy_htod(train_gpu, train_data)

    val_gpu = cuda.mem_alloc(val_data.nbytes)
    cuda.memcpy_htod(val_gpu, val_data)

    dest_gpu = cuda.mem_alloc(num_val * num_train * 4)  # uint32

    return train_gpu, val_gpu, dest_gpu


def parallel_neighbor_search(train_data, val_data, num_val, num_train, K):

    digits = np.array([item[0] for item in train_data])
    train_data = np.array([item[1:] for item in train_data])

    train_gpu, val_gpu, dest_gpu = GPU_init_memory(
        train_data, val_data, num_val, num_train)

    func = mod.get_function("get_neighbor_dist")

    counter, big_counter = 0, 0

    for i in range(num_val):
        counter += 1
        func(train_gpu, val_gpu, dest_gpu, np.int32(i), np.int32(num_train),
             block=(416, 1, 1), grid=(num_train, 1, 1))

        if (counter == num_val / 100):
            big_counter += 1
            counter = 0
            print("%" +
                  str(big_counter) + ", " + str(datetime.datetime.now()), end="\r", flush=True)

    results = np.empty([num_val, num_train], dtype=np.uint32)
    cuda.memcpy_dtoh(results, dest_gpu)

    prediction = []
    for item in results:
        index_arr = np.argpartition(item, K)[:K]
        prediction.append(most_frequent(
            [digits[index] for index in index_arr]))

    return prediction


def test():
    print("nihao")


def most_frequent(List: list) -> int:
    return max(set(List), key=List.count)
