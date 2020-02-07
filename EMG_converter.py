import numpy as np
from os import listdir
import sys
import glob

np.set_printoptions(threshold=sys.maxsize)

def convert_to_EMG(point):
    return point * 100 / 30

def determine_eating_frames(ground_file):

    ground_reader = open(ground_file, 'r')
    ground_frames = ground_reader.readlines()
    ground_tuples = []

    # print(ground_frames)
    for i in ground_frames:

        temp = i.split(',')
        start = convert_to_EMG(int(temp[0]))
        end = convert_to_EMG(int(temp[1]))
        ground_tuples.append((round(start),round(end)))
    
    ground_reader.close()
    return ground_tuples

user_array = listdir('MyoData')
user_array.sort()

def get_np_EMG_matrix(EMG_file):
  arr = []
  x = 1
  with open(EMG_file[0], 'r') as file:
    for line in file:
      line = line.split(',')
      line = list(map(int,line))
      arr.append(np.array(line))
  return np.array(arr)

def split_EMG(eating_tuples, EMG_matrix):
  eating = []
  non_eating = np.copy(EMG_matrix)
  for start, end in eating_tuples:
    eating_slice = EMG_matrix[start-1:end,:]
    non_eating = np.delete(non_eating, slice(start-1,end), 0)
    eating.append(eating_slice)
  eating = np.vstack(eating)

  return eating, non_eating

x = 1
for user in user_array:

  print("STARTING ON USER: ", user, "\n")
  fork_truth_file_string = glob.glob('groundTruth/'+user+'/fork/*.txt')
  spoon_truth_file_string = glob.glob('groundTruth/'+user+'/spoon/*.txt')
  fork_EMG_data_file_string = glob.glob('MyoData/'+user+'/fork/*EMG.txt')
  spoon_EMG_data_file_String = glob.glob('MyoData/'+user+'/spoon/*EMG.txt')

  EMG_fork_np = get_np_EMG_matrix(fork_EMG_data_file_string)
  EMG_spoon_np = get_np_EMG_matrix(spoon_EMG_data_file_String)
  fork_eating_tuples = determine_eating_frames(fork_truth_file_string[0])
  spoon_eating_tuples = determine_eating_frames(spoon_truth_file_string[0])
  fork_eating, fork_non_eating = split_EMG(fork_eating_tuples, EMG_fork_np)
  spoon_eating, spoon_non_eating = split_EMG(spoon_eating_tuples, EMG_spoon_np)

  eating = np.vstack((fork_eating, spoon_eating))
  non_eating = np.vstack((fork_non_eating, spoon_non_eating))

  sample_number = eating.shape[0]
  non_eating = non_eating[:sample_number]

  print("SAVING FOR USER: ", user, "\n")
  np.savetxt("EMG_preprocessed/"+user+"_eating_data.txt", eating, fmt="%d")
  np.savetxt("EMG_preprocessed/"+user+"_non_eating_data.txt", non_eating, fmt="%d")
  np.save("EMG_numpy_arrays/"+user+"eating",eating)
  np.save("EMG_numpy_arrays/"+user+"non_eating", non_eating)




