import numpy as np
from os import listdir
import sys
import glob
import matplotlib.pyplot as plt
import pywt 
import math
from sklearn import decomposition 
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

#Allows you to view more in the matrix when print is called
np.set_printoptions(threshold=sys.maxsize)

#  PHASE 1 CODE

##############################################################################################

def convert_to_IMU(point):
    return point * 50 / 30

def determine_eating_frames(ground_file):

    ground_reader = open(ground_file, 'r')
    ground_frames = ground_reader.readlines()
    ground_tuples = []

    for i in ground_frames:

        temp = i.split(',')
        start = convert_to_IMU(float(temp[0]))
        end = convert_to_IMU(float(temp[1]))
        ground_tuples.append((round(start),round(end)))
    
    ground_reader.close()
    return ground_tuples

def get_np_IMU_matrix(IMU_file):
  arr = []
  x = 1
  with open(IMU_file[0], 'r') as file:
    for line in file:
      line = line.split(',')
      line = list(map(float,line))
      arr.append(np.array(line))
  return np.array(arr)

def split_IMU(eating_tuples, IMU_matrix):
  eating = []
  non_eating = np.copy(IMU_matrix)
  for start, end in eating_tuples:
    eating_slice = IMU_matrix[start-1:end,:]
    non_eating = np.delete(non_eating, slice(start-1,end), 0)
    eating.append(eating_slice)
  eating = np.vstack(eating)

  return eating, non_eating

user_array = listdir('MyoData')
user_array.sort()

for user in user_array:

  print("STARTING ON USER: ", user, "\n")
  fork_truth_file_string = glob.glob('groundTruth/'+user+'/fork/*.txt')
  spoon_truth_file_string = glob.glob('groundTruth/'+user+'/spoon/*.txt')
  fork_IMU_data_file_string = glob.glob('MyoData/'+user+'/fork/*IMU.txt')
  spoon_IMU_data_file_String = glob.glob('MyoData/'+user+'/spoon/*IMU.txt')

  IMU_fork_np = get_np_IMU_matrix(fork_IMU_data_file_string)
  IMU_spoon_np = get_np_IMU_matrix(spoon_IMU_data_file_String)
  fork_eating_tuples = determine_eating_frames(fork_truth_file_string[0])
  spoon_eating_tuples = determine_eating_frames(spoon_truth_file_string[0])
  fork_eating, fork_non_eating = split_IMU(fork_eating_tuples, IMU_fork_np)
  spoon_eating, spoon_non_eating = split_IMU(spoon_eating_tuples, IMU_spoon_np)

  eating = np.vstack((fork_eating, spoon_eating))

  fork_non_eating = fork_non_eating[:fork_eating.shape[0]]
  spoon_non_eating = spoon_non_eating[:spoon_eating.shape[0]]

  non_eating = np.vstack((fork_non_eating, spoon_non_eating))

  eating_val = np.ones((eating.shape[0],1))
  non_eating_val = np.zeros((non_eating.shape[0],1))

  eating = np.hstack((eating, eating_val))
  non_eating = np.hstack((non_eating, non_eating_val))

  print("SAVING FOR USER: ", user, "\n")
  np.savetxt("IMU_preprocessed/"+user+"_eating_data.txt", eating)
  np.savetxt("IMU_preprocessed/"+user+"_non_eating_data.txt", non_eating)
  np.save("IMU_numpy_arrays/"+user+"eating",eating)
  np.save("IMU_numpy_arrays/"+user+"non_eating", non_eating)

print('Done parsing files')

#  PHASE 2 CODE

##############################################################################################

def normalize(matrix):
  return matrix / np.linalg.norm(matrix)

def quick_plots(e,n, num, en, nn):
  
  plt.plot(e, 'go', markersize=1)
  plt.title('Eating PCA Col '+str(num+1))
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.savefig('graphs/pca/'+en+str(num))

  plt.title('Non Eating PCA Col '+str(num+1))
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.plot(n, 'ro', markersize=1)
  plt.savefig('graphs/pca/'+nn+str(num))

  plt.title('Both PCA Col '+str(num+1))
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.plot(e, 'go', markersize=1)
  plt.plot(n, 'ro', markersize=1)
  plt.savefig('graphs/pca/'+en+nn+str(num))

def plot_and_show_join(eat, non, title='No title', e_name='Eating', n_name='Non-eating'):

  plt.plot(eat, 'go', markersize=1)
  plt.title('EAT '+title)
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.savefig('graphs/eating_graphs/'+e_name+title)

  plt.plot(non, 'ro', markersize=1)
  plt.title('NON-EAT '+title)
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.savefig('graphs/non_eating_graphs/'+n_name+title)

  plt.plot(eat, 'go', markersize=1)
  plt.plot(non, 'ro', markersize=1)
  plt.xlabel('Time')
  plt.ylabel('Output')
  plt.title('COMBINED '+title)
  plt.savefig('graphs/combined_graphs/'+e_name+n_name+title)

def std_for_acc(matrix):
  acc = []
  for rows in matrix:
    row_acc = np.std(rows[1:5])
    acc.append(row_acc)
  return np.matrix(acc).transpose()

def std_for_orientation(matrix):
  ori = []
  for rows in matrix:
    ori_acc = np.std(rows[5:8])
    ori.append(ori_acc)
  return np.matrix(ori).transpose()

def std_for_gyroscope(matrix):
  gyr = []
  for rows in matrix:
    row_gyr = np.std(rows[8:11])
    gyr.append(row_gyr)
  return np.matrix(gyr).transpose()

def get_average(matrix):
  ave = []
  for rows in matrix:
    row_ave = np.average(rows[1:11])
    ave.append(row_ave)
  return np.matrix(ave).transpose()

def get_std(matrix):
  std = []
  for rows in matrix:
    row_std = np.std(rows[1:11])
    std.append(row_std)
  return np.matrix(std).transpose()

def orientation_euclidean_distance(matrix):
  dist = []
  for rows in matrix:
    val = math.sqrt(rows[1]**2 + rows[2]**2 + rows[3]**2 + rows[4]**2)
    val = val * math.log(val)
    dist.append(val)
  return np.matrix(dist).transpose()

def accelerometer_euclidean_distance(matrix):
  dist = []
  for rows in matrix:
    val = math.sqrt(rows[5]**2 + rows[6]**2 + rows[7]**2)
    val = val * math.log(val)
    dist.append(val)
  return np.matrix(dist).transpose()

def gyroscope_euclidean_distance(matrix):
  dist = []
  for rows in matrix:
    val = math.sqrt(rows[8]**2 + rows[9]**2 + rows[10]**2)
    val = val * math.log(val)
    dist.append(val)
  return np.matrix(dist).transpose()

def euclidean_distance_xs(matrix):
  dist = []
  for rows in matrix:
    val = math.sqrt(rows[1]**2 + rows[5]**2 + rows[8]**2)
    dist.append(val)
  return np.matrix(dist).transpose()

def normalize_all(eating, non_eating):
  e_col1 = normalize(eating[:,1])
  n_col1 = normalize(non_eating[:,1])

  e_col2 = normalize(eating[:,2])
  n_col2 = normalize(non_eating[:,2])

  e_col3 = normalize(eating[:,3])
  n_col3 = normalize(non_eating[:,3])

  e_col4 = normalize(eating[:,4])
  n_col4 = normalize(non_eating[:,4])

  e_col5 = normalize(eating[:,5])
  n_col5 = normalize(non_eating[:,5])

  e_col6 = normalize(eating[:,6])
  n_col6 = normalize(non_eating[:,6])

  e_col7 = normalize(eating[:,7])
  n_col7 = normalize(non_eating[:,7])

  e_col8 = normalize(eating[:,8])
  n_col8 = normalize(non_eating[:,8])

  e_col9 = normalize(eating[:,9])
  n_col9 = normalize(non_eating[:,9])

  e_col10 = normalize(eating[:,10])
  n_col10 = normalize(non_eating[:,10])

  eating[:,1] = e_col1
  eating[:,2] = e_col2
  eating[:,3] = e_col3
  eating[:,4] = e_col4
  eating[:,5] = e_col5
  eating[:,6] = e_col6
  eating[:,7] = e_col7
  eating[:,8] = e_col8
  eating[:,9] = e_col9
  eating[:,10] = e_col10

  non_eating[:,1] = n_col1
  non_eating[:,2] = n_col2
  non_eating[:,3] = n_col3
  non_eating[:,4] = n_col4
  non_eating[:,5] = n_col5
  non_eating[:,6] = n_col6
  non_eating[:,7] = n_col7
  non_eating[:,8] = n_col8
  non_eating[:,9] = n_col9
  non_eating[:,10] = n_col10

  return eating, non_eating

def dwt_values(eating, non_eating, dwt):
  e_rows = []
  n_rows = []
  _, e_1 = pywt.dwt(eating[:,1], dwt)
  _, n_1 = pywt.dwt(non_eating[:,1], dwt)
  e_rows.append(e_1)
  n_rows.append(n_1)
  _, e_2 = pywt.dwt(eating[:,2], dwt)
  _, n_2 = pywt.dwt(non_eating[:,2], dwt)
  e_rows.append(e_2)
  n_rows.append(n_2)
  _, e_3 = pywt.dwt(eating[:,3], dwt)
  _, n_3 = pywt.dwt(non_eating[:,3], dwt)
  e_rows.append(e_3)
  n_rows.append(n_3)
  _, e_4 = pywt.dwt(eating[:,4], dwt)
  _, n_4 = pywt.dwt(non_eating[:,4], dwt)
  e_rows.append(e_4)
  n_rows.append(n_4)
  _, e_5 = pywt.dwt(eating[:,5], dwt)
  _, n_5 = pywt.dwt(non_eating[:,5], dwt)
  e_rows.append(e_5)
  n_rows.append(n_5)
  _, e_6 = pywt.dwt(eating[:,6], dwt)
  _, n_6 = pywt.dwt(non_eating[:,6], dwt)
  e_rows.append(e_6)
  n_rows.append(n_6)
  _, e_7 = pywt.dwt(eating[:,7], dwt)
  _, n_7 = pywt.dwt(non_eating[:,7], dwt)
  e_rows.append(e_7)
  n_rows.append(n_7)
  _, e_8 = pywt.dwt(eating[:,8], dwt)
  _, n_8 = pywt.dwt(non_eating[:,8], dwt)
  e_rows.append(e_8)
  n_rows.append(n_8)
  _, e_9 = pywt.dwt(eating[:,9], dwt)
  _, n_9 = pywt.dwt(non_eating[:,9], dwt)
  e_rows.append(e_9)
  n_rows.append(n_9)
  _, e_10 = pywt.dwt(eating[:,10], dwt)
  _, n_10 = pywt.dwt(non_eating[:,10], dwt)
  e_rows.append(e_10)
  n_rows.append(n_10)

  return e_rows, n_rows

def fft_rows(eating, non_eating):
  e_rows = []
  n_rows = []
  
  e_1 = np.fft.fft(eating[:,1]).transpose()
  e_rows.append(e_1)
  n_1 = np.fft.fft(non_eating[:,1])
  n_rows.append(n_1)

  e_2 = np.fft.fft(eating[:,2])
  e_rows.append(e_2)
  n_2 = np.fft.fft(non_eating[:,2])
  n_rows.append(n_2)

  e_3 = np.fft.fft(eating[:,3])
  e_rows.append(e_3)
  n_3 = np.fft.fft(non_eating[:,3])
  n_rows.append(n_3)

  e_4 = np.fft.fft(eating[:,4])
  e_rows.append(e_4)
  n_4 = np.fft.fft(non_eating[:,4])
  n_rows.append(n_4)

  e_5 = np.fft.fft(eating[:,5])
  e_rows.append(e_5)
  n_5 = np.fft.fft(non_eating[:,5])
  n_rows.append(n_5)

  e_6 = np.fft.fft(eating[:,6])
  e_rows.append(e_6)
  n_6 = np.fft.fft(non_eating[:,6])
  n_rows.append(n_6)

  e_7 = np.fft.fft(eating[:,7])
  e_rows.append(e_7)
  n_7 = np.fft.fft(non_eating[:,7])
  n_rows.append(n_7)

  e_8 = np.fft.fft(eating[:,8])
  e_rows.append(e_8)
  n_8 = np.fft.fft(non_eating[:,8])
  n_rows.append(n_8)

  e_9 = np.fft.fft(eating[:,9])
  e_rows.append(e_9)
  n_9 = np.fft.fft(non_eating[:,9])
  n_rows.append(n_9)

  e_10 = np.fft.fft(eating[:,10])
  e_rows.append(e_10)
  n_10 = np.fft.fft(non_eating[:,10])
  n_rows.append(n_10)
  return e_rows, n_rows

numpy_arrays = listdir('IMU_numpy_arrays')
numpy_arrays.sort()


eating = []
non_eating = []

for i in range(int(len(numpy_arrays)/2)):
  eating = numpy_arrays[i*2]
  non_eating = numpy_arrays[i*2+1]

  e_name = eating.split('.')[0]
  n_name = non_eating.split('.')[0]

  eating = np.load('IMU_numpy_arrays/user23eating.npy')
  non_eating = np.load('IMU_numpy_arrays/user23non_eating.npy')

  eating, non_eating = normalize_all(eating, non_eating)

  e_std_acc = std_for_acc(eating)
  n_std_acc = std_for_acc(non_eating)

  e_std_ori = std_for_orientation(eating)
  n_std_ori = std_for_orientation(non_eating)

  e_std_gyr = std_for_gyroscope(eating)
  n_std_gyr = std_for_gyroscope(non_eating)

  e_ave = get_average(eating)
  n_ave = get_average(non_eating)

  e_std = get_std(eating)
  n_std = get_std(non_eating)

  e_ori_dist = orientation_euclidean_distance(eating)
  n_ori_dist = orientation_euclidean_distance(non_eating)

  e_acc_dist = accelerometer_euclidean_distance(eating)
  n_acc_dist = accelerometer_euclidean_distance(non_eating)

  e_gyro_dist = gyroscope_euclidean_distance(eating)
  n_gyro_dist = gyroscope_euclidean_distance(non_eating)

  e_row_fft, n_row_fft = fft_rows(eating, non_eating)

  e_coif, n_coif = dwt_values(eating, non_eating, 'coif17')

  e_dmey, n_dmey = dwt_values(eating, non_eating, 'dmey')

  e_db, n_db = dwt_values(eating, non_eating, 'db36')

  plot_and_show_join(e_std, n_std, 'SD EACH ROW',e_name, n_name)
  plot_and_show_join(e_std_acc, n_std_acc, 'SD ACCELEROMETER',e_name, n_name)
  plot_and_show_join(e_std_ori, n_std_ori, 'SD ORIENTATION',e_name, n_name)
  plot_and_show_join(e_std_gyr, n_std_gyr, 'SD GYROSCOPE',e_name, n_name)
  for i in range(10):
    plot_and_show_join(e_row_fft[i], n_row_fft[i], 'FFT COL '+str(i + 1),e_name, n_name)
  plot_and_show_join(e_ori_dist, n_ori_dist, 'ED ORIENTATION',e_name, n_name)
  plot_and_show_join(e_acc_dist, n_acc_dist, 'ED ACCELEROMETER',e_name, n_name)
  plot_and_show_join(e_gyro_dist, n_gyro_dist, 'ED GYROSCOPE',e_name, n_name)
  plot_and_show_join(e_ave, n_ave, 'AVERAGE EACH ROW',e_name, n_name)
  for i in range(10):
    plot_and_show_join(e_coif[i], n_coif[i], 'COIF COL '+str(i + 1),e_name, n_name)
  for i in range(10):
    plot_and_show_join(e_dmey[i], n_dmey[i], 'DMEY COL '+str(i + 1),e_name, n_name)
  for i in range(10):
    plot_and_show_join(e_db[i], n_db[i], 'DB COL '+str(i + 1),e_name, n_name)

#  PHASE 3 CODE

##############################################################################################

  smallest_shape = e_dmey[0].shape[0]

  e_std = e_std[:smallest_shape]
  n_std = n_std[:smallest_shape]
  e_std_acc = e_std_acc[:smallest_shape]
  n_std_acc = n_std_acc[:smallest_shape]
  e_std_ori = e_std_ori[:smallest_shape]
  n_std_ori = n_std_ori[:smallest_shape]
  e_std_gyr = e_std_gyr[:smallest_shape]
  n_std_gyr = n_std_gyr[:smallest_shape]
  for i in range(10):
    e_row_fft[i] = e_row_fft[i][:smallest_shape]
    n_row_fft[i] = n_row_fft[i][:smallest_shape]
    e_coif[i] = e_coif[i][:smallest_shape]
    n_coif[i] = n_coif[i][:smallest_shape]
    e_db[i] = e_db[i][:smallest_shape]
    n_db[i] = n_db[i][:smallest_shape]
    e_dmey[i] = e_dmey[i][:smallest_shape]
    n_dmey[i] = n_dmey[i][:smallest_shape]
  e_ori_dist = e_ori_dist[:smallest_shape]
  n_ori_dist = n_ori_dist[:smallest_shape]
  e_acc_dist = e_acc_dist[:smallest_shape]
  n_acc_dist = n_acc_dist[:smallest_shape]
  e_gyro_dist = e_gyro_dist[:smallest_shape]
  n_gyro_dist = n_gyro_dist[:smallest_shape]
  e_ave = e_ave[:smallest_shape]
  n_ave = n_ave[:smallest_shape]

  eating_features = np.hstack(( e_std, e_std_acc, e_std_ori, e_std_gyr, e_ori_dist, e_acc_dist, e_gyro_dist, e_ave))
  non_eating_features = np.hstack(( n_std, n_std_acc, n_std_ori, n_std_gyr, n_ori_dist, n_acc_dist, n_gyro_dist, n_ave))

  feature_columns = ['SD', 'SD ACC', 'SD ORI', 'SD GYR', 'ED ORI', 'ED ACC', 'ED GYRO', 'AVE']

  for i in range(10):
    e_row_fft[i] = np.real(e_row_fft[i].reshape(-1,1))
    e_coif[i] = np.real(e_coif[i].reshape(-1,1))
    e_dmey[i] = np.real(e_dmey[i].reshape(-1,1))
    e_db[i] = np.real(e_db[i].reshape(-1,1))
    n_row_fft[i] = np.real(n_row_fft[i].reshape(-1,1))
    n_coif[i] = np.real(n_coif[i].reshape(-1,1))
    n_dmey[i] = np.real(n_dmey[i].reshape(-1,1))
    n_db[i] = np.real(n_db[i].reshape(-1,1))
    non_eating_features = np.hstack((non_eating_features, n_row_fft[i][:smallest_shape], n_coif[i][:smallest_shape], n_dmey[i][:smallest_shape], n_db[i][:smallest_shape]))
    eating_features = np.hstack((eating_features, e_row_fft[i][:smallest_shape], e_coif[i][:smallest_shape], e_dmey[i][:smallest_shape], e_db[i][:smallest_shape]))
    feature_columns.append('FFT COL '+str(i))
    feature_columns.append('COIF COL '+str(i))
    feature_columns.append('DMEY COL '+str(i))
    feature_columns.append('DB COL '+str(i))
    
  feature_matrix = np.vstack((eating_features, non_eating_features))

  features_std = StandardScaler().fit_transform(feature_matrix)
  cov = np.cov(features_std.T)
  ev , eig = np.linalg.eig(cov) 

  e_pca_result = eating_features * eig[:,:5]
  n_pca_result = non_eating_features * eig[:,:5]

  fig = go.Figure()
  for i in range(eig.shape[0]):
    fig.add_trace(go.Scatterpolar(
      r=eig[:,i],
      theta=feature_columns,
      name='Col '+str(i)
    ))
  fig.write_image('graphs/spider/'+e_name+n_name+'.png')

  quick_plots(e_pca_result[:,0], n_pca_result[:,0], 0, e_name, n_name)
  quick_plots(e_pca_result[:,1], n_pca_result[:,1], 1, e_name, n_name)
  quick_plots(e_pca_result[:,2], n_pca_result[:,2], 2, e_name, n_name)
  quick_plots(e_pca_result[:,3], n_pca_result[:,3], 3, e_name, n_name)
  quick_plots(e_pca_result[:,4], n_pca_result[:,4], 4, e_name, n_name)

  print('Done with: ',e_name)


# Helpers to write report

# var_explained = np.round(ev/np.sum(ev), decimals=3)
 
# sns.barplot(x=list(range(1,len(var_explained)+1)),
#             y=var_explained, color="limegreen")
# plt.xlabel('Principle Components', fontsize=16)
# plt.ylabel('Percent Variance Explained', fontsize=16)
# plt.show()

# pca = decomposition.PCA(n_components=5)
# pcs = pca.fit_transform(features_std)
# print(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_))
