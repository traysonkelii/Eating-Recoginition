import numpy as np 
import matplotlib.pyplot as plt
import pywt 
import math

def normalize(matrix):
  return matrix / np.linalg.norm(matrix)

def plot_and_show_split(eat, non):
  plt.subplot(1,2,1)
  plt.plot(eat, 'go', markersize=1)
  plt.title('Eat')
  plt.subplot(1,2,2)
  plt.plot(non, 'ro', markersize=1)
  plt.title('Non-Eat')
  plt.show()

def plot_and_show_join(eat, non, title='Meh'):
  plt.plot(eat, 'go', markersize=1)
  plt.plot(non, 'ro', markersize=1)
  plt.title(title)
  plt.show()

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

def dmey_wave(matrix):
  wave = []
  for rows in matrix:
    _, row_wave = pywt.dwt(rows[1:11], 'dmey')
    wave.append(row_wave)
  return np.matrix(wave)

def coif_wave(matrix):
  wave = []
  for rows in matrix:
    _, row_wave = pywt.dwt(rows[1:11], 'coif17')
    wave.append(row_wave)
  return np.matrix(wave)

def db_wave(matrix):
  wave = []
  for rows in matrix:
    _, row_wave = pywt.dwt(rows[1:11], 'db36')
    wave.append(row_wave)
  return np.matrix(wave)

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

def fft_row(matrix):
  fft = []
  for rows in matrix:
    val = np.fft.fft(rows[1:11])
    fft.append(val)
  return np.matrix(fft)

def get_db(matrix):
  db = []
  for rows in matrix: 
    _, row_db = pywt.dwt(rows[1:11], 'db36')
    db.append(row_db)
  return np.matrix(db).transpose()

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

eating = np.load('IMU_numpy_arrays/user12eating.npy')
non_eating = np.load('IMU_numpy_arrays/user12non_eating.npy')

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

e_fft = fft_row(eating)
n_fft = fft_row(non_eating)

e_row_fft, n_row_fft = fft_rows(eating, non_eating)

e_coif, n_coif = dwt_values(eating, non_eating, 'coif17')

e_dmey, n_dmey = dwt_values(eating, non_eating, 'dmey')

e_db, n_db = dwt_values(eating, non_eating, 'db36')

e_row_dmey = dmey_wave(eating)
n_row_dmey = dmey_wave(non_eating)

e_row_coif = coif_wave(eating)
n_row_coif = coif_wave(non_eating)

e_row_db = db_wave(eating)
n_row_db = db_wave(non_eating)

print('STD EACH ROW: ', e_std.shape)
print('STD ACCELEROMETER: ', e_std_acc.shape)
print('STD ORIENTATION: ', e_std_ori.shape)
print('STD GYROSCOPE: ',e_std_gyr.shape)
print('FFT EACH ROW: ', e_fft.shape)
print('FFT COL SHAPE: ', e_row_fft[0].shape)
print('DIST ORIENTATION: ', e_ori_dist.shape)
print('DIST ACCELEROMETER: ', e_acc_dist.shape)
print('DIST GYROSCOPE: ', e_gyro_dist.shape)
print('AVERAGE OF EACH ROW: ', e_ave.shape)
print('COIF SHAPE: ', e_coif[0].shape)
print('DMEY SHAPE: ', e_dmey[0].shape)
print('DB SHAPE: ', e_db[0].shape)
print('DMEY Each Row: ', e_row_dmey.shape)
print('COIF Each Row: ', e_row_coif.shape)
print('DB Each Row: ', e_row_db.shape)
print(pywt.wavelist(kind='discrete'))

plot_and_show_join(e_std, n_std, 'Standard Deviation')
plot_and_show_join(e_std_acc, n_std_acc, 'STD ACCELEROMETER')
plot_and_show_join(e_std_ori, n_std_ori, 'STD ORIENTATION')
plot_and_show_join(e_std_gyr, n_std_gyr, 'STD GYROSCOPE')
plot_and_show_join(e_fft, n_fft, 'FFT')
for i in range(10):
  plot_and_show_join(e_row_fft[i], n_row_fft[i], 'FFT Column '+str(i))
plot_and_show_join(e_ori_dist, n_ori_dist, 'Euclidean Orientation')
plot_and_show_join(e_acc_dist, n_acc_dist, 'Euclidean Acceleromter')
plot_and_show_join(e_gyro_dist, n_gyro_dist, 'Euclidean Gyroscope')
plot_and_show_join(e_ave, n_ave, 'Average')
for i in range(10):
  plot_and_show_join(e_coif[i], n_coif[i], 'COIF COL '+str(i))
for i in range(10):
  plot_and_show_join(e_dmey[i], n_dmey[i], 'DMEY COL '+str(i))
print(len(e_db))
for i in range(10):
  plot_and_show_join(e_db[i], n_db[i], 'DB COL '+str(i))
plot_and_show_join(e_row_dmey, n_row_dmey, 'DMEY Each Row')
plot_and_show_join(e_row_coif, n_row_coif, 'COIF Each Row')
plot_and_show_join(e_row_db, n_row_db, 'DB Each Row')



# print(e_db_coe1)


# wavelet coefficient: Abrupt transitions in signals result in wavelet coefficients with large absolute values.

# keep dmey






