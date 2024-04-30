import apriltag
import cv2 
import numpy as np
from tqdm import tqdm
import glob
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
def plot_positions(img, folder, color = (255,0,0)):
	detector = apriltag.Detector()
	for fn in tqdm(glob.glob(folder + '*.png', recursive=True)):
		curr_img = cv2.imread(fn)
		gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)
		for pt in results:
			img = cv2.circle(img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=color, thickness=-1)
	return img

folder1 = '/home/grablab/VisionForceEstimator/real_world_data/1026_test_data/40_1/images/'
folder2 = '/home/grablab/VisionForceEstimator/real_world_data/1025_data/wall_contact_40_1/images/'

example_img = cv2.imread(folder1 + '0.png')
example_img = plot_positions(example_img, folder1, (255,0,0))
example_img = plot_positions(example_img, folder2, (0,0,255))
# example_img = plot_positions(example_img, folder3, (0,0,255))
cv2.imshow('',example_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()

def visualize():
	detector = apriltag.Detector()
	example_img = cv2.imread(f'/home/yifan/Downloads/1006_data/test_jello/0.png')

	for i in tqdm(range(1000)):
		fn = f'/home/yifan/Downloads/1006_data/test_jello/{i}.png'
		img = cv2.imread(fn)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)
		for pt in results:
			example_img = cv2.circle(example_img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=(255, 0, 0), thickness=-1)

	for i in tqdm(range(5000)):
		fn = f'/home/yifan/Downloads/1006_data/train_50/{i}.png'
		img = cv2.imread(fn)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)
		for pt in results:
			example_img = cv2.circle(example_img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=(0, 0,255), thickness=-1)


	for i in tqdm(range(5000)):
		fn = f'/home/yifan/Downloads/1006_data/train_25/{i}.png'
		img = cv2.imread(fn)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)
		for pt in results:
			example_img = cv2.circle(example_img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=(0, 0,255), thickness=-1)


	for i in tqdm(range(5000)):
		fn = f'/home/yifan/Downloads/1006_data/train_40/{i}.png'
		img = cv2.imread(fn)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		results = detector.detect(gray)
		for pt in results:
			example_img = cv2.circle(example_img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=(0, 0,255), thickness=-1)


		for i in tqdm(range(5000)):
			fn = f'/home/yifan/Downloads/1006_data/train_20/{i}.png'
			img = cv2.imread(fn)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			results = detector.detect(gray)
			for pt in results:
				example_img = cv2.circle(example_img, (int(pt.center[0]), int(pt.center[1])), radius=0, color=(0, 0,255), thickness=-1)


	cv2.imshow('',example_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_tag_positions(folder):
	print(folder)
	detector = apriltag.Detector()
	tag_positions = []
	fns = glob.glob(os.path.join(folder, '*.png'))
	for fn in tqdm(fns): 
		img = cv2.imread(fn)
		try:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			results = detector.detect(gray)
			positions = np.ones(8)*-1
			for pt in results:
				tag_id = int(pt.tag_id) - 1
				positions[tag_id*2] = pt.center[0]
				positions[tag_id*2+1] = pt.center[1]
		
			if sum(positions < 0) > 0:
				continue

			tag_positions.append(list(positions))
		except:
			pass
	return tag_positions

def run_tsne(X, Y):
	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(X)
	df = pd.DataFrame()
	df["y"] = Y
	df["comp-1"] = z[:,0]
	df["comp-2"] = z[:,1]

	sns.scatterplot(x="comp-1", y="comp-2", hue=Y.tolist(),
					palette=sns.color_palette("tab10"), alpha=0.3,
					data=df).set(title="Finger position data T-SNE projection")
	plt.show()


def load_tag_positions(folders, starting_label = 0):
	X = None
	for folder in folders:
		name = folder.split('/')[-1]
		print(os.path.join(home_folder, name ))
		tag_positions = np.load(os.path.join(home_folder, name ))
		if X is None:
			X = tag_positions
			Y = np.ones(X.shape[0])*starting_label
		else:
			try:
				X = np.concatenate((X, tag_positions), axis = 0)
				Y = np.concatenate((Y, np.ones(tag_positions.shape[0])*starting_label))
			except:
				print('This dataset seems to be weird:', name)
		starting_label += 1
		print(X.shape, Y.shape)
	return X, Y, starting_label

if __name__=='__main__':
	import glob
	home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1024_data/' 
	#folders = glob.glob(home_folder + '*')
	folders = [ f.path for f in os.scandir(home_folder) if f.is_dir() ]
	print(folders)
	for folder in folders:
		name = folder.split('/')[-1]
		if not os.path.exists(os.path.join(home_folder, name) + '.npy'):
			print(name)
			tag_positions = get_tag_positions(os.path.join(home_folder , name) + '/images/')			
			np.save(os.path.join(home_folder, name), np.array(tag_positions))
		# tag_positions = np.load(os.path.join(home_folder, name + '.npy'))
	# exit()

	# home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1019_assembled_data/' 
	# folders = glob.glob(home_folder + '*.npy')
	# X, Y, starting_label = load_tag_positions(folders)
	# # home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1005_data/' 
	# # folders = glob.glob(home_folder + '*.npy')
	# # X2, Y2 = load_tag_positions(folders)
	# home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1020_testing_data/' 
	# names = [home_folder + 'test_4.npy']
	# X3, Y3,_ = load_tag_positions(names, starting_label)
	# X = np.concatenate((X,X3))
	# Y = np.concatenate((Y,Y3))
	# run_tsne(X, Y)

	# X = np.concatenate((X,X2,X3))
	# Y = np.concatenate((Y,Y2,Y3))
	# print(X.shape, Y.shape)

	home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1019_assembled_data/' 
	folders = glob.glob(home_folder + '*.npy')
	X, Y, starting_label = load_tag_positions(folders)
	#counter = 0
	Y = np.zeros(X.shape[0])

	# for i in range(len(Y)):
	# 	if (i+1)%1000 == 0:
	# 		counter += 1
	# 	Y[i] = counter


	home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1023_data/' 
	folders = glob.glob(home_folder + '*.npy')
	X2, Y2, starting_label = load_tag_positions(folders, 1)
	Y2 = np.ones(X2.shape[0])*1



	home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1024_data/' 
	folders = glob.glob(home_folder + '*.npy')
	X3, Y3, starting_label = load_tag_positions(folders, 1)
	Y3 = np.ones(X3.shape[0])*2

	home_folder = '/home/grablab/VisionForceEstimator/real_world_data/1020_testing_data/' 
	names = [home_folder + 'test_4.npy']
	X4, Y4, _ = load_tag_positions(names, starting_label= 3)
	print(Y.shape, Y2.shape, Y3.shape, Y4.shape)
	X = np.concatenate((X,X2,X3,X4))
	Y = np.concatenate((Y,Y2,Y3,Y4))

	run_tsne(X, Y)
