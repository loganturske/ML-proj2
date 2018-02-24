import numpy
import pandas
from copy import deepcopy

glass_data_set = None
iris_data_set = None
spam_data_set = None
def subtract_points(a,b):
	return a - b

def square_it(a):
	return a * a
def divide_a_by_b(a,b):
	if b == 0:
		return 0
	return a/b

def get_max_of(a, b):
	if a > b:
		return a
	return b
def take_square_root(a):
	return numpy.sqrt(a)

def euclidean_dist(arr1, arr2):
	total = 0

	for x,y in zip(arr1,arr2):
		sub = subtract_points(x,y)
		sqr = square_it(sub)
		total = total + sqr
	return take_square_root(total)

def get_absolute_value(x):
	return numpy.absolute(x)

def move_id_from_glass_dataframe():
	global glass_data_set
	# glass_data_set = glass_data_set.drop(['Id'], axis=1)
	cols = list(glass_data_set.columns.values)
	cols.pop(cols.index('Id'))
	glass_data_set = glass_data_set[cols+['Id']]
	pass

def read_glass_csv():
	global glass_data_set
	glass_data_set = pandas.read_csv('glass.csv')
	pass

def read_iris_csv():
	global iris_data_set
	iris_data_set = pandas.read_csv('iris.csv')
	pass

def get_num_columns(df):
	return len(df.columns)

def get_num_rows(df):
	return df.shape[0]

def get_max_value_of_data_set(df):
	d = df.max(numeric_only=True)
	return d.max()

def read_spam_csv():
	global spam_data_set
	spam_data_set = pandas.read_csv('spam.csv')
	pass

def give_initial_random_k_means(num_of_k, num_of_dim, max_num):
	return numpy.random.randint(max_num, size=(num_of_k, num_of_dim))

def change_in_centroids(old, new):
	change = False
	if 'Class' in old.columns:
		old= old.drop(['Class'], axis=1)
	if 'Class' in new.columns:
		new = new.drop(['Class'], axis=1)
	if 'cluster' in old.columns:
		old = old.drop(['cluster'], axis=1)
	if 'cluster' in new.columns:
		new = new.drop(['cluster'], axis=1)

	for row,row2 in zip(old.iterrows(),new.iterrows()):
		print row.values
		for col in row[0].columns:
			diff = subtract_points(row[col], row2[col])
			diff = get_absolute_value(diff)
			if diff > .5:
				change = True
	return change

def find_closest_centroid(point, centroids):

	argmin = None
	chosen_centroid = None
	i = 0

	for index,cent in centroids.iterrows():
		if argmin is None:
			argmin = euclidean_dist(point.values, cent.values)
			chosen_centroid = i
		if euclidean_dist(point.values, cent) < argmin:
			argmin = euclidean_dist(point.values, cent.values)
			chosen_centroid = i
		i += 1

	# print "CHOSE: " + str(chosen_centroid)
	return chosen_centroid

def recalculate_new_centroids(df, centroids):

	if 'Class' in centroids.columns:
		centroids = centroids.drop(['Class'], axis=1)
	new_centroids = pandas.DataFrame()	
	i = 0
	for index,cent in centroids.iterrows():
		clusters = df.loc[df['cluster'] == i]
		if clusters.empty:
			new_centroids = new_centroids.append(cent, ignore_index=True)
		else:
			centroid = clusters.mean(skipna=True)
			centroid['cluster'] = i
			new_centroids = new_centroids.append(centroid, ignore_index=True)
		i += 1

	if 'Class' in new_centroids.columns:
			new_centroids = new_centroids.drop(['Class'], axis=1)
	return new_centroids

def a_sub_i(point, df):
	total = 0
	num = 0

	for index,row in df.iterrows():
		if row['cluster'] == point['cluster']:
			total += euclidean_dist(point, row)
			num += 1
	if num == 0:
		return total
	return divide_a_by_b(total,num)

def b_sub_i(point, df, centroids):
	mini = None
	for index,row in centroids.iterrows():		
		
		if row['cluster'] != point['cluster']:
			total = 0
			num = 0
			print df
			for index,row2 in df.iterrows():
				if row2['cluster'] == row['cluster']:
					print "Alkit"
					total += euclidean_dist(point, row2)
					num += 1
			print total
			b = divide_a_by_b(total,num)
			if mini == None:
				mini = b
				continue
			if b < mini:
				mini = b
	return mini

def silhouette_coefficient(point, df,centroids):
	# print "####"
	a = a_sub_i(point, df)
	# print "a: " + str(a)
	b = b_sub_i(point, df, centroids)
	# print "b: " + str(b)
	top = subtract_points(b, a)
	# print "###"
	# print top
	bottom = get_max_of(a, b)
	# print bottom
	return divide_a_by_b(top, bottom)


def overall_silhouette_coefficient(df, centroids):
	total = 0
	num = 0
	for index,row in df.iterrows():
		total += silhouette_coefficient(row, df, centroids)
		num += 1

	return divide_a_by_b(total, num)

def k_means(df,k):

	# centroids = give_initial_random_k_means(k, get_num_columns(d)-1, 2)
	df['cluster'] = 1
	centroids = df.sample(n=k)
	centroid_change = 0
	new_centroids = None
	print centroids
	while centroid_change < 5:
		for index,row in df.iterrows():
			# df = df.set_value(index, col='cluster', value=find_closest_centroid(row, centroids))
			df.iloc[index]['cluster'] = find_closest_centroid(row, centroids)

		new_centroids = recalculate_new_centroids(df, centroids)
		centroid_change += 1# change_in_centroids(centroids, new_centroids)
		centroids = new_centroids

	print centroids
	print df
	# for index,row in centroids.iterrows():
	# 	print "Cluster: " + str(row['cluster'])
	# 	print df['cluster'] == row['cluster']
	# a = overall_silhouette_coefficient(df, centroids)
	# print a
	# print centroids
	# return a
def get_num_of_k(data):
	arr = []
	for j in range(len(data):
		if data[j][-1] i
			arr.append(data[j][-1])

	return arr.size

def stepwise_forward_feature_selection(data):
	
	features = [data[j][:-1] for j in range(len(data))]
	num_of_features = features[1].size

	f_sub_0 = []
	basePref = -100000

	while features.size != 0:
		bestPerf = -100000
		bestF = features[1]
		for candidate in features:
			f_sub_0.append(candidate)
	# 		full = f_sub_0 + [response]
	# 		print full
			currPerf = k_means(data,num_of_features)
	# 		print currPerf
	# 		if currPerf > bestPerf:
	# 			bestPerf = currPerf
	# 			bestF = candidate
	# 		f_sub_0.remove(candidate)
	# 	if bestPerf > basePref:
	# 		basePref = bestPerf
	# 		features.drop([bestF])
	# 		f_sub_0.append(bestF)
	# 	else:
	# 		break
	# print bestPerf
	# print f_sub_0
read_glass_csv()
read_iris_csv()
move_id_from_glass_dataframe()

def dist1(a, b):
	mini = 100000
	i = 0
	r = 0
	for cent in b:
		total = 0
		for x,y in zip(a,cent):
			sub = subtract_points(x,y)
			sqr = square_it(sub)
			total = total + sqr
		t = take_square_root(total)
		if t < mini:
			r = i
			mini = t
		i += 1
	return r
def dist(a, b, ax=1):
	return numpy.linalg.norm(a - b, axis=ax)

def ai(point, points, C, clusters, k):
	mini = 100000000
	other_points = [points[j][:-1] for j in range(len(points)) if clusters[j] == C]
	total = 0
	for r in range(len(other_points)):
		total += euclidean_dist(point, other_points[r])
	final = divide_a_by_b(total, len(other_points))
	if final < mini:
		mini = final
	return mini
def bi(point, points, C, clusters, k):
	mini = 100000000
	for i in range(k):
		# print "START: " + str(mini)
		if i == C:
			continue
		other_points = [points[j][:-1] for j in range(len(points)) if clusters[j] == i]
		if not other_points:
			continue
		total = 0
		for r in range(len(other_points)):
			total += euclidean_dist(point, other_points[r])
			# print "Total: " + str(total)

		# print "ALGO: " + str(total) + " / " + str(len(other_points))
		final = divide_a_by_b(total, len(other_points))
		# print "FINAL: " + str(final)
		# print "MINI: " + str(mini)
		if final < mini:
			mini = final
			# print "SETTING"
	# print mini
	return mini

def sil_coe(point, points, C, clusters, k):
	b_i = bi(point, points, C, clusters, k)
	# print "B: " + str(b_i)
	a_i = ai(point, points, C, clusters, k)
	# print "A: " + str(a_i)
	top = b_i - a_i
	bottom = get_max_of(a_i, b_i)
	return divide_a_by_b(top, bottom)
def sil(points, clusters, k):
	d = len(points)
	total = 0
	for i in range(d):
		C = clusters[i]
		total += sil_coe(points[i][:-1], points, C, clusters, k)

	return divide_a_by_b(total, d)
def k_means(data, k, num_of_features):
	# C = give_initial_random_k_means(3, 5, 5)
	# # Number of clusters
	k = 3
	X = data.as_matrix()

	# # X coordinates of random centroids
	# C_x = numpy.random.randint(0, 5, size=k)
	# # Y coordinates of random centroids
	# C_y = numpy.random.randint(0, 5, size=k)
	# C = numpy.array(list(zip(C_x, C_y)), dtype=numpy.float32)
	C =  give_initial_random_k_means(3, 4, 5)
	# To store the value of centroids when it updates
	C_old = numpy.zeros(C.shape)
	clusters = numpy.zeros(len(X))
	print C
	# Error func. - Distance between new centroids and old centroids
	error = dist(C, C_old, None)
	# Loop will run till the error becomes zero
	while error != 0:
		# Assigning each value to its closest cluster
		for i in range(len(X)):
			# distances = dist1(X[i][:-1], C)
			# cluster = numpy.argmin(distances)
			clusters[i] = dist1(X[i][:-1], C)
		# Storing the old centroid values
		C_old = deepcopy(C)
		# Finding the new centroids by taking the average value
		for i in range(k):
			points = [X[j][:-1] for j in range(len(X)) if clusters[j] == i]
			if not points:
				C[i][:] =numpy.nan# numpy.random.randint(1, size=(1, 4))
			else:
				C[i] = numpy.mean(points, axis=0)
		error = dist(C, C_old, None)
	print C
	print sil(X,clusters,3)
# here()
stepwise_forward_feature_selection(iris_data_set.as_matrix())
# k_means(iris_data_set,3)
# stepwise_forward_feature_selection(iris_data_set)
# stepwise_forward_feature_selection(iris_data_set)