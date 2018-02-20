import numpy
import pandas

glass_data_set = None
iris_data_set = None
spam_data_set = None
def subtract_points(a,b):
	return a - b

def square_it(a):
	return a * a

def take_square_root(a):
	return numpy.sqrt(a)

def euclidean_dist(arr1, arr2):
	total = 0
	
	for i in xrange(0,subtract_points(len(arr2),1)):
		sub = subtract_points(arr1[i], arr2[i])
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

def change_in_centroid(old, new):
	change = False
	for i in xrange(0,subtract_points(len(old),1)):
		diff = subtract_points(old[i], new[i])
		diff = get_absolute_value(diff)
		if diff > .1:
			change = True
	return change

def change_in_any_centroids(old_centroids, new_centroids):
	change = False
	for index in xrange(0, subtract_points(len(old_centroids),1)):
		if change_in_centroid(old_centroids[index], new_centroids[index]) is True:
			return True
	return False
def find_closest_centroid(point, centroids):

	argmin = None
	chosen_centroid = None
	index = 0
	for cent in centroids:
		if argmin is None:
			argmin = euclidean_dist(point.values, cent)
			chosen_centroid = index
		if argmin > euclidean_dist(point.values, cent):
			argmin = euclidean_dist(point.values, cent)
			chosen_centroid = index
		index += 1

	return chosen_centroid

def recalculate_new_centroid(d, old_cent_num, default_cent):
	cluster = d.loc[d['cluster'] == old_cent_num]
	new_centroid = cluster.mean(skipna=True)
	for i in new_centroid.values:
		if numpy.isnan(i):
			return default_cent
	return new_centroid.values[:-1]


def k_means(d,k):
	centroids = give_initial_random_k_means(k, get_num_columns(d)-1, get_max_value_of_data_set(d))

	print centroids
	d = d.assign(cluster = 0)
	change_in_centroids = True
	while change_in_centroids:

		for index,row in d.iterrows():
			d.at[index,'cluster'] = find_closest_centroid(row, centroids)
		new_centroids = centroids
		for index in xrange(0, len(centroids)-1):
			new_centroids[index] = recalculate_new_centroid(d, index, centroids[index])
		change_in_centroids = change_in_any_centroids(centroids, new_centroids)
		centroids = new_centroids
	print
	print centroids
read_glass_csv()
read_iris_csv()
move_id_from_glass_dataframe()

k_means(iris_data_set, 3)
