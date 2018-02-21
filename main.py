import numpy
import pandas

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
		if diff > .05:
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

def a_sub_i(point, df):
	total = 0
	num = 0

	for row in df.iterrows():
		if row[1]['cluster'] == point[1]['cluster']:
			total += euclidean_dist(point[1].values[:-1], row[1].values[:-1])
			num += 1

	if num == 0:
		return total
	return divide_a_by_b(total,num)

def b_sub_i(point, df):
	mini = None
	clusters = df.drop_duplicates(subset='cluster')
	print "CLUSTER"
	print clusters
	for c in clusters.iterrows():
		x = c[1]['cluster']
		total = 0
		num = 0
		for row in df.iterrows():
			if row[1]['cluster'] == x:
				total += euclidean_dist(point[1].values[:-1], row[1].values[:-1])
				num += 1
		b = divide_a_by_b(total,num)
		if mini == None:
			mini = b
			continue
		if b < mini:
			mini = b
	return mini

def silhouette_coefficient(point, df):
	a = a_sub_i(point, df)
	b = b_sub_i(point, df)
	top = subtract_points(b, a)
	bottom = get_max_of(a, b)
	return divide_a_by_b(top, bottom)

def overall_silhouette_coefficient(df):
	total = 0
	num = 0
	for row in df.iterrows():
		total += silhouette_coefficient(row, df)
		num += 1
	return divide_a_by_b(total, num)

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

	print centroids
	a = overall_silhouette_coefficient(d)
	print a
	return a

def stepwise_forward_feature_selection(data):
	f_sub_0 = []
	basePref = -100000
	features = data.columns[:-1]
	response = data.columns[-1]

	selected = []
	current_score, best_new_score = 0.0, 0.0

	while features.empty != True and (current_score == best_new_score):
		scores_with_candidates = []
		for candidate in features:
			selected.append(candidate)
			full = selected + [response]
			score = k_means(data[full],3)
			print score
			scores_with_candidates.append((score, candidate))
		scores_with_candidates.sort()
		best_new_score, best_candidate = scores_with_candidates.pop()
		if current_score < best_new_score:
			features.drop([best_candidate], axis=1)
			selected.append(best_candidate)
			current_score = best_new_score

	print selected
read_glass_csv()
read_iris_csv()
move_id_from_glass_dataframe()

k_means(iris_data_set, 3)
# stepwise_forward_feature_selection(iris_data_set)