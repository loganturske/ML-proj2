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

	for x,y in zip(arr1[:-2],arr2[:-2]):
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
	i = 0
	for row in old.values:
		arr1 = row[:-2]
		arr2 = new.values[i][:-2]
		for x,y in zip(arr1, arr2):
			diff = subtract_points(x, y)
			diff = get_absolute_value(diff)
			if diff > .5:
				change = True
		i += 1
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
	print "START"
	new_centroids = pandas.DataFrame()	
	i = 0
	for index,cent in centroids.iterrows():
		clusters = df.loc[df['cluster'] == i]
		centroid = clusters.mean(skipna=True)
		new_centroids = new_centroids.append(centroid, ignore_index=True)
		print new_centroids
		i += 1
	print "END"
	new_centroids['Class'] = 0
	print "QWERQERQWERQWER"
	print new_centroids
	return new_centroids

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
	for c in [0,1,2]:
		me = point[1]['cluster']
		
		total = 0
		num = 0
		for row in df.iterrows():

			if row[1]['cluster'] != me:
				print "ASDFASDFADF"
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
	# print "###"
	# print top
	bottom = get_max_of(a, b)
	# print bottom
	return divide_a_by_b(top, bottom)

def overall_silhouette_coefficient(df):
	total = 0
	num = 0
	for row in df.iterrows():
		total += silhouette_coefficient(row, df)
		num += 1

	return divide_a_by_b(total, num)

def k_means(df,k):

	# centroids = give_initial_random_k_means(k, get_num_columns(d)-1, 2)
	df['cluster'] = 1
	centroids = df.sample(n=k)
	centroid_change = True
	new_centroids = None
	print centroids
	while centroid_change:
		print "BEGIN"
		for index,row in df.iterrows():
			df = df.set_value(index, col='cluster', value=find_closest_centroid(row, centroids))
		print "ASDF"
		new_centroids = recalculate_new_centroids(df, centroids)
		centroid_change = change_in_centroids(centroids, new_centroids)
		centroids = new_centroids

	print centroids
	# a = overall_silhouette_coefficient(df)
	# print a
	# print centroids
	# return a

def stepwise_forward_feature_selection(data):
	features = data.columns[:-1]
	response = data.columns[-1]
	f_sub_0 = []
	basePref = -100000

	while features.empty != True:
		bestPerf = -100000
		bestF = features[1]
		for candidate in features:
			f_sub_0.append(candidate)
			full = f_sub_0 + [response]
			print full
			currPerf = k_means(data[full],3)
			print currPerf
			if currPerf > bestPerf:
				bestPerf = currPerf
				bestF = candidate
			f_sub_0.remove(candidate)
		if bestPerf > basePref:
			basePref = bestPerf
			features.drop([bestF])
			f_sub_0.append(bestF)
		else:
			break
	print bestPerf
	print f_sub_0
read_glass_csv()
read_iris_csv()
move_id_from_glass_dataframe()

k_means(iris_data_set,3)
# stepwise_forward_feature_selection(iris_data_set)
# stepwise_forward_feature_selection(iris_data_set)